//! Test the actual CLI argument boundary, including global config/name collisions.
#[test]
fn qwen_inspection_does_not_read_model_metadata_as_server_config() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/qwen3-0.6b.config.json");
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"))
        .arg("inspect-qwen-config")
        .arg(path)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(v["hidden_size"], 1024);
    assert_eq!(v["query_width"], 2048);
    assert_eq!(v["weight_loading"], "local_single_file_or_sharded");
}

#[test]
fn native_tokenizer_command_derives_export_identity() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/demo-tokenizer.json");
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"))
        .arg("inspect-tokenizer")
        .arg(path)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let actual: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("../../../fixtures/tokenizer-identity.json")).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn weight_inspection_reports_real_shapes_and_invalid_mapping_exits_nonzero() {
    let folder = tempfile::tempdir().unwrap();
    let weights = folder.path().join("tiny.safetensors");
    let bytes = tesslot_artifacts::encode_f32(&std::collections::BTreeMap::from([(
        "weight".into(),
        (vec![1, 2], vec![1.0, 2.0]),
    )]))
    .unwrap();
    std::fs::write(&weights, bytes).unwrap();
    let run = |mapping: Option<&std::path::Path>| {
        let mut c = std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"));
        c.arg("inspect-weights").arg(&weights);
        if let Some(path) = mapping {
            c.arg("--mapping").arg(path);
        }
        c.output().unwrap()
    };
    let output = run(None);
    assert!(output.status.success());
    let v: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(v["tensors"]["weight"]["shape"], serde_json::json!([1, 2]));
    let mapping = folder.path().join("mapping.json");
    std::fs::write(
        &mapping,
        br#"{"expected":{"dest":[1,3]},"sources":{"dest":"weight"},"initialize":[]}"#,
    )
    .unwrap();
    let bad = run(Some(&mapping));
    assert!(!bad.status.success());
    let v: serde_json::Value = serde_json::from_slice(&bad.stdout).unwrap();
    assert!(!v["incompatible"].as_array().unwrap().is_empty());
}

#[test]
fn demo_qwen_converts_to_a_new_directory_and_preserves_source_tensors() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tiny-qwen");
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("kidra");
    let run = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"))
            .arg("convert-qwen")
            .arg("--model-config")
            .arg(root.join("config.json"))
            .arg("--weights")
            .arg(root.join("model.safetensors"))
            .arg("--output")
            .arg(&out)
            .output()
            .unwrap()
    };
    let result = run();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("conversion.json")).unwrap()).unwrap();
    assert_eq!(
        manifest["report"]["compatible"].as_object().unwrap().len(),
        13
    );
    let mut file = std::fs::File::open(out.join("model.safetensors")).unwrap();
    let inventory = tesslot_artifacts::inspect(&mut file).unwrap();
    assert_eq!(inventory.tensors.len(), 20);
    assert!(!run().status.success());
    assert!(out.join("conversion.json").exists());
}

#[test]
#[ignore = "opt-in actual Qwen/Remix CLI training, multi-sample cursor and checkpoint resume"]
fn native_remix_training_cli_resumes_without_resetting_sample_cursor() {
    use std::process::Command;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let temp = tempfile::tempdir().unwrap();
    let export = temp.path().join("train.jsonl");
    let data = std::fs::read_to_string(root.join("fixtures/autograd.jsonl")).unwrap()
        + &std::fs::read_to_string(root.join("fixtures/canonical.jsonl")).unwrap();
    std::fs::write(&export, &data).unwrap();
    let first = temp.path().join("first");
    let resumed = temp.path().join("resumed");
    let uninterrupted = temp.path().join("uninterrupted");
    let run = |export: &std::path::Path,
               out: &std::path::Path,
               steps: &str,
               resume: Option<&std::path::Path>| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_tesslot"));
        command
            .arg("train-remix-demo")
            .arg(export)
            .arg("--tokenizer-file")
            .arg(root.join("fixtures/demo-tokenizer.json"))
            .arg("--output")
            .arg(out)
            .arg("--steps")
            .arg(steps);
        if let Some(path) = resume {
            command.arg("--resume").arg(path);
        }
        command.output().unwrap()
    };
    let a = run(&export, &first, "2", None);
    assert!(a.status.success(), "{}", String::from_utf8_lossy(&a.stderr));
    let b = run(&export, &resumed, "1", Some(&first));
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
    let events = String::from_utf8(b.stdout).unwrap();
    let event: serde_json::Value = serde_json::from_str(events.lines().next().unwrap()).unwrap();
    assert_eq!(event["sample_id"], "autograd-tiny");
    assert_eq!(event["metrics"]["step"], 3);
    let c = run(&export, &uninterrupted, "3", None);
    assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
    for file in [
        "model.safetensors",
        "optimizer.safetensors",
        "manifest.json",
    ] {
        assert_eq!(
            std::fs::read(resumed.join(file)).unwrap(),
            std::fs::read(uninterrupted.join(file)).unwrap()
        );
    }
    let again = run(&export, &first, "1", None);
    assert!(!again.status.success());
    std::fs::write(&export, format!("{data}\n")).unwrap();
    let bad = run(&export, &temp.path().join("changed"), "1", Some(&first));
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("dataset hash"));
    let invalid = temp.path().join("invalid.jsonl");
    std::fs::write(
        &invalid,
        std::fs::read_to_string(root.join("fixtures/autograd.jsonl")).unwrap() + "{broken}\n",
    )
    .unwrap();
    let out = temp.path().join("invalid-checkpoint");
    let bad = run(&invalid, &out, "1", None);
    assert!(!bad.status.success());
    assert!(
        bad.stdout.is_empty(),
        "invalid suffix must be rejected before any optimizer update"
    );
    assert!(!out.exists());
}

#[test]
fn complete_pretrained_conversion_is_loadable_and_checksums_fail_closed() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("base");
    std::fs::create_dir(&source).unwrap();
    for (name, bytes) in [
        (
            "config.json",
            include_bytes!("../../../fixtures/tiny-qwen/config.json").as_slice(),
        ),
        (
            "model.safetensors",
            include_bytes!("../../../fixtures/tiny-qwen/model.safetensors").as_slice(),
        ),
        (
            "tokenizer.json",
            include_bytes!("../../../fixtures/demo-tokenizer.json").as_slice(),
        ),
    ] {
        std::fs::write(source.join(name), bytes).unwrap();
    }
    let out = temp.path().join("kidra");
    let run = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"))
            .arg("convert-pretrained")
            .arg("--input")
            .arg(&source)
            .arg("--output")
            .arg(&out)
            .arg("--windows")
            .arg("7")
            .output()
            .unwrap()
    };
    let result = run();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!run().status.success());
    let artifact = tesslot_artifacts::pretrained::load_converted(&out, 128 * 1024).unwrap();
    assert_eq!(artifact.config.vocab_size, 131);
    assert_eq!(artifact.tokenizer.special_tokens.noop, 130);
    assert_eq!(
        &artifact.weights["model.embed_tokens.weight"].1[..128 * 16],
        &tesslot_artifacts::read_f32(
            &mut std::io::Cursor::new(include_bytes!(
                "../../../fixtures/tiny-qwen/model.safetensors"
            )),
            "model.embed_tokens.weight",
            128 * 1024
        )
        .unwrap()
        .1
    );
    let model = tesslot_tensor::qwen::Model::from_pretrained(
        &out,
        true,
        tesslot_tensor::qwen::runtime::Plan::cpu(1, 128 * 1024),
        128 * 1024,
    )
    .unwrap();
    assert_eq!(
        model.remix_state().unwrap().tokenizer.sha256,
        artifact.tokenizer.sha256
    );
    let checkpoint = temp.path().join("checkpoint");
    model.save(&checkpoint).unwrap();
    let restored = tesslot_tensor::qwen::Model::restore(&checkpoint, true).unwrap();
    assert_eq!(
        restored.remix_state().unwrap().tokenizer.sha256,
        artifact.tokenizer.sha256
    );
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("config.json")).unwrap()).unwrap();
    config["rope_theta"] = serde_json::json!(10000);
    std::fs::write(
        out.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    assert!(tesslot_artifacts::pretrained::load_converted(&out, 128 * 1024).is_err());
    std::fs::write(
        out.join("config.json"),
        serde_json::to_vec(&artifact.config).unwrap(),
    )
    .unwrap();
    std::fs::write(out.join("model.safetensors"), b"corrupt").unwrap();
    assert!(
        tesslot_artifacts::pretrained::load_converted(&out, 128 * 1024)
            .err()
            .unwrap()
            .to_string()
            .contains("checksum")
    );
}
#[test]
#[ignore = "opt-in user-supplied HF checkpoint forward CLI (tiny fixture only, no downloads)"]
fn local_checkpoint_forward_command_executes_actual_weights() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tiny-qwen");
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"))
        .arg("forward-qwen")
        .arg("--input")
        .arg(root)
        .arg("--tokens")
        .arg("0,97,98")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(v["logits_shape"], serde_json::json!([3, 128]));
    assert!(v["next_token_id"].as_u64().unwrap() < 128);
}

#[test]
#[ignore = "opt-in real exporter + complete pretrained conversion + training + checkpoint resume"]
fn local_pretrained_remix_cli_trains_and_resumes_with_extended_controls() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("base");
    std::fs::create_dir(&base).unwrap();
    for (name, bytes) in [
        (
            "config.json",
            include_bytes!("../../../fixtures/tiny-qwen/config.json").as_slice(),
        ),
        (
            "model.safetensors",
            include_bytes!("../../../fixtures/tiny-qwen/model.safetensors").as_slice(),
        ),
        (
            "tokenizer.json",
            include_bytes!("../../../fixtures/demo-tokenizer.json").as_slice(),
        ),
    ] {
        std::fs::write(base.join(name), bytes).unwrap();
    }
    let converted = temp.path().join("kidra");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"))
        .arg("convert-pretrained")
        .arg("--input")
        .arg(&base)
        .arg("--output")
        .arg(&converted)
        .arg("--windows")
        .arg("7")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let run = |destination: &str, steps: usize, resume: bool| {
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"));
        command
            .arg("train-remix-local")
            .arg(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../fixtures/pretrained-autograd.jsonl"),
            )
            .arg("--tokenizer-file")
            .arg(converted.join("tokenizer.json"))
            .arg("--output")
            .arg(temp.path().join(destination))
            .arg("--steps")
            .arg(steps.to_string())
            .arg("--recompute");
        if resume {
            command.arg("--resume").arg(temp.path().join("first"));
        } else {
            command.arg("--pretrained").arg(&converted);
        }
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let events = String::from_utf8(result.stdout)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            events
                .iter()
                .filter(|e| e["event"] == "optimizer_updated")
                .count(),
            steps
        );
    };
    run("first", 2, false);
    run("resumed", 2, true);
    run("full", 4, false);
    for name in ["model.safetensors", "optimizer.safetensors"] {
        assert_eq!(
            std::fs::read(temp.path().join("resumed").join(name)).unwrap(),
            std::fs::read(temp.path().join("full").join(name)).unwrap()
        );
    }
    let first = tesslot_tensor::qwen::Model::restore(&temp.path().join("first"), true).unwrap();
    let resumed = tesslot_tensor::qwen::Model::restore(&temp.path().join("resumed"), true).unwrap();
    assert_eq!(first.step, 2);
    assert_eq!(resumed.step, 4);
    assert_eq!(resumed.remix_state().unwrap().completed_samples, 4);
}
