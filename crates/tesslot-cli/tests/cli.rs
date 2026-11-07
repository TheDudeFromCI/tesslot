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
    assert_eq!(v["weight_loading"], "unimplemented");
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
