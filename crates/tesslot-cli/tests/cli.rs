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

#[test]
fn tiny_training_cli_splits_flushes_tails_evaluates_and_exports_repeatable_weights() {
    use tesslot_dataset::{Limits, Reader, Split};
    let folder = tempfile::tempdir().unwrap();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let export = folder.path().join("training.jsonl");
    let model_config = folder.path().join("model.json");
    let training_config = folder.path().join("training.json");
    let training = tesslot_train::Config {
        epochs: 2,
        accumulation_samples: 2,
        train_basis_points: 5000,
        validation_basis_points: 2500,
        ..tesslot_train::Config::default()
    };
    let config = tesslot_core::Config {
        model: tesslot_core::ModelConfig {
            hidden_size: 2,
            ..tesslot_core::ModelConfig::default()
        },
        ..tesslot_core::Config::default()
    };
    std::fs::write(&model_config, serde_json::to_vec(&config).unwrap()).unwrap();
    std::fs::write(&training_config, serde_json::to_vec(&training).unwrap()).unwrap();
    let chunks: Vec<serde_json::Value> = include_str!("../../../fixtures/autograd.jsonl")
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let mut selected = [0; 3];
    let mut data = String::new();
    for i in 0..100 {
        let mut candidate = String::new();
        for chunk in &chunks {
            let mut c = chunk.clone();
            c["trajectory_uuid"] = format!("{i:08x}-1234-5678-1234-567812345678").into();
            c["sample_id"] = format!("sample-{i}").into();
            if c["chunk_index"] == 0 {
                // Distinct contexts make the seeded sample order affect updates.
                c["rows"][2]["token_id"] = (90 + i % 10).into();
                c["token_ids"][2] = (90 + i % 10).into();
            }
            candidate.push_str(&serde_json::to_string(&c).unwrap());
            candidate.push('\n');
        }
        let s = Reader::new(
            std::io::Cursor::new(&candidate),
            serde_json::from_str(include_str!("../../../fixtures/tokenizer-identity.json"))
                .unwrap(),
            Limits::default(),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
        let index = match s.split(training.seed, 5000, 2500).unwrap() {
            Split::Train => 0,
            Split::Development => 1,
            Split::Test => 2,
        };
        if selected[index] < [3, 1, 1][index] {
            selected[index] += 1;
            data.push_str(&candidate);
        }
        if selected == [3, 1, 1] {
            break;
        }
    }
    assert_eq!(selected, [3, 1, 1]);
    std::fs::write(&export, data).unwrap();
    let run = |output: &std::path::Path| {
        std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"))
            .arg("--config")
            .arg(&model_config)
            .arg("train")
            .arg(&export)
            .arg("--tokenizer")
            .arg(root.join("fixtures/tokenizer-identity.json"))
            .arg("--training-config")
            .arg(&training_config)
            .arg("--output")
            .arg(output)
            .output()
            .unwrap()
    };
    let weights = folder.path().join("model.safetensors");
    let result = run(&weights);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let events: Vec<serde_json::Value> = String::from_utf8(result.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let steps: Vec<_> = events
        .iter()
        .filter(|e| e["event"] == "optimizer_step")
        .collect();
    assert_eq!(steps.len(), 4);
    assert_eq!(
        steps
            .iter()
            .map(|e| e["metrics"]["samples"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [2, 1, 2, 1]
    );
    let completed = events.last().unwrap();
    assert_eq!(
        completed["split_samples"],
        serde_json::json!({"train":3,"validation":1,"test":1})
    );
    assert_eq!(completed["resumable"], false);
    assert_eq!(completed["final_metrics"]["validation"]["samples"], 1);
    assert!(
        completed["final_metrics"]["train"]["weighted_loss"]
            .as_f64()
            .unwrap()
            < events[0]["metrics"]["train"]["weighted_loss"]
                .as_f64()
                .unwrap()
    );
    let inventory =
        tesslot_artifacts::inspect(&mut std::fs::File::open(&weights).unwrap()).unwrap();
    let expected = tesslot_model::ReferenceModel::new(config.model.clone())
        .unwrap()
        .parameters();
    assert_eq!(inventory.tensors.len(), expected.len());
    let mut reloaded = tesslot_tensor::Parameters::new();
    let mut file = std::fs::File::open(&weights).unwrap();
    for (name, matrix) in &expected {
        let (shape, values) = tesslot_artifacts::read_f32(&mut file, name, 1024 * 1024).unwrap();
        assert_eq!(shape, [matrix.len(), matrix[0].len()]);
        reloaded.insert(
            name.clone(),
            values
                .chunks(shape[1])
                .map(|row| row.iter().map(|v| f64::from(*v)).collect())
                .collect(),
        );
    }
    let loaded =
        tesslot_train::Trainer::from_parameters(config.model, training.clone(), reloaded).unwrap();
    let input = Reader::new(
        std::io::BufReader::new(std::fs::File::open(&export).unwrap()),
        serde_json::from_str(include_str!("../../../fixtures/tokenizer-identity.json")).unwrap(),
        Limits::default(),
    )
    .unwrap();
    let mut measured = tesslot_train::Metrics::default();
    for sample in input {
        let sample = sample.unwrap();
        if sample.split(training.seed, 5000, 2500).unwrap() == Split::Train {
            measured.merge(&loaded.evaluate(&sample).unwrap()).unwrap();
        }
    }
    assert!(
        (measured.loss()
            - completed["final_metrics"]["train"]["weighted_loss"]
                .as_f64()
                .unwrap())
        .abs()
            < 1e-6
    );
    let repeat = folder.path().join("repeat.safetensors");
    assert!(run(&repeat).status.success());
    assert_eq!(
        std::fs::read(&weights).unwrap(),
        std::fs::read(&repeat).unwrap()
    );
    // Multi-sample shuffle and accumulation tails also resume exactly across epochs.
    let checkpoint_dir = folder.path().join("checkpoints");
    let resumed_weights = folder.path().join("resumed.safetensors");
    for resume in [false, true] {
        let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"));
        cmd.arg("--config")
            .arg(&model_config)
            .arg("train")
            .arg(&export)
            .arg("--tokenizer")
            .arg(root.join("fixtures/tokenizer-identity.json"))
            .arg("--training-config")
            .arg(&training_config)
            .arg("--output")
            .arg(&resumed_weights)
            .arg("--checkpoint-dir")
            .arg(&checkpoint_dir);
        if resume {
            cmd.arg("--resume");
        } else {
            cmd.arg("--stop-after-epochs").arg("1");
        }
        let result = cmd.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    assert_eq!(
        std::fs::read(&weights).unwrap(),
        std::fs::read(&resumed_weights).unwrap()
    );
    // Existing output fails before any step, preserving its bytes.
    let existing = run(&weights);
    assert!(!existing.status.success());
    assert!(existing.stdout.is_empty());
    assert_eq!(
        std::fs::read(&weights).unwrap(),
        std::fs::read(&repeat).unwrap()
    );
    // A malformed late record is rejected during whole-export preflight.
    use std::io::Write;
    writeln!(
        std::fs::OpenOptions::new()
            .append(true)
            .open(&export)
            .unwrap(),
        "{{}}"
    )
    .unwrap();
    let bad_output = folder.path().join("bad.safetensors");
    let bad = run(&bad_output);
    assert!(!bad.status.success());
    assert!(bad.stdout.is_empty());
    assert!(!bad_output.exists());
}

#[test]
fn checkpoint_cli_resume_matches_uninterrupted_and_recovers_after_process_kill() {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    let folder = tempfile::tempdir().unwrap();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let controls = folder.path().join("training.json");
    std::fs::write(
        &controls,
        serde_json::to_vec(&tesslot_train::Config {
            epochs: 4,
            train_basis_points: 10000,
            validation_basis_points: 0,
            warmup_steps: 2,
            decay_steps: Some(4),
            ..Default::default()
        })
        .unwrap(),
    )
    .unwrap();
    let command = |output: &std::path::Path| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_tesslot"));
        cmd.arg("train")
            .arg(root.join("fixtures/autograd.jsonl"))
            .arg("--tokenizer")
            .arg(root.join("fixtures/tokenizer-identity.json"))
            .arg("--training-config")
            .arg(&controls)
            .arg("--output")
            .arg(output);
        cmd
    };
    let continuous = folder.path().join("continuous.safetensors");
    assert!(command(&continuous).output().unwrap().status.success());
    for kill in [false, true] {
        let checkpoints = folder.path().join(if kill { "killed" } else { "stopped" });
        let output = folder.path().join(if kill {
            "killed.safetensors"
        } else {
            "stopped.safetensors"
        });
        let mut cmd = command(&output);
        cmd.arg("--checkpoint-dir").arg(&checkpoints);
        if kill {
            let mut child = cmd
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .unwrap();
            let mut reader = BufReader::new(child.stdout.take().unwrap());
            let mut line = String::new();
            loop {
                line.clear();
                assert!(
                    reader.read_line(&mut line).unwrap() > 0,
                    "child exited before checkpoint"
                );
                let event: serde_json::Value = serde_json::from_str(&line).unwrap();
                if event["event"] == "checkpoint_saved" {
                    break;
                }
            }
            child.kill().unwrap();
            assert!(!child.wait().unwrap().success());
        } else {
            let stopped = cmd.arg("--stop-after-epochs").arg("1").output().unwrap();
            assert!(
                stopped.status.success(),
                "{}",
                String::from_utf8_lossy(&stopped.stderr)
            );
            assert!(!output.exists());
        }
        // A different optimizer config is rejected before resuming any update.
        let mismatched = folder.path().join("mismatch.json");
        let mut changed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&controls).unwrap()).unwrap();
        changed["learning_rate"] = 0.1.into();
        std::fs::write(&mismatched, serde_json::to_vec(&changed).unwrap()).unwrap();
        let rejected = Command::new(env!("CARGO_BIN_EXE_tesslot"))
            .arg("train")
            .arg(root.join("fixtures/autograd.jsonl"))
            .arg("--tokenizer")
            .arg(root.join("fixtures/tokenizer-identity.json"))
            .arg("--training-config")
            .arg(mismatched)
            .arg("--output")
            .arg(&output)
            .arg("--checkpoint-dir")
            .arg(&checkpoints)
            .arg("--resume")
            .output()
            .unwrap();
        assert!(!rejected.status.success());
        assert!(!output.exists());
        let resumed = command(&output)
            .arg("--checkpoint-dir")
            .arg(&checkpoints)
            .arg("--resume")
            .output()
            .unwrap();
        assert!(
            resumed.status.success(),
            "{}",
            String::from_utf8_lossy(&resumed.stderr)
        );
        assert_eq!(
            std::fs::read(&continuous).unwrap(),
            std::fs::read(output).unwrap()
        );
    }
}

#[test]
fn preflight_reports_insufficient_budgets_and_checks_local_source_headers() {
    use tesslot_train::feasibility::{self, Scenario};
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"))
        .arg("preflight-training")
        .arg(root.join("examples/qwen3-0.6b.config.json"))
        .arg("--scenario")
        .arg(root.join("examples/preflight-0.6b.json"))
        .output()
        .unwrap();
    assert!(!result.status.success());
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["memory"]["base_parameters"], 596_049_920);
    assert_eq!(report["memory"]["training_ready"], false);
    assert_eq!(
        report["memory"]["devices"][0]["status"],
        "insufficient_budget"
    );
    assert_eq!(report["hardware_measured"], false);
    let folder = tempfile::tempdir().unwrap();
    let mut model = tesslot_model::qwen::Config::from_json(include_bytes!(
        "../../../examples/qwen3-0.6b.config.json"
    ))
    .unwrap();
    model.hidden_size = 4;
    model.intermediate_size = 8;
    model.head_dim = 2;
    model.num_attention_heads = 2;
    model.num_key_value_heads = 1;
    model.num_hidden_layers = 2;
    model.vocab_size = 16;
    model.bos_token_id = 1;
    model.eos_token_id = 2;
    let mut scenario: Scenario =
        serde_json::from_slice(include_bytes!("../../../examples/preflight-0.6b.json")).unwrap();
    scenario.layers[0].end = 2;
    scenario.devices[0].margin_bytes = 0;
    let config_path = folder.path().join("config.json");
    let scenario_path = folder.path().join("scenario.json");
    let weights = folder.path().join("model.safetensors");
    std::fs::write(&config_path, serde_json::to_vec(&model).unwrap()).unwrap();
    std::fs::write(&scenario_path, serde_json::to_vec(&scenario).unwrap()).unwrap();
    let shapes = feasibility::parameter_shapes(&model).unwrap();
    let mut tensors = shapes
        .into_iter()
        .map(|(n, shape)| {
            let len = shape.iter().product();
            (n, (shape, vec![0.25; len]))
        })
        .collect();
    std::fs::write(&weights, tesslot_artifacts::encode_f32(&tensors).unwrap()).unwrap();
    let run = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"))
            .arg("preflight-training")
            .arg(&config_path)
            .arg("--scenario")
            .arg(&scenario_path)
            .arg("--weights")
            .arg(&weights)
            .output()
            .unwrap()
    };
    let good = run();
    assert!(
        good.status.success(),
        "{}",
        String::from_utf8_lossy(&good.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&good.stdout).unwrap();
    assert_eq!(
        report["memory"]["devices"][0]["status"],
        "incomplete_estimate"
    );
    assert_eq!(report["memory"]["training_ready"], false);
    assert_eq!(report["weights_payload_verified"], false);
    assert!(
        report["source_mapping"]["missing"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    tensors.remove("model.layers.0.self_attn.q_norm.weight");
    tensors.insert("unexpected.weight".into(), (vec![1], vec![0.0]));
    std::fs::write(&weights, tesslot_artifacts::encode_f32(&tensors).unwrap()).unwrap();
    assert!(!run().status.success());
}

#[test]
fn local_model_directory_command_validates_real_shard_headers_against_qwen_shapes() {
    let folder = tempfile::tempdir().unwrap();
    let mut model = tesslot_model::qwen::Config::from_json(include_bytes!(
        "../../../examples/qwen3-0.6b.config.json"
    ))
    .unwrap();
    model.hidden_size = 4;
    model.intermediate_size = 8;
    model.head_dim = 2;
    model.num_attention_heads = 2;
    model.num_key_value_heads = 1;
    model.num_hidden_layers = 2;
    model.vocab_size = 16;
    model.bos_token_id = 1;
    model.eos_token_id = 2;
    std::fs::write(
        folder.path().join("config.json"),
        serde_json::to_vec(&model).unwrap(),
    )
    .unwrap();
    let shapes = tesslot_train::feasibility::parameter_shapes(&model).unwrap();
    let mut shards = [
        std::collections::BTreeMap::new(),
        std::collections::BTreeMap::new(),
    ];
    let mut weight_map = std::collections::BTreeMap::new();
    let mut bytes = 0;
    for (i, (name, shape)) in shapes.into_iter().enumerate() {
        let count: usize = shape.iter().product();
        bytes += count * 4;
        weight_map.insert(name.clone(), format!("shard-{}.safetensors", i % 2));
        shards[i % 2].insert(name, (shape, vec![0.25; count]));
    }
    for (i, shard) in shards.iter().enumerate() {
        std::fs::write(
            folder.path().join(format!("shard-{i}.safetensors")),
            tesslot_artifacts::encode_f32(shard).unwrap(),
        )
        .unwrap();
    }
    let index = folder.path().join("model.safetensors.index.json");
    std::fs::write(
        &index,
        serde_json::to_vec(
            &serde_json::json!({"metadata":{"total_size":bytes},"weight_map":weight_map}),
        )
        .unwrap(),
    )
    .unwrap();
    let run = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"))
            .arg("inspect-model-directory")
            .arg(folder.path())
            .output()
            .unwrap()
    };
    let good = run();
    assert!(
        good.status.success(),
        "{}",
        String::from_utf8_lossy(&good.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&good.stdout).unwrap();
    assert_eq!(report["weights"]["stored_parameters"], 380);
    assert_eq!(report["weights_payload_verified"], false);
    assert_eq!(report["training_ready"], false);
    // Standalone files use the same config/shape validation without a model directory.
    let standalone = tempfile::tempdir().unwrap();
    let weights = standalone.path().join("qwen base.safetensors");
    let tensors = shards
        .iter()
        .flat_map(|s| s.iter().map(|(n, v)| (n.clone(), v.clone())))
        .collect();
    std::fs::write(&weights, tesslot_artifacts::encode_f32(&tensors).unwrap()).unwrap();
    let inspect = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"))
            .arg("inspect-weights")
            .arg(&weights)
            .arg("--model-config")
            .arg(folder.path().join("config.json"))
            .output()
            .unwrap()
    };
    let result = inspect();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let standalone_report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(
        standalone_report["source_mapping"]["compatible"]
            .as_object()
            .unwrap()
            .len(),
        weight_map.len()
    );
    assert_eq!(standalone_report["weights_payload_verified"], false);
    model.hidden_size = 8;
    std::fs::write(
        folder.path().join("config.json"),
        serde_json::to_vec(&model).unwrap(),
    )
    .unwrap();
    assert!(!inspect().status.success());
    std::fs::remove_file(folder.path().join("shard-0.safetensors")).unwrap();
    assert!(!run().status.success());
}

#[test]
fn backward_only_cli_checks_native_trajectory_and_preserves_converted_files() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("base");
    std::fs::create_dir(&source).unwrap();
    for (name, data) in [
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
        std::fs::write(source.join(name), data).unwrap();
    }
    let converted = temp.path().join("kidra");
    let conversion = std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"))
        .arg("convert-pretrained")
        .arg("--input")
        .arg(&source)
        .arg("--output")
        .arg(&converted)
        .arg("--windows")
        .arg("24")
        .output()
        .unwrap();
    assert!(
        conversion.status.success(),
        "{}",
        String::from_utf8_lossy(&conversion.stderr)
    );
    let before = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "conversion.json",
    ]
    .map(|n| (n, std::fs::read(converted.join(n)).unwrap()));
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let run = |export: &std::path::Path| {
        std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"))
            .arg("backward-remix")
            .arg("--input")
            .arg(&converted)
            .arg("--export")
            .arg(export)
            .output()
            .unwrap()
    };
    let valid = run(&root.join("fixtures/pretrained-backward-tiny/trajectory.jsonl"));
    assert!(
        valid.status.success(),
        "{}",
        String::from_utf8_lossy(&valid.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&valid.stdout).unwrap();
    assert_eq!(report["window_capacity"], 24);
    assert_eq!(report["active_windows"], 2);
    assert_eq!(report["chunks"], 3);
    assert_eq!(report["optimizer_allocated"], false);
    assert_eq!(report["updates"], 0);
    assert_eq!(report["sample_cursor"], 0);
    assert_eq!(report["summary"]["all_gradients_finite"], true);
    assert!(report["summary"]["gradient_norm"].as_f64().unwrap() > 0.0);
    assert_eq!(report["persistent_state_estimate"]["optimizer"], 0);
    assert_eq!(report["persistent_state_estimate"]["update_temporaries"], 0);
    let prefix = std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"))
        .arg("backward-remix")
        .arg("--input")
        .arg(&converted)
        .arg("--export")
        .arg(root.join("fixtures/pretrained-backward-tiny/trajectory.jsonl"))
        .arg("--target-prefix")
        .output()
        .unwrap();
    assert!(
        prefix.status.success(),
        "{}",
        String::from_utf8_lossy(&prefix.stderr)
    );
    let prefix_report: serde_json::Value = serde_json::from_slice(&prefix.stdout).unwrap();
    assert_eq!(prefix_report["target_prefix"], true);
    assert_eq!(prefix_report["updates"], 0);
    assert_eq!(prefix_report["sample_cursor"], 0);
    assert!(
        (prefix_report["summary"]["loss"].as_f64().unwrap()
            - report["summary"]["loss"].as_f64().unwrap())
        .abs()
            < 2e-5
    );
    assert!(
        (prefix_report["summary"]["gradient_norm"].as_f64().unwrap()
            - report["summary"]["gradient_norm"].as_f64().unwrap())
        .abs()
            < 1e-4
    );
    let phases = String::from_utf8_lossy(&prefix.stderr)
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .collect::<Vec<_>>();
    assert!(phases.iter().any(|p| p["phase"] == "forward_complete"));
    assert!(phases.iter().any(|p| p["phase"] == "backward_returned"));
    assert!(phases.iter().any(|p| p["phase"] == "gradient_summary"));

    // The Qwen tokenizer fixture must not be accepted against the tiny converted tokenizer.
    let wrong = run(&root.join("fixtures/pretrained-backward-24/trajectory.jsonl"));
    assert!(!wrong.status.success());
    assert!(wrong.stdout.is_empty());
    let multiple = temp.path().join("two.jsonl");
    let text =
        std::fs::read_to_string(root.join("fixtures/pretrained-backward-tiny/trajectory.jsonl"))
            .unwrap();
    let other = text
        .replace("pretrained-backward-two-window", "different-sample")
        .replace(
            "c0ffee24-1234-5678-1234-567812345678",
            "c0ffee25-1234-5678-1234-567812345678",
        );
    std::fs::write(&multiple, text + &other).unwrap();
    assert!(!run(&multiple).status.success());
    for (name, data) in before {
        assert_eq!(std::fs::read(converted.join(name)).unwrap(), data);
    }
    assert_eq!(std::fs::read_dir(&converted).unwrap().count(), 4);
}
