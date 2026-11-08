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
    std::fs::remove_file(folder.path().join("shard-0.safetensors")).unwrap();
    assert!(!run().status.success());
}
