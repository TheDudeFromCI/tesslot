//! Numerical optimizer/objective contracts on the actual multi-chunk export fixture.
use tesslot_core::ModelConfig;
use tesslot_dataset::{Limits, Reader, Sample};
use tesslot_tensor::{Model, Parameters};
use tesslot_train::{Config, Optimizer, Trainer};

fn sample() -> Sample {
    Reader::new(
        std::io::Cursor::new(include_str!("../../../fixtures/autograd.jsonl")),
        serde_json::from_str(include_str!("../../../fixtures/tokenizer-identity.json")).unwrap(),
        Limits::default(),
    )
    .unwrap()
    .next()
    .unwrap()
    .unwrap()
}
fn model() -> ModelConfig {
    ModelConfig {
        hidden_size: 2,
        layers: 2,
        max_windows: 7,
        ..ModelConfig::default()
    }
}
fn config() -> Config {
    Config {
        optimizer: Optimizer::Sgd,
        clip_norm: None,
        accumulation_samples: 2,
        ..Config::default()
    }
}
fn derivative(parameters: &Parameters, sample: &Sample) -> (Parameters, f64) {
    let model = Model::new(model(), parameters, false).unwrap();
    let predictions = model
        .forward(&sample.rows, &sample.metadata.windows)
        .unwrap();
    let (numerator, weight) = model.loss_parts(&predictions, &sample.rows).unwrap();
    (model.gradients(&numerator).unwrap(), weight)
}
fn close(a: &Parameters, b: &Parameters, tolerance: f64) {
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
    for (name, matrix) in a {
        for (x, y) in matrix.iter().flatten().zip(b[name].iter().flatten()) {
            assert!(
                (x - y).abs() <= tolerance * (1.0 + y.abs()),
                "{name}: {x} vs {y}"
            );
        }
    }
}

#[test]
fn weighted_accumulation_is_one_normalized_full_parameter_sgd_update() {
    let first = sample();
    let mut second = first.clone();
    // Different context and fractional weight make averaging sample means incorrect.
    for row in &mut second.rows {
        row.loss_weight *= 0.013;
        if row.loss_weight > 0.0 {
            row.token_id = (row.token_id + 7) % 128;
        }
    }
    let mut trainer = Trainer::new(model(), config()).unwrap();
    let before = trainer.parameters().clone();
    let (a, wa) = derivative(&before, &first);
    let (b, wb) = derivative(&before, &second);
    let rate = config().learning_rate;
    let mut expected = before.clone();
    for (name, matrix) in &mut expected {
        for (i, row) in matrix.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                let g = ((a[name][i][j] + b[name][i][j]) / (wa + wb).max(1.0)) as f32;
                *v = f64::from(*v as f32 - rate * g);
            }
        }
    }
    let step = trainer.train_batch(&[&first, &second]).unwrap();
    assert_eq!(step.metrics.samples, 2);
    assert!((step.metrics.effective_weight - wa - wb).abs() < 1e-12);
    close(trainer.parameters(), &expected, 1e-7);
    // Zero direct-loss context embeddings still receive gradient through the prefix.
    let token = first
        .rows
        .iter()
        .find(|r| r.phase == tesslot_core::Phase::B && r.tick == 0)
        .unwrap()
        .token_id as usize;
    assert_ne!(
        before["tokens"][token],
        trainer.parameters()["tokens"][token]
    );
}

#[test]
fn chunk_boundaries_evaluation_and_recomputation_do_not_change_training() {
    let chunked = sample();
    assert!(chunked.chunk_ends.len() > 1);
    let mut full = chunked.clone();
    full.chunk_ends = vec![full.rows.len()];
    let mut plain = Trainer::new(model(), config()).unwrap();
    let mut recomputed = Trainer::new(
        model(),
        Config {
            recompute: true,
            ..config()
        },
    )
    .unwrap();
    let initial = plain.evaluate(&chunked).unwrap();
    for _ in 0..4 {
        let before = plain.parameters().clone();
        plain.evaluate(&chunked).unwrap();
        close(plain.parameters(), &before, 0.0);
        plain.train_batch(&[&chunked]).unwrap();
        recomputed.train_batch(&[&full]).unwrap();
        close(plain.parameters(), recomputed.parameters(), 2e-6);
    }
    assert!(plain.evaluate(&chunked).unwrap().loss() < initial.loss());
    assert_eq!(plain.steps(), 4);
}

#[test]
fn adamw_matches_independent_bias_corrected_updates_and_seeded_runs() {
    let sample = sample();
    let controls = Config {
        optimizer: Optimizer::AdamW {
            beta1: 0.8,
            beta2: 0.95,
            epsilon: 1e-6,
            weight_decay: 0.1,
        },
        ..config()
    };
    let mut trainer = Trainer::new(model(), controls.clone()).unwrap();
    let mut repeat = Trainer::new(model(), controls.clone()).unwrap();
    let mut expected = trainer.parameters().clone();
    let mut first = expected.clone();
    let mut second = expected.clone();
    for v in first
        .values_mut()
        .chain(second.values_mut())
        .flatten()
        .flatten()
    {
        *v = 0.0;
    }
    for step in 1..=3 {
        let (gradients, weight) = derivative(trainer.parameters(), &sample);
        for (name, matrix) in &mut expected {
            for (i, row) in matrix.iter_mut().enumerate() {
                for (j, v) in row.iter_mut().enumerate() {
                    let g = gradients[name][i][j] / weight.max(1.0);
                    // Independent f64 recurrence checks FP32 moments within declared tolerance.
                    let m = 0.8 * first[name][i][j] + 0.2 * g;
                    let s = 0.95 * second[name][i][j] + 0.05 * g * g;
                    first.get_mut(name).unwrap()[i][j] = m;
                    second.get_mut(name).unwrap()[i][j] = s;
                    *v = *v * (1.0 - f64::from(controls.learning_rate) * 0.1)
                        - f64::from(controls.learning_rate) * (m / (1.0 - 0.8_f64.powi(step)))
                            / ((s / (1.0 - 0.95_f64.powi(step))).sqrt() + 1e-6);
                }
            }
        }
        trainer.train_batch(&[&sample]).unwrap();
        repeat.train_batch(&[&sample]).unwrap();
        close(trainer.parameters(), &expected, 3e-6);
        close(trainer.parameters(), repeat.parameters(), 0.0);
    }
    let changed = Trainer::new(ModelConfig { seed: 7, ..model() }, controls).unwrap();
    assert_ne!(changed.parameters(), trainer.parameters());
}

#[test]
fn global_clipping_and_fractional_weight_normalization_match_manual_sgd() {
    let mut sample = sample();
    for row in &mut sample.rows {
        row.loss_weight *= 0.001;
    }
    let bound = 1e-6;
    let mut trainer = Trainer::new(
        model(),
        Config {
            clip_norm: Some(bound),
            ..config()
        },
    )
    .unwrap();
    let (gradients, weight) = derivative(trainer.parameters(), &sample);
    assert!(weight < 1.0);
    let norm = gradients
        .values()
        .flatten()
        .flatten()
        .fold(0.0_f64, |n, g| n.hypot(*g));
    assert!(norm > bound);
    let mut expected = trainer.parameters().clone();
    for (name, matrix) in &mut expected {
        for (i, row) in matrix.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = f64::from(
                    *v as f32
                        - config().learning_rate * (gradients[name][i][j] * bound / norm) as f32,
                );
            }
        }
    }
    let step = trainer.train_batch(&[&sample]).unwrap();
    assert!((step.gradient_norm - norm).abs() < 1e-10);
    assert!((step.clip_scale - bound / norm).abs() < 1e-10);
    close(trainer.parameters(), &expected, 1e-8);
}

#[test]
fn failures_leave_parameters_moments_and_step_unchanged() {
    let sample = sample();
    let mut trainer = Trainer::new(model(), Config::default()).unwrap();
    let mut control = Trainer::new(model(), Config::default()).unwrap();
    trainer.train_batch(&[&sample]).unwrap();
    control.train_batch(&[&sample]).unwrap();
    for weight in [0.0, f64::NAN, f64::INFINITY, f64::MAX] {
        let mut bad = sample.clone();
        for row in &mut bad.rows {
            row.loss_weight = weight;
        }
        assert!(trainer.train_batch(&[&bad]).is_err());
        assert_eq!(trainer.steps(), 1);
        close(trainer.parameters(), control.parameters(), 0.0);
    }
    let mut bad = sample.clone();
    bad.chunk_ends[0] = 0;
    assert!(trainer.train_batch(&[&bad]).is_err());
    assert!(trainer.train_batch(&[]).is_err());
    trainer.train_batch(&[&sample]).unwrap();
    control.train_batch(&[&sample]).unwrap();
    close(trainer.parameters(), control.parameters(), 0.0);
    // Force a candidate moment overflow after valid forward/backward.
    let mut overflow = Trainer::new(
        model(),
        Config {
            optimizer: Optimizer::AdamW {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: f32::MAX,
            },
            learning_rate: f32::MAX,
            clip_norm: None,
            ..config()
        },
    )
    .unwrap();
    let before = overflow.parameters().clone();
    assert!(
        overflow
            .train_batch(&[&sample])
            .unwrap_err()
            .contains("nonfinite optimizer update")
    );
    assert_eq!(overflow.steps(), 0);
    close(overflow.parameters(), &before, 0.0);
}

#[test]
fn warmup_decay_and_invalid_controls() {
    let config = Config {
        learning_rate: 1.0,
        warmup_steps: 2,
        decay_steps: Some(2),
        ..config()
    };
    assert_eq!(
        (1..=5).map(|step| config.rate(step)).collect::<Vec<_>>(),
        vec![0.5, 1.0, 0.5, 0.0, 0.0]
    );
    let no_warmup = Config {
        warmup_steps: 0,
        ..config.clone()
    };
    assert_eq!(no_warmup.rate(1), 1.0);
    assert_eq!(no_warmup.rate(3), 0.0);
    let invalid = [
        Config {
            accumulation_samples: 0,
            ..config.clone()
        },
        Config {
            learning_rate: f32::NAN,
            ..config.clone()
        },
        Config {
            decay_steps: Some(0),
            ..config.clone()
        },
        Config {
            clip_norm: Some(0.0),
            ..config.clone()
        },
        Config {
            train_basis_points: 10000,
            ..config.clone()
        },
        Config {
            optimizer: Optimizer::AdamW {
                beta1: 1.0,
                beta2: 0.99,
                epsilon: 1e-8,
                weight_decay: 0.0,
            },
            ..config.clone()
        },
    ];
    for controls in invalid {
        assert!(controls.validate().is_err());
    }
    assert!(serde_json::from_str::<Config>(r#"{"typo":1}"#).is_err());
    let mut too_small = Trainer::new(
        model(),
        Config {
            max_sample_rows: 1,
            ..config
        },
    )
    .unwrap();
    assert!(too_small.train_batch(&[&sample()]).is_err());
}

#[test]
fn checkpoints_restore_sgd_and_adam_moments_schedule_and_next_update() {
    use tesslot_train::checkpoint::{self, Progress};
    for optimizer in [Optimizer::Sgd, Config::default().optimizer] {
        let controls = Config {
            epochs: 3,
            warmup_steps: 2,
            decay_steps: Some(4),
            optimizer,
            ..config()
        };
        let mut original = Trainer::new(model(), controls.clone()).unwrap();
        let sample = sample();
        original.train_batch(&[&sample]).unwrap();
        let root = tempfile::tempdir().unwrap();
        let identity =
            serde_json::from_str(include_str!("../../../fixtures/tokenizer-identity.json"))
                .unwrap();
        let progress = Progress {
            completed_epochs: 1,
            dataset_sha256: "a".repeat(64),
            tokenizer: identity,
        };
        let first = checkpoint::save(root.path(), &original, progress.clone()).unwrap();
        let (mut restored, loaded) = checkpoint::resume(
            root.path(),
            &model(),
            &controls,
            &progress.dataset_sha256,
            &progress.tokenizer,
        )
        .unwrap();
        assert_eq!(loaded.completed_epochs, 1);
        close(original.parameters(), restored.parameters(), 0.0);
        for _ in 0..2 {
            let a = original.train_batch(&[&sample]).unwrap();
            let b = restored.train_batch(&[&sample]).unwrap();
            assert_eq!(a.learning_rate, b.learning_rate);
            assert_eq!(a.step, b.step);
            close(original.parameters(), restored.parameters(), 0.0);
        }
        assert!(
            checkpoint::resume(
                root.path(),
                &model(),
                &controls,
                &"b".repeat(64),
                &progress.tokenizer
            )
            .is_err()
        );
        assert!(
            checkpoint::resume(
                root.path(),
                &ModelConfig { seed: 7, ..model() },
                &controls,
                &progress.dataset_sha256,
                &progress.tokenizer
            )
            .is_err()
        );
        let partial = root.path().join(".partial-interrupted");
        std::fs::create_dir(&partial).unwrap();
        std::fs::write(partial.join("state.json"), b"half written").unwrap();
        assert_eq!(checkpoint::load_latest(root.path()).unwrap().0.steps(), 1);
        // A fully published directory without a pointer commit must also be ignored.
        let pointer = std::fs::read(root.path().join("latest")).unwrap();
        checkpoint::save(
            root.path(),
            &original,
            Progress {
                completed_epochs: 3,
                ..progress.clone()
            },
        )
        .unwrap();
        std::fs::write(root.path().join("latest"), &pointer).unwrap();
        assert_eq!(checkpoint::load_latest(root.path()).unwrap().0.steps(), 1);
        assert!(
            checkpoint::save(
                root.path(),
                &original,
                Progress {
                    completed_epochs: 4,
                    ..progress
                }
            )
            .is_err()
        );
        assert_eq!(std::fs::read(root.path().join("latest")).unwrap(), pointer);
        // Even a correctly checksummed malformed snapshot must fail semantic validation.
        let state_path = first.join("state.json");
        let manifest_path = first.join("manifest.json");
        let original_state = std::fs::read(&state_path).unwrap();
        let original_manifest = std::fs::read(&manifest_path).unwrap();
        for (key, replacement) in [
            ("version", serde_json::json!("unsupported/99")),
            ("placement", serde_json::json!("cuda:0")),
            ("rng_scheme", serde_json::json!("unknown")),
            ("first", serde_json::json!({"unknown_parameter": [[1.0]]})),
            ("second", serde_json::json!({"output": [[-1.0]]})),
        ] {
            use sha2::{Digest, Sha256};
            let mut state: serde_json::Value = serde_json::from_slice(&original_state).unwrap();
            state[key] = replacement;
            let bytes = serde_json::to_vec(&state).unwrap();
            let mut manifest: serde_json::Value =
                serde_json::from_slice(&original_manifest).unwrap();
            manifest["files"]["state.json"] = format!("{:x}", Sha256::digest(&bytes)).into();
            std::fs::write(&state_path, bytes).unwrap();
            std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
            assert!(
                checkpoint::load(&first).is_err(),
                "accepted malformed {key}"
            );
        }
        std::fs::write(&state_path, original_state).unwrap();
        std::fs::write(&manifest_path, original_manifest).unwrap();
        // Real payload corruption is caught before restoration.
        let weights = first.join("model.safetensors");
        let mut bytes = std::fs::read(&weights).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(weights, bytes).unwrap();
        assert!(
            checkpoint::load(&first)
                .err()
                .unwrap()
                .to_string()
                .contains("checksum mismatch")
        );
        std::fs::write(root.path().join("latest"), b"../outside").unwrap();
        assert!(checkpoint::load_latest(root.path()).is_err());
    }
}
