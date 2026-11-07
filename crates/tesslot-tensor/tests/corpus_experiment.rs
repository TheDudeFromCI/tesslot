//! Computational gates remain explicitly opt-in, including corpus training/evaluation.
use tesslot_tensor::qwen::{Model, corpus::Split};
fn model() -> Model {
    let mut model = Model::echo_demo(true).unwrap();
    model.start_echo_corpus(Default::default()).unwrap();
    model.enable_adamw(Default::default(), 0.003).unwrap();
    model
}
#[test]
#[ignore = "opt-in 600-pattern AdamW training, held-out evaluation and exact resume"]
fn varied_patterns_generalize_and_checkpoint_exactly() {
    let mut model = model();
    let before = model.evaluate_echo_corpus(Split::Development, 16).unwrap();
    for _ in 0..600 {
        let m = model.train_echo_corpus_step().unwrap();
        assert!(m.loss.is_finite() && m.gradient_norm > 0.0);
        if m.step.is_multiple_of(100) {
            eprintln!("corpus step {} loss {}", m.step, m.loss);
        }
    }
    let state = model.echo_corpus().unwrap().clone();
    let dev = model.evaluate_echo_corpus(Split::Development, 16).unwrap();
    let test = model.evaluate_echo_corpus(Split::Test, 16).unwrap();
    eprintln!(
        "initial dev loss {} accuracy {}; final dev loss {} accuracy {} exact {}; test loss {} accuracy {} exact {}; source-masked test loss {} accuracy {}",
        before.loss,
        before.free_running_accuracy,
        dev.loss,
        dev.free_running_accuracy,
        dev.exact_match_rate,
        test.loss,
        test.free_running_accuracy,
        test.exact_match_rate,
        test.source_masked_loss,
        test.source_masked_accuracy
    );
    assert_eq!(
        model.echo_corpus().unwrap(),
        &state,
        "evaluation must not advance sampling"
    );
    assert_eq!(model.step, 600);
    for result in [&dev, &test] {
        assert_eq!(result.target_weight, 192);
        assert!(result.free_running_accuracy >= 0.9, "{result:?}");
        assert!(result.loss < before.loss * 0.2);
        assert!(result.source_masked_accuracy + 0.5 < result.free_running_accuracy);
        assert!(result.source_masked_loss > result.loss + 1.0);
        assert_eq!(result.samples, result.cases.len());
        let prefix = if result.split == Split::Development {
            1u64 << 62
        } else {
            2u64 << 62
        };
        for (i, case) in result.cases.iter().enumerate() {
            assert_eq!(case.seed, (prefix + i as u64).to_string());
        }
    }
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("saved");
    model.save(&path).unwrap();
    let mut restored = Model::restore(&path, true).unwrap();
    assert_eq!(model.echo_corpus(), restored.echo_corpus());
    assert_eq!(
        model.echo_corpus().unwrap().next_seed(),
        restored.echo_corpus().unwrap().next_seed()
    );
    assert_eq!(restored.optimizer_id(), "adamw-f32/1");
    let a = model.train_echo_corpus_step().unwrap();
    let b = restored.train_echo_corpus_step().unwrap();
    assert_eq!(a.loss, b.loss);
    assert_eq!(a.gradient_norm, b.gradient_norm);
    let left = root.path().join("next-a");
    let right = root.path().join("next-b");
    model.save(&left).unwrap();
    restored.save(&right).unwrap();
    for name in [
        "model.safetensors",
        "optimizer.safetensors",
        "manifest.json",
    ] {
        assert_eq!(
            std::fs::read(left.join(name)).unwrap(),
            std::fs::read(right.join(name)).unwrap(),
            "resume mismatch in {name}"
        );
    }
    let moment_file = path.join("optimizer.safetensors");
    let mut data = std::fs::read(&moment_file).unwrap();
    let last = data.len() - 1;
    data[last] ^= 1;
    std::fs::write(moment_file, data).unwrap();
    assert!(matches!(Model::restore(&path,true),Err(e) if e.contains("checksum")));
}
#[test]
#[ignore = "opt-in fresh optimizer checkpoint and rejected corpus state"]
fn fresh_resume_and_invalid_sampling_state() {
    let model = model();
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("zero");
    model.save(&path).unwrap();
    let mut restored = Model::restore(&path, true).unwrap();
    assert_eq!(restored.echo_corpus(), model.echo_corpus());
    assert!(restored.evaluate_echo_corpus(Split::Train, 1).is_err());
    assert!(restored.evaluate_echo_corpus(Split::Test, 17).is_err());
    assert!(
        restored
            .train_echo_step(&tesslot_tensor::qwen::echo::pattern(42))
            .is_err()
    );
    let changed = tesslot_tensor::qwen::corpus::Dataset {
        shuffle_seed: 43,
        ..Default::default()
    };
    assert!(restored.start_echo_corpus(changed.clone()).is_err());
    let manifest = path.join("manifest.json");
    let mut json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    json["corpus"]["completed_samples"] = serde_json::json!(1);
    std::fs::write(&manifest, serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(Model::restore(&path, true).is_err());
    json["corpus"]["completed_samples"] = serde_json::json!(0);
    json["corpus"]["dataset"]["format"] = serde_json::json!("future/2");
    std::fs::write(&manifest, serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(Model::restore(&path, true).is_err());
    // Resume across an epoch boundary without replaying or skipping a training seed.
    let mut small = Model::echo_demo(true).unwrap();
    small
        .start_echo_corpus(tesslot_tensor::qwen::corpus::Dataset {
            train_size: 2,
            ..Default::default()
        })
        .unwrap();
    small.enable_adamw(Default::default(), 0.003).unwrap();
    for _ in 0..3 {
        small.train_echo_corpus_step().unwrap();
    }
    let epoch = root.path().join("epoch");
    small.save(&epoch).unwrap();
    let mut resumed = Model::restore(&epoch, true).unwrap();
    assert_eq!(
        small.echo_corpus().unwrap().next_seed(),
        resumed.echo_corpus().unwrap().next_seed()
    );
    assert_eq!(
        small.train_echo_corpus_step().unwrap().loss,
        resumed.train_echo_corpus_step().unwrap().loss
    );
}
