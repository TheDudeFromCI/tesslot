//! Explicit opt-in two-window computational/causality experiments.
use tesslot_tensor::qwen::{
    Model,
    echo::{LETTERS, pattern},
};

#[test]
#[ignore = "opt-in two-window echo learning, ablation and checkpoint experiment"]
fn two_window_echo_learning_and_resume() {
    let mut model = Model::echo_demo(true).unwrap();
    let source = pattern(42);
    let first = model.evaluate_echo(&source, true).unwrap();
    for _ in 0..400 {
        let m = model.train_echo_step(&source).unwrap();
        assert!(m.loss.is_finite() && m.gradient_norm > 0.0);
        if m.step.is_multiple_of(100) {
            eprintln!("echo step {} loss {}", m.step, m.loss);
        }
    }
    let last = model.evaluate_echo(&source, true).unwrap();
    let ablated = model.evaluate_echo(&source, false).unwrap();
    let replay = model.replay_echo(&source, true).unwrap();
    let masked = model.replay_echo(&source, false).unwrap();
    let unseen = model.replay_echo(&pattern(43), true).unwrap();
    eprintln!(
        "echo {first} -> {last}; ablated loss {ablated}; source {}; echo {}; accuracy {}; masked accuracy {}; unseen-seed accuracy {}",
        replay.source, replay.echo, replay.accuracy, masked.accuracy, unseen.accuracy
    );
    assert!(last < first * 0.5);
    assert_eq!(replay.accuracy, 1.0);
    assert!(masked.accuracy < replay.accuracy);
    assert!(ablated > last + 0.1, "source ablation must harm loss");
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("echo");
    model.save(&path).unwrap();
    let mut restored = Model::restore(&path, true).unwrap();
    assert!(restored.is_echo());
    assert_eq!(replay, restored.replay_echo(&source, true).unwrap());
    let a = model.train_echo_step(&source).unwrap();
    let b = restored.train_echo_step(&source).unwrap();
    assert!((a.loss - b.loss).abs() < 1e-6);
    assert_eq!(
        model.replay_echo(&source, true).unwrap(),
        restored.replay_echo(&source, true).unwrap()
    );
}

#[test]
#[ignore = "opt-in multilayer parallel-window causal and gradient experiment"]
fn causal_echo_and_source_gradients() {
    for layers in [1, 2] {
        let mut config = tesslot_model::qwen::Config::from_json(include_bytes!(
            "../../../fixtures/tiny-qwen/config.json"
        ))
        .unwrap();
        config.num_hidden_layers = layers;
        let weights =
            tesslot_artifacts::qwen::random_source(&config, 20261106, 128 * 1024).unwrap();
        let bytes = tesslot_artifacts::encode_f32(&weights).unwrap();
        let model = Model::echo_from_base(config, &bytes, 20261107, false).unwrap();
        let source = pattern(42);
        assert_eq!(source.len(), LETTERS);
        assert_eq!(source, pattern(42));
        assert_ne!(source, pattern(43));
        let a = model
            .echo_logits(&source, &source[..LETTERS - 1], LETTERS, true)
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        // Full teacher-forced sequence and incremental prefix recomputation agree.
        for p in 0..LETTERS {
            let prefix = model
                .echo_logits(&source, &source[..p], p + 1, true)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            for (x, y) in a[p * 128..(p + 1) * 128]
                .iter()
                .zip(&prefix[p * 128..(p + 1) * 128])
            {
                assert!(
                    (x - y).abs() < 2e-6,
                    "full/prefix mismatch in layer count {layers}, position {p}"
                );
            }
        }
        let mut changed = source.clone();
        changed[5] = if changed[5] == 97 { 98 } else { 97 };
        let b = model
            .echo_logits(&changed, &changed[..LETTERS - 1], LETTERS, true)
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        // Source position 5 commits at tick 5; echo positions 0..4 decide at ticks 1..5.
        for (x, y) in a[..5 * 128].iter().zip(&b[..5 * 128]) {
            assert!((x - y).abs() < 1e-6);
        }
        assert!(
            a[5 * 128..6 * 128]
                .iter()
                .zip(&b[5 * 128..6 * 128])
                .any(|(x, y)| (x - y).abs() > 1e-7)
        );
        // Changing an echo target at tick 5 cannot alter its own label-free query.
        let mut history = source[..LETTERS - 1].to_vec();
        history[4] = 120;
        let c = model
            .echo_logits(&source, &history, LETTERS, true)
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for (x, y) in a[..5 * 128].iter().zip(&c[..5 * 128]) {
            assert!((x - y).abs() < 1e-6);
        }
        let ablated = model
            .echo_logits(&source, &source[..LETTERS - 1], LETTERS, false)
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(a.iter().zip(ablated).any(|(x, y)| (x - y).abs() > 1e-6));
    }
}
