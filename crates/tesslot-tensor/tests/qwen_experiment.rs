//! Computational experiments are deliberately excluded from ordinary CI.
use tesslot_tensor::qwen::Model;
#[test]
#[ignore = "opt-in Qwen generation, SGD and checkpoint experiment"]
fn train_abc_generate_checkpoint_resume_and_detect_corruption() {
    let mut model = Model::demo(true).unwrap();
    let text = b"a b c a b c a b c a b c ";
    let ids: Vec<_> = text.iter().map(|v| u32::from(*v)).collect();
    let (inputs, targets) = (&ids[..ids.len() - 1], &ids[1..]);
    let first = model.evaluate(inputs, targets).unwrap();
    for _ in 0..60 {
        let m = model.train_step(inputs, targets).unwrap();
        assert!(m.gradient_norm > 0.0 && m.gradient_norm.is_finite());
    }
    let last = model.evaluate(inputs, targets).unwrap();
    eprintln!("actual tiny Qwen loss: {first} -> {last}");
    assert!(last < first * 0.65);
    let tokens = model.generate(&[97, 32], 12).unwrap();
    assert!(!tokens.is_empty());
    assert!(tokens.iter().all(|t| *t < 128));
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("checkpoint");
    model.save(&path).unwrap();
    assert!(model.save(&path).is_err());
    let mut restored = Model::restore(&path, true).unwrap();
    assert_eq!(restored.step, model.step);
    assert_eq!(restored.generate(&[97, 32], 12).unwrap(), tokens);
    assert!((restored.evaluate(inputs, targets).unwrap() - last).abs() < 1e-6);
    let a = model.train_step(inputs, targets).unwrap();
    let b = restored.train_step(inputs, targets).unwrap();
    assert!((a.loss - b.loss).abs() < 1e-6);
    assert_eq!(
        model.generate(&[97, 32], 8).unwrap(),
        restored.generate(&[97, 32], 8).unwrap()
    );
    let file = path.join("model.safetensors");
    let mut bytes = std::fs::read(&file).unwrap();
    let n = bytes.len();
    bytes[n - 1] ^= 1;
    std::fs::write(file, bytes).unwrap();
    assert!(Model::restore(&path, false).is_err());
}
#[test]
#[ignore = "opt-in executable Qwen causal-mask experiment"]
fn future_tokens_cannot_change_prefix_logits() {
    let m = Model::demo(false).unwrap();
    let a = m.forward(&[97, 32, 98, 32, 99]).unwrap();
    let b = m.forward(&[97, 32, 120, 121, 122]).unwrap();
    let x = a
        .slice([0..2, 0..128])
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let y = b
        .slice([0..2, 0..128])
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    for (a, b) in x.iter().zip(y) {
        assert!((a - b).abs() < 1e-6);
    }
}
