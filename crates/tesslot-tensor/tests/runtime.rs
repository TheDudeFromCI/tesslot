//! Ordinary metadata tests avoid running full forward/backward experiments.
use tesslot_model::qwen::Config;
use tesslot_tensor::qwen::{
    Model,
    runtime::{Plan, Precision},
};
#[test]
fn placement_budget_and_capability_errors_are_actionable() {
    let mut model = Model::demo(false).unwrap();
    assert!(
        model
            .configure_runtime(Plan::cpu(2, 65536))
            .unwrap_err()
            .contains("one target per layer")
    );
    assert!(
        model
            .configure_runtime(Plan::cpu(1, 1))
            .unwrap_err()
            .contains("budget")
    );
    let mut plan = Plan::cpu(1, 65536);
    plan.precision = Precision::Bf16;
    assert!(
        model
            .configure_runtime(plan)
            .unwrap_err()
            .contains("require CUDA")
    );
    let mut tf32 = Plan::cpu(1, 65536);
    tf32.precision = Precision::Tf32;
    assert!(
        model
            .configure_runtime(tf32)
            .unwrap_err()
            .contains("require CUDA")
    );
    #[cfg(not(feature = "cuda"))]
    {
        let plan = Plan {
            root: tesslot_tensor::qwen::runtime::Target::Cuda { index: 0 },
            layers: vec![tesslot_tensor::qwen::runtime::Target::Cuda { index: 0 }],
            precision: Precision::F32,
            parameter_budget_bytes: 65536,
        };
        assert!(
            model
                .configure_runtime(plan)
                .unwrap_err()
                .contains("compile")
        );
    }
    let mixed = Plan {
        root: tesslot_tensor::qwen::runtime::Target::Cpu,
        layers: vec![tesslot_tensor::qwen::runtime::Target::Cuda { index: 0 }],
        precision: Precision::F32,
        parameter_budget_bytes: 65536,
    };
    assert!(
        model
            .configure_runtime(mixed)
            .unwrap_err()
            .contains("mixed CPU/CUDA")
    );
    assert_eq!(model.estimate().weights, 5184 * 4);
}
#[test]
fn explicit_budget_loader_accepts_dimensions_above_toy_limits() {
    let mut c =
        Config::from_json(include_bytes!("../../../fixtures/tiny-qwen/config.json")).unwrap();
    c.head_dim = 128;
    c.num_attention_heads = 2;
    c.num_key_value_heads = 1;
    c.hidden_size = 80;
    c.intermediate_size = 160;
    c.vocab_size = 300;
    c.max_position_embeddings = 32768;
    let weights = tesslot_artifacts::qwen::random_source(&c, 42, 1024 * 1024).unwrap();
    let model = Model::from_tensors(c, weights, None, false, Plan::cpu(1, 1024 * 1024)).unwrap();
    assert!(model.estimate().weights > 65536);
    let temp = tempfile::tempdir().unwrap();
    assert!(
        model
            .save(&temp.path().join("unbound"))
            .unwrap_err()
            .contains("tokenizer identity")
    );
}

#[test]
#[ignore = "opt-in forward, causality and all-parameter backward above the tiny demo dimensions"]
fn head_128_and_nonstandard_projection_width_execute_and_train() {
    let mut config =
        Config::from_json(include_bytes!("../../../fixtures/tiny-qwen/config.json")).unwrap();
    config.head_dim = 128;
    config.hidden_size = 80;
    config.num_attention_heads = 2;
    config.num_key_value_heads = 1;
    config.intermediate_size = 160;
    config.vocab_size = 300;
    config.max_position_embeddings = 32768;
    let weights = tesslot_artifacts::qwen::random_source(&config, 42, 1024 * 1024).unwrap();
    let mut model =
        Model::from_tensors(config, weights, None, true, Plan::cpu(1, 1024 * 1024)).unwrap();
    let values = |x: burn_tensor::Tensor<2>| x.into_data().try_to_vec::<f32>().unwrap();
    let a = values(model.forward(&[0, 97]).unwrap().slice([0..1, 0..300]));
    let b = values(model.forward(&[0, 98]).unwrap().slice([0..1, 0..300]));
    assert_eq!(a, b);
    let initial = model.evaluate(&[0, 97], &[97, 98]).unwrap();
    model.learning_rate = 0.02;
    let metrics = model.train_step(&[0, 97], &[97, 98]).unwrap();
    assert!(metrics.gradient_norm.is_finite() && metrics.gradient_norm > 0.0);
    assert!(model.evaluate(&[0, 97], &[97, 98]).unwrap() < initial);
}
