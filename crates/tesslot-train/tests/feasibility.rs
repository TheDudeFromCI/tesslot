//! Analytic parameter/memory checks without allocating tensors or probing CUDA.
use tesslot_model::qwen;
use tesslot_train::feasibility::{self, Device, LayerRange, Optimizer, Precision, Scenario};
fn tiny() -> qwen::Config {
    let mut c = qwen::Config::from_json(include_bytes!("../../../examples/qwen3-0.6b.config.json"))
        .unwrap();
    c.hidden_size = 4;
    c.intermediate_size = 8;
    c.head_dim = 2;
    c.num_attention_heads = 2;
    c.num_key_value_heads = 1;
    c.num_hidden_layers = 2;
    c.vocab_size = 16;
    c.bos_token_id = 1;
    c.eos_token_id = 2;
    c
}
fn device(id: &str) -> Device {
    Device {
        id: id.into(),
        budget_bytes: 1 << 30,
        additional_parameters: 0,
        activation_bytes: None,
        temporary_bytes: None,
        checkpoint_staging_bytes: None,
        margin_bytes: 0,
    }
}
fn scenario() -> Scenario {
    Scenario {
        precision: Precision::F32,
        optimizer: Optimizer::AdamW,
        devices: vec![device("a")],
        layers: vec![LayerRange {
            start: 0,
            end: 2,
            device: "a".into(),
        }],
        embedding_device: "a".into(),
        output_device: "a".into(),
    }
}
#[test]
fn analytical_shapes_tying_and_memory_components() {
    let c = tiny();
    let shapes = feasibility::parameter_shapes(&c).unwrap();
    assert_eq!(shapes["model.layers.0.self_attn.k_proj.weight"], [2, 4]);
    assert_eq!(shapes["model.layers.0.self_attn.q_norm.weight"], [2]);
    assert!(!shapes.contains_key("lm_head.weight"));
    let report = feasibility::estimate(&c, scenario()).unwrap();
    // 16*4 embedding + 4 norm + 2*(2*4 norms + 2*2 head norms
    // + 2*4*4 Q/O + 2*4*2 K/V + 3*4*8 FFN) = 380.
    assert_eq!(report.base_parameters, 380);
    let d = &report.devices[0];
    assert_eq!(
        (
            d.weight_bytes,
            d.gradient_bytes,
            d.master_bytes,
            d.optimizer_bytes
        ),
        (1520, 1520, 0, 3040)
    );
    assert_eq!(d.persistent_bytes, 6080);
    assert_eq!(report.checkpoint_tensor_payload_bytes, 380 * 12);
    assert!(!report.estimate_complete);
    assert!(!report.training_ready);
    assert_eq!(d.status, "incomplete_estimate");
    let mut plan = scenario();
    plan.precision = Precision::Bf16;
    plan.devices[0].additional_parameters = 20;
    plan.devices[0].activation_bytes = Some(100);
    plan.devices[0].temporary_bytes = Some(200);
    plan.devices[0].checkpoint_staging_bytes = Some(300);
    plan.devices[0].margin_bytes = 400;
    let report = feasibility::estimate(&c, plan).unwrap();
    let d = &report.devices[0];
    assert_eq!(d.persistent_bytes, 400 * 18);
    assert_eq!(d.estimated_required_bytes, Some(8200));
    assert_eq!(d.status, "estimate_within_budget_unverified");
    assert!(report.estimate_complete);
    assert!(!report.training_ready);
    for precision in [Precision::F16, Precision::Bf16, Precision::F32] {
        let report = feasibility::estimate(
            &c,
            Scenario {
                optimizer: Optimizer::Sgd,
                precision,
                ..scenario()
            },
        )
        .unwrap();
        assert_eq!(report.devices[0].optimizer_bytes, 0);
        assert_eq!(
            report.devices[0].persistent_bytes,
            380 * if matches!(precision, Precision::F32) {
                8
            } else {
                10
            }
        );
    }
}
#[test]
fn per_device_budgets_are_never_pooled_and_untied_head_has_its_own_owner() {
    let mut c = tiny();
    c.tie_word_embeddings = false;
    let mut plan = scenario();
    plan.devices.push(device("b"));
    plan.output_device = "b".into();
    plan.layers = vec![
        LayerRange {
            start: 0,
            end: 1,
            device: "a".into(),
        },
        LayerRange {
            start: 1,
            end: 2,
            device: "b".into(),
        },
    ];
    plan.devices[0].budget_bytes = 220 * 16 - 1;
    let report = feasibility::estimate(&c, plan.clone()).unwrap();
    assert_eq!(report.base_parameters, 444);
    assert_eq!(report.devices[0].parameters, 220);
    assert_eq!(report.devices[1].parameters, 224);
    assert!(!report.budgets_sufficient_for_known_bytes);
    assert_eq!(report.devices[0].status, "insufficient_budget");
    assert_eq!(report.devices[1].status, "incomplete_estimate");
    c.tie_word_embeddings = true;
    assert!(
        feasibility::estimate(&c, plan)
            .unwrap_err()
            .contains("tied")
    );
}
#[test]
fn invalid_placement_and_arithmetic_overflow_fail_before_allocating_tensors() {
    let mut gap = scenario();
    gap.layers[0].end = 1;
    let mut overlap = scenario();
    overlap.layers.push(overlap.layers[0].clone());
    let mut unknown = scenario();
    unknown.layers[0].device = "missing".into();
    let mut duplicate = scenario();
    duplicate.devices.push(device("a"));
    let mut overflow = scenario();
    overflow.devices[0].additional_parameters = u64::MAX;
    let mut budget_overflow = scenario();
    budget_overflow.devices[0].margin_bytes = u64::MAX;
    for plan in [gap, overlap, unknown, duplicate, overflow, budget_overflow] {
        assert!(feasibility::estimate(&tiny(), plan).is_err());
    }
    let mut c = tiny();
    c.num_hidden_layers = 1025;
    assert!(feasibility::parameter_shapes(&c).is_err());
    c = tiny();
    c.intermediate_size = usize::MAX;
    assert!(feasibility::estimate(&c, scenario()).is_err());
    assert!(serde_json::from_str::<Scenario>(r#"{"precision":"int8"}"#).is_err());
}
#[test]
fn actual_0_6b_metadata_exceeds_eight_gib_before_activations() {
    let c = qwen::Config::from_json(include_bytes!("../../../examples/qwen3-0.6b.config.json"))
        .unwrap();
    let plan: Scenario =
        serde_json::from_slice(include_bytes!("../../../examples/preflight-0.6b.json")).unwrap();
    let report = feasibility::estimate(&c, plan).unwrap();
    assert_eq!(report.base_parameters, 596_049_920);
    assert_eq!(report.devices[0].persistent_bytes, 9_536_798_720);
    assert!(!report.budgets_sufficient_for_known_bytes);
    assert!(!report.estimate_complete);
    assert!(!report.training_ready);
}
