//! Opt-in model-level hardware qualification. No NVIDIA hardware is required by ordinary CI.
#![cfg(feature = "cuda")]
use std::io::Cursor;
use tesslot_dataset::{Limits, Reader, Sample, TokenizerIdentity};
use tesslot_tensor::qwen::{
    Model,
    optimizer::AdamWConfig,
    remix::Execution,
    runtime::{Plan, Precision, Target},
};
fn devices() -> usize {
    let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index", "--format=csv,noheader"])
        .output()
    else {
        return 0;
    };
    if !output.status.success() {
        return 0;
    }
    let count = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|s| !s.trim().is_empty())
        .count();
    match std::env::var("CUDA_VISIBLE_DEVICES") {
        Ok(v) if v.trim().is_empty() || v.trim() == "-1" => 0,
        Ok(v) => count.min(v.split(',').count()),
        Err(_) => count,
    }
}
fn identity() -> TokenizerIdentity {
    serde_json::from_slice(include_bytes!("../../../fixtures/tokenizer-identity.json")).unwrap()
}
fn sample() -> Sample {
    Reader::new(
        Cursor::new(include_bytes!("../../../fixtures/autograd.jsonl")),
        identity(),
        Limits::default(),
    )
    .unwrap()
    .next()
    .unwrap()
    .unwrap()
}
fn model() -> Model {
    let mut c = tesslot_model::qwen::Config::from_json(include_bytes!(
        "../../../fixtures/tiny-qwen/config.json"
    ))
    .unwrap();
    c.num_hidden_layers = 3;
    c.vocab_size = 101;
    c.bos_token_id = 1;
    c.eos_token_id = 2;
    c.max_position_embeddings = 256;
    let w = tesslot_artifacts::qwen::random_source(&c, 42, 256 * 1024).unwrap();
    let b = tesslot_artifacts::encode_f32(&w).unwrap();
    Model::remix_from_aligned(c, &b, identity(), 7, 42, true).unwrap()
}
fn flat(t: burn_tensor::Tensor<2>) -> Vec<f32> {
    t.into_data().try_to_vec().unwrap()
}
fn compare(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance * (1.0 + b.abs()),
            "{a} vs {b}, tolerance {tolerance}"
        );
    }
}
fn run(precision: Precision, dual: bool) {
    let required = if dual { 2 } else { 1 };
    if devices() < required {
        eprintln!("SKIP model CUDA gate: requires {required} visible GPU(s)");
        return;
    }
    let s = sample();
    let mut cpu = model();
    let mut gpu = model();
    let plan = Plan {
        root: Target::Cuda { index: 0 },
        layers: vec![
            Target::Cuda { index: 0 },
            Target::Cuda {
                index: usize::from(dual),
            },
            Target::Cuda {
                index: usize::from(dual),
            },
        ],
        precision,
        parameter_budget_bytes: 256 * 1024,
    };
    gpu.configure_runtime(plan.clone()).unwrap();
    let tol = match precision {
        Precision::F32 => 4e-4,
        Precision::Bf16 => 0.04,
        Precision::F16 => 0.01,
    };
    let full = cpu.remix_forward(&s, Execution::Dense).unwrap();
    let cached = gpu.remix_forward(&s, Execution::Chunked).unwrap();
    let dense = gpu.remix_forward(&s, Execution::Dense).unwrap();
    for ((cpu, gpu), dense) in full.iter().zip(&cached).zip(&dense) {
        assert_eq!(cpu.row_index, gpu.row_index);
        compare(&flat(gpu.logits.clone()), &flat(cpu.logits.clone()), tol);
        compare(&flat(dense.logits.clone()), &flat(gpu.logits.clone()), tol);
    }
    let a = cpu
        .remix_gradients(&cpu.remix_loss(&full, &s).unwrap())
        .unwrap();
    let b = gpu
        .remix_gradients(&gpu.remix_loss(&cached, &s).unwrap())
        .unwrap();
    for (name, gradient) in &a {
        compare(&b[name], gradient, tol);
    }
    assert!(
        b["model.layers.0.self_attn.v_proj.weight"]
            .iter()
            .any(|x| x.abs() > 1e-8),
        "first partition must receive later-loss gradients"
    );
    assert!(
        b["kidra.exchange.out.weight"]
            .iter()
            .any(|x| x.abs() > 1e-8),
        "shared exchange must receive gradients across layers"
    );
    cpu.enable_adamw(AdamWConfig::default(), 0.001).unwrap();
    gpu.enable_adamw(AdamWConfig::default(), 0.001).unwrap();
    cpu.train_remix_step(&s).unwrap();
    gpu.train_remix_step(&s).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("gpu");
    gpu.save(&path).unwrap();
    let mut resume = Model::restore(&path, true).unwrap();
    resume.configure_runtime(plan).unwrap();
    cpu.train_remix_step(&s).unwrap();
    gpu.train_remix_step(&s).unwrap();
    resume.train_remix_step(&s).unwrap();
    let logits = |m: &Model| {
        flat(
            m.remix_forward(&s, Execution::Dense)
                .unwrap()
                .last()
                .unwrap()
                .logits
                .clone(),
        )
    };
    compare(&logits(&gpu), &logits(&cpu), tol);
    compare(&logits(&resume), &logits(&gpu), 1e-5);
    eprintln!(
        "PASS model CUDA: precision={precision:?}, dual={dual}, full/cached/all-gradients/AdamW/checkpoint"
    );
}
#[test]
#[ignore = "NVIDIA model-level FP32 full-parameter gate"]
fn qwen_cuda_f32() {
    run(Precision::F32, false)
}
#[test]
#[ignore = "NVIDIA model-level BF16 gate"]
fn qwen_cuda_bf16() {
    run(Precision::Bf16, false)
}
#[test]
#[ignore = "NVIDIA model-level FP16 gate"]
fn qwen_cuda_f16() {
    run(Precision::F16, false)
}
#[test]
#[ignore = "requires two NVIDIA GPUs; asymmetric layer and shared-parameter gradients"]
fn qwen_dual_cuda() {
    run(Precision::F32, true)
}
