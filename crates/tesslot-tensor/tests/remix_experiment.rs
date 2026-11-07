//! Explicitly opt-in Qwen/Remix numerical execution and training gates.
use std::io::Cursor;
use tesslot_dataset::{Limits, Reader, Sample, TokenizerIdentity};
use tesslot_tensor::qwen::{
    Model,
    remix::{Cache, Execution},
};
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
fn model(training: bool) -> Model {
    Model::remix_demo(identity(), 7, 42, training).unwrap()
}
fn model_layers(layers: usize, training: bool) -> Model {
    if layers == 1 {
        return model(training);
    }
    let mut config = tesslot_model::qwen::Config::from_json(include_bytes!(
        "../../../fixtures/tiny-qwen/config.json"
    ))
    .unwrap();
    config.num_hidden_layers = layers;
    config.vocab_size = 101;
    config.bos_token_id = 1;
    config.eos_token_id = 2;
    config.max_position_embeddings = 256;
    let weights = tesslot_artifacts::qwen::random_source(&config, 42, 128 * 1024).unwrap();
    let bytes = tesslot_artifacts::encode_f32(&weights).unwrap();
    Model::remix_from_aligned(config, &bytes, identity(), 7, 42, training).unwrap()
}
fn values(t: burn_tensor::Tensor<2>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
}
#[test]
#[ignore = "opt-in dense/projected-KV chunk logits and all-parameter gradient parity"]
fn dense_and_attached_chunked_execution_match() {
    for layers in [1, 2] {
        let s = sample();
        assert!(
            model(false)
                .remix_tick(
                    "missing-prefix",
                    s.ticks().nth(1).unwrap(),
                    &s.metadata.windows,
                    &mut Cache::default()
                )
                .is_err()
        );
        let a = model_layers(layers, true);
        let b = model_layers(layers, true);
        let dense = a.remix_forward(&s, Execution::Dense).unwrap();
        let cached = b.remix_forward(&s, Execution::Chunked).unwrap();
        assert_eq!(dense.len(), cached.len());
        for (x, y) in dense.iter().zip(&cached) {
            assert_eq!(x.row_index, y.row_index);
            for (x, y) in values(x.logits.clone())
                .iter()
                .zip(values(y.logits.clone()))
            {
                assert!((x - y).abs() < 2e-5, "logit {x} vs {y}");
            }
        }
        let loss_a = a.remix_loss(&dense, &s).unwrap();
        let loss_b = b.remix_loss(&cached, &s).unwrap();
        let ga = a.remix_gradients(&loss_a).unwrap();
        let gb = b.remix_gradients(&loss_b).unwrap();
        for (name, g) in &ga {
            for (x, y) in g.iter().zip(&gb[name]) {
                assert!(
                    (x - y).abs() < 3e-4 * (1.0 + x.abs()),
                    "gradient {name}: {x} vs {y}"
                );
            }
        }
        let mut cache = Cache::default();
        for tick in s.ticks() {
            a.remix_tick(
                &s.metadata.trajectory_uuid,
                tick,
                &s.metadata.windows,
                &mut cache,
            )
            .unwrap();
            assert!(cache.windows().iter().all(|id| {
                s.metadata
                    .windows
                    .iter()
                    .any(|w| w.id == *id && !w.close.is_some_and(|close| close <= tick[0].tick))
            }));
            if tick[0].tick == 5 {
                assert!(cache.is_empty());
            }
        }
    }
}

#[test]
#[ignore = "opt-in complete-trajectory optimization and checkpoint continuation"]
fn train_complete_trajectory_and_restore_optimizer() {
    let s = sample();
    let mut m = model(true);
    m.enable_adamw(Default::default(), 0.003).unwrap();
    m.bind_remix_dataset("a".repeat(64)).unwrap();
    let loss = |m: &Model| {
        let p = m.remix_forward(&s, Execution::Chunked).unwrap();
        m.remix_loss(&p, &s)
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0]
    };
    let first = loss(&m);
    for _ in 0..80 {
        let metric = m.train_remix_step(&s).unwrap();
        assert!(metric.gradient_norm.is_finite());
    }
    let last = loss(&m);
    eprintln!("Qwen/Remix actual loss {first} -> {last}");
    assert!(last < first * 0.5);
    assert_eq!(m.remix_state().unwrap().completed_samples, 80);
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("saved");
    m.save(&path).unwrap();
    let mut restored = Model::restore(&path, true).unwrap();
    assert_eq!(loss(&m), loss(&restored));
    assert!(restored.bind_remix_dataset("b".repeat(64)).is_err());
    let a = m.train_remix_step(&s).unwrap();
    let b = restored.train_remix_step(&s).unwrap();
    assert_eq!(a.loss, b.loss);
    let x = root.path().join("a");
    let y = root.path().join("b");
    m.save(&x).unwrap();
    restored.save(&y).unwrap();
    for file in [
        "model.safetensors",
        "optimizer.safetensors",
        "manifest.json",
    ] {
        assert_eq!(
            std::fs::read(x.join(file)).unwrap(),
            std::fs::read(y.join(file)).unwrap()
        );
    }
    let mut cache = Cache::default();
    restored
        .remix_tick(
            "first",
            s.ticks().next().unwrap(),
            &s.metadata.windows,
            &mut cache,
        )
        .unwrap();
    assert!(
        restored
            .remix_tick(
                "other",
                s.ticks().nth(1).unwrap(),
                &s.metadata.windows,
                &mut cache
            )
            .is_err()
    );
    restored.train_remix_step(&s).unwrap();
    assert!(
        restored
            .remix_tick(
                "first",
                s.ticks().nth(1).unwrap(),
                &s.metadata.windows,
                &mut cache
            )
            .is_err()
    );
    let mut bad = s.clone();
    bad.metadata.tokenizer_sha256 = "0".repeat(64);
    assert!(restored.train_remix_step(&bad).is_err());
}

fn changed_sample(change: impl Fn(&mut tesslot_core::Row)) -> Sample {
    let chunks = include_str!("../../../fixtures/autograd.jsonl")
        .lines()
        .map(|line| {
            let mut chunk: tesslot_dataset::Chunk = serde_json::from_str(line).unwrap();
            for row in &mut chunk.rows {
                change(row);
            }
            chunk.token_ids = chunk.rows.iter().map(|r| r.token_id).collect();
            chunk.labels = chunk
                .rows
                .iter()
                .map(|r| {
                    if r.loss_mask == 1 {
                        r.token_id as i64
                    } else {
                        -100
                    }
                })
                .collect();
            chunk.loss_mask = chunk.rows.iter().map(|r| r.loss_mask).collect();
            chunk.loss_weights = chunk.rows.iter().map(|r| r.loss_weight).collect();
            serde_json::to_string(&chunk).unwrap() + "\n"
        })
        .collect::<String>();
    Reader::new(Cursor::new(chunks), identity(), Limits::default())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
}
#[test]
#[ignore = "opt-in Qwen weighted objective, label leakage and NOOP durable-cache gate"]
fn weights_causal_targets_and_noop_state_are_correct() {
    use tesslot_core::Phase;
    let m = model_layers(2, false);
    let s = sample();
    let p = m.remix_forward(&s, Execution::Dense).unwrap();
    let changed = changed_sample(|r| {
        if r.phase == Phase::D && r.op == "generate" && r.durable && r.token_id != 2 {
            r.token_id = 90;
        }
    });
    let q = m.remix_forward(&changed, Execution::Dense).unwrap();
    for (a, b) in p
        .iter()
        .zip(&q)
        .filter(|(a, _)| s.rows[a.row_index].tick <= 1)
    {
        for (x, y) in values(a.logits.clone())
            .iter()
            .zip(values(b.logits.clone()))
        {
            assert!((x - y).abs() < 1e-6, "same-tick/future label leaked");
        }
    }
    let changed = changed_sample(|r| {
        if r.phase == Phase::B && r.tick == 2 {
            r.token_id = 88;
        }
    });
    let q = m.remix_forward(&changed, Execution::Dense).unwrap();
    let supervised = s
        .rows
        .iter()
        .find(|r| r.phase == Phase::B && r.loss_weight > 0.0)
        .unwrap();
    let a = p.iter().find(|p| p.row_index == supervised.index).unwrap();
    let b = q.iter().find(|p| p.row_index == supervised.index).unwrap();
    assert_eq!(
        values(a.logits.clone()),
        values(b.logits.clone()),
        "B target leaked before its insertion"
    );
    let tiny = changed_sample(|r| {
        if r.loss_mask == 1 {
            r.loss_weight = 0.01;
        }
    });
    let predictions = m.remix_forward(&tiny, Execution::Dense).unwrap();
    let zeroed = predictions
        .iter()
        .map(|p| tesslot_tensor::qwen::remix::Prediction {
            row_index: p.row_index,
            logits: p.logits.clone() * 0.0,
        })
        .collect::<Vec<_>>();
    let weight = tiny.rows.iter().map(|r| r.loss_weight).sum::<f64>();
    assert!(weight < 1.0);
    let actual = m
        .remix_loss(&zeroed, &tiny)
        .unwrap()
        .into_data()
        .try_to_vec::<f32>()
        .unwrap()[0];
    assert!(
        (actual - ((101f64).ln() * weight) as f32).abs() < 1e-6,
        "effective weight clamp is required"
    );
    assert!(m.remix_loss(&zeroed[1..], &tiny).is_err());
    let mut cache = Cache::default();
    let mut sizes = Vec::new();
    for tick in s.ticks() {
        m.remix_tick(
            &s.metadata.trajectory_uuid,
            tick,
            &s.metadata.windows,
            &mut cache,
        )
        .unwrap();
        sizes.push(cache.len());
    }
    assert_eq!(
        sizes[3], sizes[4],
        "non-durable NOOP must not grow projected KV"
    );
    assert_eq!(sizes[5], 0);
    let wrong = model(false);
    let mut cache = Cache::default();
    m.remix_tick(
        "x",
        s.ticks().next().unwrap(),
        &s.metadata.windows,
        &mut cache,
    )
    .unwrap();
    assert!(
        wrong
            .remix_tick(
                "x",
                s.ticks().nth(1).unwrap(),
                &s.metadata.windows,
                &mut cache
            )
            .is_err()
    );
}
#[test]
#[ignore = "opt-in actual exporter headers, late opens, replacement and tool-result parity"]
fn canonical_tools_headers_and_dynamic_windows_match() {
    let s = Reader::new(
        Cursor::new(include_bytes!("../../../fixtures/canonical.jsonl")),
        identity(),
        Limits::default(),
    )
    .unwrap()
    .next()
    .unwrap()
    .unwrap();
    let m = model(false);
    let a = m.remix_forward(&s, Execution::Dense).unwrap();
    let b = m.remix_forward(&s, Execution::Chunked).unwrap();
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(&b) {
        assert_eq!(a.row_index, b.row_index);
        for (x, y) in values(a.logits.clone())
            .iter()
            .zip(values(b.logits.clone()))
        {
            assert!((x - y).abs() < 3e-5, "canonical logit {x} vs {y}");
        }
    }
    assert!(s.rows.iter().any(|r| r.op == "header"));
    assert!(s.metadata.windows.len() > 2);
    let mut single = s.clone();
    single.chunk_ends = vec![s.rows.len()];
    single.metadata.chunk_count = 1;
    single.metadata.rows = s.rows.clone();
    single.metadata.tick_end = s.rows.last().unwrap().tick;
    single.metadata.token_ids = s.rows.iter().map(|r| r.token_id).collect();
    single.metadata.labels = s
        .rows
        .iter()
        .map(|r| {
            if r.loss_mask == 1 {
                r.token_id as i64
            } else {
                -100
            }
        })
        .collect();
    single.metadata.loss_mask = s.rows.iter().map(|r| r.loss_mask).collect();
    single.metadata.loss_weights = s.rows.iter().map(|r| r.loss_weight).collect();
    let c = m.remix_forward(&single, Execution::Chunked).unwrap();
    for (a, c) in b.iter().zip(c) {
        assert_eq!(values(a.logits.clone()), values(c.logits));
    }
}

#[test]
#[ignore = "opt-in complete gradient and AdamW parity for exact target-prefix recomputation"]
fn target_recomputation_preserves_all_gradients_and_optimizer_resume() {
    use tesslot_tensor::qwen::optimizer::AdamWConfig;
    let s = sample();
    let a = model_layers(2, true);
    let b = model_layers(2, true);
    let out = a.remix_forward(&s, Execution::Dense).unwrap();
    let loss = a.remix_loss(&out, &s).unwrap();
    let value = loss.clone().into_data().try_to_vec::<f32>().unwrap()[0];
    let full = a.remix_gradients(&loss).unwrap();
    let (recomputed, gradients) = b.recomputed_remix_gradients(&s).unwrap();
    assert!((value - recomputed).abs() < 2e-5);
    for (name, g) in full {
        for (x, y) in g.iter().zip(&gradients[&name]) {
            assert!((x - y).abs() < 3e-4 * (1.0 + x.abs()), "{name}: {x} vs {y}");
        }
    }
    let mut a = model_layers(2, true);
    let mut b = model_layers(2, true);
    a.enable_adamw(AdamWConfig::default(), 0.001).unwrap();
    b.enable_adamw(AdamWConfig::default(), 0.001).unwrap();
    a.train_remix_step(&s).unwrap();
    b.train_remix_recomputed(&s).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("recomputed");
    b.save(&path).unwrap();
    let mut resumed = Model::restore(&path, true).unwrap();
    b.train_remix_recomputed(&s).unwrap();
    resumed.train_remix_recomputed(&s).unwrap();
    let logits = |m: &Model| {
        values(
            m.remix_forward(&s, Execution::Dense)
                .unwrap()
                .last()
                .unwrap()
                .logits
                .clone(),
        )
    };
    assert_eq!(logits(&b), logits(&resumed));
    // First update follows the same objective with a different summation order.
    let mut again = model_layers(2, true);
    again.enable_adamw(AdamWConfig::default(), 0.001).unwrap();
    again.train_remix_recomputed(&s).unwrap();
    for (x, y) in logits(&a).iter().zip(logits(&again)) {
        assert!((x - y).abs() < 3e-4, "updated logits {x} vs {y}");
    }
}
