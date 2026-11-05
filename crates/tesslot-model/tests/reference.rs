//! Numerical equivalence and multi-layer causality acceptance tests.
use std::io::Cursor;
use tesslot_core::{ModelConfig, Phase};
use tesslot_dataset::{Limits, Reader, Sample};
use tesslot_model::{Cache, Prediction, ReferenceModel, WeightedLoss};

fn sample() -> Sample {
    Reader::new(
        Cursor::new(include_str!("../../../fixtures/canonical.jsonl")),
        serde_json::from_str(include_str!("../../../fixtures/tokenizer-identity.json")).unwrap(),
        Limits::default(),
    )
    .unwrap()
    .next()
    .unwrap()
    .unwrap()
}
fn equal(a: &[Prediction], b: &[Prediction]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.row_index, b.row_index);
        for (x, y) in a.logits.iter().zip(&b.logits) {
            assert!((x - y).abs() < 1e-12, "{x} != {y}, row {}", a.row_index);
        }
    }
}

#[test]
fn full_incremental_cached_uncached_and_loss_agree() {
    let s = sample();
    let model = ReferenceModel::new(ModelConfig::default()).unwrap();
    let full = model.forward(&s.rows, &s.metadata.windows).unwrap();
    let mut cache = Cache::default();
    let mut incremental = Vec::new();
    for group in s.ticks() {
        let current = model.tick(group, &s.metadata.windows, &mut cache).unwrap();
        if [0, 1, 3, 4, 5, 20, 72].contains(&group[0].tick) {
            let prefix = &s.rows[..group.last().unwrap().index + 1];
            let recomputed = model.forward(prefix, &s.metadata.windows).unwrap();
            let current_recomputed: Vec<_> = recomputed
                .into_iter()
                .filter(|p| s.rows[p.row_index].tick == group[0].tick)
                .collect();
            equal(&current, &current_recomputed);
        }
        incremental.extend(current);
        if group[0].tick >= 4 {
            assert_eq!(cache.window_tokens("b"), 0, "EOS must evict b");
        }
        if group[0].tick >= 46 {
            assert_eq!(cache.window_tokens("a"), 0, "EOS must evict a");
        }
    }
    equal(&full, &incremental);
    let mut a = WeightedLoss::default();
    let mut b = WeightedLoss::default();
    for p in &full {
        let r = &s.rows[p.row_index];
        a.add(&p.logits, r.token_id as usize, r.loss_weight)
            .unwrap();
    }
    for p in &incremental {
        let r = &s.rows[p.row_index];
        b.add(&p.logits, r.token_id as usize, r.loss_weight)
            .unwrap();
    }
    assert!((a.mean() - b.mean()).abs() < 1e-12);
    assert!(a.effective_weight > 0.0 && a.mean().is_finite());
    equal(
        &full,
        &ReferenceModel::new(ModelConfig::default())
            .unwrap()
            .forward(&s.rows, &s.metadata.windows)
            .unwrap(),
    );
}

#[test]
fn multilayer_future_and_same_tick_target_changes_cannot_leak() {
    let s = sample();
    let model = ReferenceModel::new(ModelConfig {
        layers: 3,
        ..ModelConfig::default()
    })
    .unwrap();
    let full = model.forward(&s.rows, &s.metadata.windows).unwrap();
    let mut changed = s.rows.clone();
    for r in &mut changed {
        if r.tick > 1 || (r.tick == 1 && r.phase == Phase::D) {
            r.token_id = (r.token_id + 17) % 128;
        }
    }
    let outputs = model.forward(&changed, &s.metadata.windows).unwrap();
    let old: Vec<_> = full
        .into_iter()
        .filter(|p| s.rows[p.row_index].tick <= 1)
        .collect();
    let new: Vec<_> = outputs
        .into_iter()
        .filter(|p| s.rows[p.row_index].tick <= 1)
        .collect();
    equal(&old, &new);
    // A supervised B target must also be excluded from its own query input.
    let target = s
        .rows
        .iter()
        .find(|r| r.phase == Phase::B && r.loss_weight > 0.0)
        .unwrap();
    let mut changed = s.rows.clone();
    changed[target.index].token_id = 95;
    let a = model.forward(&s.rows, &s.metadata.windows).unwrap();
    let b = model.forward(&changed, &s.metadata.windows).unwrap();
    equal(
        &[a.iter()
            .find(|p| p.row_index == target.index)
            .unwrap()
            .clone()],
        &[b.iter()
            .find(|p| p.row_index == target.index)
            .unwrap()
            .clone()],
    );
}

#[test]
fn cross_window_exchange_and_local_only_ablation() {
    let s = sample();
    let mut changed = s.rows.clone();
    let target = changed
        .iter_mut()
        .find(|r| r.window == "a" && r.tick == 1 && r.phase == Phase::D)
        .unwrap();
    target.token_id = 96;
    for global in [true, false] {
        let model = ReferenceModel::new(ModelConfig {
            global_attention: global,
            ..ModelConfig::default()
        })
        .unwrap();
        let a = model.forward(&s.rows, &s.metadata.windows).unwrap();
        let b = model.forward(&changed, &s.metadata.windows).unwrap();
        let select = |p: &Prediction| {
            let r = &s.rows[p.row_index];
            r.window == "b" && r.tick == 2
        };
        let a: Vec<_> = a.into_iter().filter(select).collect();
        let b: Vec<_> = b.into_iter().filter(select).collect();
        assert_eq!(a.len(), 1);
        let difference: f64 = a[0]
            .logits
            .iter()
            .zip(&b[0].logits)
            .map(|(x, y)| (x - y).abs())
            .sum();
        if global {
            assert!(difference > 1e-6);
        } else {
            assert_eq!(difference, 0.0);
        }
    }
}

#[test]
fn noop_does_not_grow_cache_and_window_embeddings_are_configurable() {
    let s = sample();
    let model = ReferenceModel::new(ModelConfig::default()).unwrap();
    let mut cache = Cache::default();
    let opening: Vec<_> = s.rows.iter().filter(|r| r.tick == 0).cloned().collect();
    model
        .tick(&opening, &s.metadata.windows, &mut cache)
        .unwrap();
    let size = cache.durable_tokens();
    let mut noop = s
        .rows
        .iter()
        .find(|r| r.op == "noop" && r.window == "b")
        .unwrap()
        .clone();
    noop.tick = 1;
    model
        .tick(&[noop.clone()], &s.metadata.windows, &mut cache)
        .unwrap();
    assert_eq!(cache.durable_tokens(), size);
    let different_model = ReferenceModel::new(ModelConfig {
        seed: 43,
        ..ModelConfig::default()
    })
    .unwrap();
    let mut next = noop.clone();
    next.tick = 2;
    assert!(
        different_model
            .tick(&[next], &s.metadata.windows, &mut cache)
            .unwrap_err()
            .to_string()
            .contains("different model")
    );
    assert!(
        model
            .tick(&[noop], &s.metadata.windows, &mut cache)
            .is_err()
    );
    let a = model.forward(&s.rows, &s.metadata.windows).unwrap();
    let no_identity = ReferenceModel::new(ModelConfig {
        window_embeddings: false,
        ..ModelConfig::default()
    })
    .unwrap();
    let b = no_identity.forward(&s.rows, &s.metadata.windows).unwrap();
    assert_ne!(a[0].logits, b[0].logits);
    let config = ModelConfig {
        max_windows: 3,
        ..ModelConfig::default()
    };
    assert!(
        ReferenceModel::new(config)
            .unwrap()
            .forward(&s.rows, &s.metadata.windows)
            .is_err()
    );
}
