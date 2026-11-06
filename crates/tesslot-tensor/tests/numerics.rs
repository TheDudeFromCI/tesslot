//! Exhaustive parameter derivatives against an independent scalar implementation.
use tesslot_core::ModelConfig;
use tesslot_dataset::{Limits, Reader, Sample};
use tesslot_model::{ReferenceModel, WeightedLoss};
use tesslot_tensor::{Cache, Model, Parameters};
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
fn config() -> ModelConfig {
    ModelConfig {
        hidden_size: 2,
        layers: 2,
        max_windows: 7,
        ..ModelConfig::default()
    }
}
fn scalar(values: &Parameters, s: &Sample) -> f64 {
    let model = ReferenceModel::from_parameters(config(), values).unwrap();
    let mut loss = WeightedLoss::default();
    for p in model.forward(&s.rows, &s.metadata.windows).unwrap() {
        let r = &s.rows[p.row_index];
        loss.add(&p.logits, r.token_id as usize, r.loss_weight)
            .unwrap();
    }
    loss.mean()
}
#[test]
fn every_parameter_matches_finite_differences_and_checkpointed_graph() {
    let s = sample();
    let oracle = ReferenceModel::new(config()).unwrap();
    let values = oracle.parameters();
    let expected = oracle.forward(&s.rows, &s.metadata.windows).unwrap();
    let mut baseline = None;
    for checkpoint in [false, true] {
        let model = Model::new(config(), &values, checkpoint).unwrap();
        let output = model.forward(&s.rows, &s.metadata.windows).unwrap();
        for (a, b) in output.iter().zip(&expected) {
            assert_eq!(a.row_index, b.row_index);
            let actual = a.logits.clone().into_data().try_to_vec::<f64>().unwrap();
            for (x, y) in actual.iter().zip(&b.logits) {
                assert!((x - y).abs() < 1e-10, "logit: {x} vs {y}");
            }
        }
        let loss = model.loss(&output, &s.rows).unwrap();
        let number = loss.clone().into_data().try_to_vec::<f64>().unwrap()[0];
        assert!((number - scalar(&values, &s)).abs() < 1e-10);
        let gradients = model.gradients(&loss).unwrap();
        if let Some(old) = &baseline {
            assert_gradients(&gradients, old, 1e-10);
        } else {
            baseline = Some(gradients.clone());
        }
        for (name, matrix) in &values {
            for (i, row) in matrix.iter().enumerate() {
                for (j, _) in row.iter().enumerate() {
                    let mut plus = values.clone();
                    let mut minus = values.clone();
                    plus.get_mut(name).unwrap()[i][j] += 1e-6;
                    minus.get_mut(name).unwrap()[i][j] -= 1e-6;
                    let numeric = (scalar(&plus, &s) - scalar(&minus, &s)) / 2e-6;
                    let analytic = gradients[name][i][j];
                    assert!(
                        (numeric - analytic).abs() < 2e-5 * (1.0 + numeric.abs()),
                        "{name}[{i},{j}]: finite {numeric} vs autodiff {analytic}"
                    );
                }
            }
        }
        // One real full-parameter SGD update must reduce this tiny objective.
        let mut updated = values.clone();
        for (name, m) in &mut updated {
            for (row, g) in m.iter_mut().zip(&gradients[name]) {
                for (v, g) in row.iter_mut().zip(g) {
                    *v -= 0.001 * g;
                }
            }
        }
        assert!(scalar(&updated, &s) < number);
    }
}
fn assert_gradients(a: &Parameters, b: &Parameters, tolerance: f64) {
    for (name, m) in a {
        for (x, y) in m.iter().flatten().zip(b[name].iter().flatten()) {
            assert!((x - y).abs() < tolerance, "gradient mismatch: {name}");
        }
    }
}
#[test]
fn attached_chunks_and_fresh_sample_caches_match() {
    let s = sample();
    let values = ReferenceModel::new(config()).unwrap().parameters();
    let model = Model::new(config(), &values, false).unwrap();
    let full = model.forward(&s.rows, &s.metadata.windows).unwrap();
    let loss = model.loss(&full, &s.rows).unwrap();
    let expected = model.gradients(&loss).unwrap();
    // Fresh forward after consumed backward, retaining one cache across storage chunks.
    let model = Model::new(config(), &values, false).unwrap();
    let mut cache = Cache::default();
    let mut output = Vec::new();
    let mut begin = 0;
    for &end in &s.chunk_ends {
        let mut start = begin;
        while start < end {
            let next = start + s.rows[start..end].partition_point(|r| r.tick == s.rows[start].tick);
            output.extend(
                model
                    .tick(&s.rows[start..next], &s.metadata.windows, &mut cache)
                    .unwrap(),
            );
            start = next;
        }
        begin = end;
    }
    let loss = model.loss(&output, &s.rows).unwrap();
    let actual = model.gradients(&loss).unwrap();
    assert_gradients(&actual, &expected, 1e-10);
    // The initial external token has zero direct loss but influences later targets.
    let token = s
        .rows
        .iter()
        .find(|r| r.phase == tesslot_core::Phase::B && r.tick == 0)
        .unwrap()
        .token_id as usize;
    assert!(actual["tokens"][token].iter().any(|g| g.abs() > 1e-9));
}

#[test]
fn future_and_same_tick_targets_cannot_contaminate_tensor_predictions() {
    let s = sample();
    let values = ReferenceModel::new(config()).unwrap().parameters();
    let model = Model::new(config(), &values, false).unwrap();
    let a = model.forward(&s.rows, &s.metadata.windows).unwrap();
    let mut rows = s.rows.clone();
    for r in &mut rows {
        if r.tick > 1 || (r.tick == 1 && r.phase == tesslot_core::Phase::D) {
            r.token_id = (r.token_id + 11) % 128;
        }
    }
    let b = model.forward(&rows, &s.metadata.windows).unwrap();
    for (a, b) in a
        .iter()
        .zip(b)
        .filter(|(a, _)| s.rows[a.row_index].tick <= 1)
    {
        let x = a.logits.clone().into_data().try_to_vec::<f64>().unwrap();
        let y = b.logits.into_data().try_to_vec::<f64>().unwrap();
        for (x, y) in x.iter().zip(y) {
            assert!((x - y).abs() < 1e-10);
        }
    }
}

#[test]
fn cache_cannot_cross_model_instances() {
    let s = sample();
    let values = ReferenceModel::new(config()).unwrap().parameters();
    let a = Model::new(config(), &values, false).unwrap();
    let b = Model::new(config(), &values, false).unwrap();
    let mut cache = Cache::default();
    let first = s.ticks().next().unwrap();
    a.tick(first, &s.metadata.windows, &mut cache).unwrap();
    let error = b
        .tick(first, &s.metadata.windows, &mut cache)
        .err()
        .unwrap();
    assert!(error.contains("different model"));
}
