//! Exact target-prefix replay: trade repeated forward work for fewer live prediction activations.
//! This is graph-preserving recomputation, not truncated BPTT or whole-block checkpointing.
use super::*;
use tesslot_core::{Phase, visible_to};
use tesslot_dataset::Sample;
impl Model {
    /// Run one full-parameter, norm-clipped SGD update on a single-target trajectory,
    /// then evaluate the same target without autograd. This is an ephemeral diagnostic,
    /// not a persistent training job or a selection of the eventual training optimizer.
    pub fn remix_single_target_sgd_probe(
        &mut self,
        sample: &Sample,
        learning_rate: f32,
    ) -> Result<remix::SgdProbe, String> {
        if !self.training
            || self.adam.is_some()
            || self.step != 0
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
        {
            return Err(
                "SGD probe requires a fresh autodiff model, no AdamW and a positive finite rate"
                    .into(),
            );
        }
        eprintln!("{}", serde_json::json!({"phase":"sgd_forward_start"}));
        let loss = self.remix_single_target_loss(sample)?;
        let before = loss
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| e.to_string())?[0];
        if !before.is_finite() {
            return Err("nonfinite SGD objective".into());
        }
        eprintln!(
            "{}",
            serde_json::json!({"phase":"sgd_forward_complete","loss":before})
        );
        let saved_rate = self.learning_rate;
        self.learning_rate = learning_rate;
        eprintln!("{}", serde_json::json!({"phase":"sgd_backward_and_update"}));
        let metrics = match self.update(loss) {
            Ok(metrics) => metrics,
            Err(error) => {
                self.learning_rate = saved_rate;
                return Err(error);
            }
        };
        self.remix
            .as_mut()
            .ok_or("missing Remix state")?
            .completed_samples = metrics.step;
        eprintln!(
            "{}",
            serde_json::json!({"phase":"sgd_update_committed","step":metrics.step})
        );
        // Sharing storage without autodiff avoids retaining a second backward graph.
        let evaluation = self.inference_snapshot();
        evaluation.validate_remix_sample(sample)?;
        let target = sample
            .rows
            .iter()
            .find(|r| r.loss_weight > 0.0)
            .ok_or("missing target")?;
        let after = evaluation
            .target_prefix_loss(sample, target, target.loss_weight.max(1.0))?
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| e.to_string())?[0];
        if !after.is_finite() {
            return Err("nonfinite post-update SGD objective".into());
        }
        eprintln!(
            "{}",
            serde_json::json!({"phase":"sgd_evaluation_complete","loss":after})
        );
        Ok(remix::SgdProbe {
            loss_before: before,
            loss_after: after,
            loss_decreased: after < before,
            gradient_norm: metrics.gradient_norm,
            learning_rate,
            updates: metrics.step,
            sample_cursor: self
                .remix
                .as_ref()
                .ok_or("missing Remix state")?
                .completed_samples,
            all_gradients_finite: true,
            all_updates_finite: true,
        })
    }
    /// Exact normalized loss for a trajectory with one effective target. All durable
    /// causal ancestors remain attached; future and zero-weight prediction branches
    /// cannot affect this objective and are never constructed.
    pub fn remix_single_target_loss(&self, sample: &Sample) -> Result<Tensor<1>, String> {
        self.validate_remix_sample(sample)?;
        if !self.training {
            return Err("target-prefix backward requires autodiff".into());
        }
        let targets = sample
            .rows
            .iter()
            .filter(|r| r.loss_weight > 0.0)
            .collect::<Vec<_>>();
        if targets.len() != 1 {
            return Err("target-prefix probe requires exactly one effective target".into());
        }
        let denominator = targets[0].loss_weight.max(1.0);
        self.target_prefix_loss(sample, targets[0], denominator)
    }
    fn target_prefix_loss(
        &self,
        sample: &Sample,
        target: &tesslot_core::Row,
        denominator: f64,
    ) -> Result<Tensor<1>, String> {
        let memory = sample
            .rows
            .iter()
            .filter(|r| {
                r.durable
                    && (r.tick < target.tick
                        || (r.tick == target.tick
                            && (r.phase == Phase::A
                                || (r.phase == Phase::B
                                    && (target.phase == Phase::D || r.index < target.index)))))
            })
            .collect::<Vec<_>>();
        let rows = memory
            .iter()
            .map(|r| (*r, false))
            .chain(std::iter::once((target, true)))
            .collect::<Vec<_>>();
        let states = rows
            .iter()
            .map(|(r, q)| self.state_row(r, &sample.metadata.windows, *q))
            .collect::<Result<Vec<_>, _>>()?;
        let allowed = rows
            .iter()
            .flat_map(|(q, _)| {
                memory
                    .iter()
                    .map(|k| visible_to(q, k, &sample.metadata.windows))
            })
            .collect::<Vec<_>>();
        let frame = self.window_frame_logits(
            &states,
            memory.len(),
            &[],
            &allowed,
            memory.len()..memory.len() + 1,
        )?;
        let p = frame.logits;
        let loss = -log_softmax(p, 1)
            .slice([0..1, target.token_id as usize..target.token_id as usize + 1])
            .sum()
            * (target.loss_weight / denominator) as f32;
        Ok(loss)
    }

    /// Accumulate each normalized target's complete gradient before a single optimizer update.
    /// No optimizer step or detachment occurs at a storage-chunk boundary.
    fn recomputed(&self, sample: &Sample) -> Result<(f32, BTreeMap<String, T>), String> {
        self.validate_remix_sample(sample)?;
        if !self.training {
            return Err("recomputation requires autodiff".into());
        }
        let denominator = sample
            .rows
            .iter()
            .map(|r| r.loss_weight)
            .sum::<f64>()
            .max(1.0);
        let mut gradients = self
            .parameters
            .iter()
            .map(|(n, p)| (n.clone(), p.clone().zeros_like().detach()))
            .collect::<BTreeMap<_, _>>();
        let mut total = 0.0;
        let mut targets = 0;
        for target in sample.rows.iter().filter(|r| r.loss_weight > 0.0) {
            targets += 1;
            let loss = self.target_prefix_loss(sample, target, denominator)?;
            total += loss
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|e| e.to_string())?[0];
            let gs = loss.backward();
            for (name, p) in &self.parameters {
                if let Some(g) = p.grad(&gs) {
                    gradients.insert(name.clone(), gradients[name].clone() + g.detach());
                }
            }
        }
        if targets == 0 {
            return Err("sample has no effective training targets".into());
        }
        Ok((total, gradients))
    }
    /// Diagnostic full gradient snapshot for comparing exact recomputation to ordinary replay.
    pub fn recomputed_remix_gradients(
        &self,
        sample: &Sample,
    ) -> Result<(f32, BTreeMap<String, Vec<f32>>), String> {
        let (loss, gs) = self.recomputed(sample)?;
        let gs = gs
            .into_iter()
            .map(|(n, g)| {
                Ok((
                    n,
                    g.into_data()
                        .try_to_vec::<f32>()
                        .map_err(|e| e.to_string())?,
                ))
            })
            .collect::<Result<_, String>>()?;
        Ok((loss, gs))
    }
    /// Exact full-trajectory update with target-prefix replay; sampler advances only after commit.
    pub fn train_remix_recomputed(&mut self, sample: &Sample) -> Result<Metrics, String> {
        let (loss, gs) = self.recomputed(sample)?;
        let metrics = self.apply_gradients(loss, gs)?;
        self.remix
            .as_mut()
            .ok_or("missing Remix state")?
            .completed_samples += 1;
        Ok(metrics)
    }
}

#[cfg(test)]
mod single_target_probe_tests {
    use super::*;
    fn setup() -> (Model, Sample) {
        let identity: tesslot_dataset::TokenizerIdentity = serde_json::from_slice(include_bytes!(
            "../../../../fixtures/pretrained-backward-tiny/tokenizer-identity.json"
        ))
        .unwrap();
        let sample = tesslot_dataset::Reader::new(
            Cursor::new(include_bytes!(
                "../../../../fixtures/pretrained-backward-tiny/trajectory.jsonl"
            )),
            identity.clone(),
            tesslot_dataset::Limits::default(),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
        (Model::remix_demo(identity, 24, 42, true).unwrap(), sample)
    }
    #[test]
    fn single_target_prefix_matches_attached_replay_loss_and_every_parameter_gradient() {
        let (ordinary, sample) = setup();
        let output = ordinary
            .remix_forward(&sample, super::super::remix::Execution::Chunked)
            .unwrap();
        let loss = ordinary.remix_loss(&output, &sample).unwrap();
        let number = loss.clone().into_data().try_to_vec::<f32>().unwrap()[0];
        let expected = ordinary.remix_gradients(&loss).unwrap();
        let (prefix, sample) = setup();
        let loss = prefix.remix_single_target_loss(&sample).unwrap();
        let number_prefix = loss.clone().into_data().try_to_vec::<f32>().unwrap()[0];
        assert!((number - number_prefix).abs() < 2e-5);
        let actual = prefix.remix_gradients(&loss).unwrap();
        for (name, values) in &expected {
            for (a, b) in values.iter().zip(&actual[name]) {
                assert!((a - b).abs() < 3e-4 * (1.0 + a.abs()), "{name}: {a} vs {b}");
            }
        }
        assert_eq!(prefix.step, 0);
        assert_eq!(prefix.remix_state().unwrap().completed_samples, 0);
        assert!(prefix.adam.is_none());
    }
}

#[cfg(test)]
mod sgd_probe_tests {
    use super::*;
    fn setup() -> (Model, Sample) {
        let identity: tesslot_dataset::TokenizerIdentity = serde_json::from_slice(include_bytes!(
            "../../../../fixtures/pretrained-backward-tiny/tokenizer-identity.json"
        ))
        .unwrap();
        let sample = tesslot_dataset::Reader::new(
            Cursor::new(include_bytes!(
                "../../../../fixtures/pretrained-backward-tiny/trajectory.jsonl"
            )),
            identity.clone(),
            tesslot_dataset::Limits::default(),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
        (Model::remix_demo(identity, 24, 42, true).unwrap(), sample)
    }
    #[test]
    fn one_step_matches_manual_clipped_sgd_and_reduces_tiny_objective() {
        let (reference, sample) = setup();
        let grads = reference
            .remix_gradients(&reference.remix_single_target_loss(&sample).unwrap())
            .unwrap();
        let norm = grads
            .values()
            .flatten()
            .fold(0.0_f64, |n, g| n.hypot(f64::from(*g))) as f32;
        let rate = 0.0001;
        let expected = reference
            .parameters
            .iter()
            .map(|(name, p)| {
                let before = p.clone().into_data().try_to_vec::<f32>().unwrap();
                let after = before
                    .iter()
                    .zip(&grads[name])
                    .map(|(p, g)| *p - *g * (rate / norm.max(1.0)))
                    .collect::<Vec<_>>();
                (name.clone(), after)
            })
            .collect::<BTreeMap<_, _>>();
        let (mut model, sample) = setup();
        let before = model
            .parameters
            .iter()
            .map(|(n, p)| {
                (
                    n.clone(),
                    p.clone().into_data().try_to_vec::<f32>().unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let result = model.remix_single_target_sgd_probe(&sample, rate).unwrap();
        assert_eq!(result.updates, 1);
        assert_eq!(result.sample_cursor, 1);
        assert!(result.loss_decreased && result.loss_after < result.loss_before);
        assert!(result.all_updates_finite && result.all_gradients_finite);
        assert!(model.adam.is_none());
        let mut changed = false;
        for (name, p) in &model.parameters {
            let actual = p.clone().into_data().try_to_vec::<f32>().unwrap();
            changed |= actual != before[name];
            for (a, b) in actual.iter().zip(&expected[name]) {
                assert!((a - b).abs() < 1e-7, "{name}: {a} vs {b}");
            }
        }
        assert!(changed);
        let snapshot = model.inference_snapshot();
        assert!(!snapshot.training && !snapshot.device.is_autodiff());
        assert!(snapshot.parameters.values().all(|p| !p.is_autodiff()));
        assert!(
            model
                .remix_single_target_sgd_probe(&sample, rate)
                .unwrap_err()
                .contains("fresh")
        );
    }
    #[test]
    fn invalid_rate_and_existing_adam_reject_without_mutating_state() {
        let (mut model, sample) = setup();
        let rate = model.learning_rate;
        for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                model
                    .remix_single_target_sgd_probe(&sample, invalid)
                    .is_err()
            );
            assert_eq!(model.step, 0);
            assert_eq!(model.learning_rate, rate);
            assert_eq!(model.remix_state().unwrap().completed_samples, 0);
        }
        model.enable_adamw(Default::default(), 0.001).unwrap();
        assert!(
            model
                .remix_single_target_sgd_probe(&sample, 0.0001)
                .is_err()
        );
        assert!(model.adam.as_ref().unwrap().moments.is_empty());
        assert_eq!(model.step, 0);
    }
}
