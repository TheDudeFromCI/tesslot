//! Exact target-prefix replay: trade repeated forward work for fewer live prediction activations.
//! This is graph-preserving recomputation, not truncated BPTT or whole-block checkpointing.
use super::*;
use tesslot_core::{Phase, visible_to};
use tesslot_dataset::Sample;
impl Model {
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
