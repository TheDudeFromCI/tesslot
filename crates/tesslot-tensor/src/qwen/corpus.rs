//! Bounded random-pattern corpus with disjoint seed namespaces and resumable sampling.
use super::{
    echo::{LETTERS, Replay, pattern},
    *,
};

/// Dataset/sampling semantics, independent of model and checkpoint formats.
pub const DATASET_FORMAT: &str = "echo-seed-corpus/1";
/// Explicit split; held-out seeds never enter training through the sampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    /// Optimizer samples only this split.
    Train,
    /// Used for progress monitoring and engineering choices.
    Development,
    /// Explicit final evaluation; never automatically selected by training jobs.
    Test,
}
/// Small procedural dataset. Patterns are generated on demand instead of retained in RAM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    /// Must be `echo-seed-corpus/1`.
    pub format: String,
    /// Reproducible epoch shuffle, independent of pattern generation/model initialization.
    pub shuffle_seed: u64,
    /// Number of training patterns (1..4096).
    pub train_size: usize,
    /// Number of development patterns (1..16).
    pub development_size: usize,
    /// Number of final-test patterns (1..16).
    pub test_size: usize,
}
impl Default for Dataset {
    fn default() -> Self {
        Self {
            format: DATASET_FORMAT.into(),
            shuffle_seed: 42,
            train_size: 1024,
            development_size: 16,
            test_size: 16,
        }
    }
}
impl Dataset {
    /// Reject unknown versions and unbounded corpora before any tensor work.
    pub fn validate(&self) -> Result<(), String> {
        if self.format != DATASET_FORMAT
            || !(1..=4096).contains(&self.train_size)
            || !(1..=16).contains(&self.development_size)
            || !(1..=16).contains(&self.test_size)
        {
            return Err("unsupported echo corpus: require version 1, 1..4096 train and 1..16 development/test patterns".into());
        }
        Ok(())
    }
    /// Exact number of patterns in a split.
    pub fn len(&self, split: Split) -> usize {
        match split {
            Split::Train => self.train_size,
            Split::Development => self.development_size,
            Split::Test => self.test_size,
        }
    }
    /// Seed identity encodes its split in the upper two bits; indices are bounded.
    pub fn seed(&self, split: Split, index: usize) -> Result<u64, String> {
        self.validate()?;
        if index >= self.len(split) {
            return Err("echo corpus index exceeds split length".into());
        }
        Ok(match split {
            Split::Train => 0,
            Split::Development => 1u64 << 62,
            Split::Test => 2u64 << 62,
        } + index as u64)
    }
    /// Hash of the complete sampling/data definition for evaluation/checkpoint provenance.
    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        Ok(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(self).map_err(|e| e.to_string())?)
        ))
    }
    fn training_seed(&self, cursor: u64) -> Result<u64, String> {
        self.validate()?;
        // Fisher-Yates without replacement per epoch. Reconstructing this tiny index
        // vector avoids a version-dependent RNG object or mutable permutation in checkpoints.
        let epoch = cursor / self.train_size as u64;
        let mut rng = self.shuffle_seed ^ epoch.wrapping_mul(0xd1342543de82ef95);
        let mut indices = (0..self.train_size).collect::<Vec<_>>();
        for i in (1..indices.len()).rev() {
            rng = rng.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            indices.swap(i, (z % (i + 1) as u64) as usize);
        }
        self.seed(
            Split::Train,
            indices[(cursor % self.train_size as u64) as usize],
        )
    }
}
/// Sampling state saved atomically with weights at complete optimizer boundaries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingState {
    /// Immutable corpus and shuffle definition.
    pub dataset: Dataset,
    /// Successfully committed pattern updates, including previous epochs.
    pub completed_samples: u64,
}
impl TrainingState {
    /// Next training seed; monitoring/evaluation cannot advance this cursor.
    pub fn next_seed(&self) -> Result<u64, String> {
        self.dataset.training_seed(self.completed_samples)
    }
}

/// Per-pattern measurements, including actual free-running output for inspection.
#[derive(Debug, Serialize)]
pub struct Case {
    /// Decimal seed string in the requested split, preserving all bits in browser clients.
    pub seed: String,
    /// Echo-only mean cross entropy with reference prefix history.
    pub loss: f32,
    /// Same objective with source visibility disabled in every layer.
    pub source_masked_loss: f32,
    /// Greedy predictions fed back into the echo window.
    pub replay: Replay,
    /// Free-running accuracy under source masking.
    pub source_masked_accuracy: f32,
}
/// Measured aggregate, never substituted for a training loss or fabricated GPU metric.
#[derive(Debug, Serialize)]
pub struct Evaluation {
    /// `echo-evaluation/1`.
    pub format: &'static str,
    /// Complete dataset definition hash.
    pub dataset_id: String,
    /// Held-out split actually measured.
    pub split: Split,
    /// Model optimizer step at evaluation.
    pub model_step: u64,
    /// Number of measured patterns.
    pub samples: usize,
    /// Effective target weight: samples × 12, all echo labels have weight one.
    pub target_weight: usize,
    /// Weighted echo-only teacher-forced loss.
    pub loss: f32,
    /// Accuracy given reference prefix history (not a free-running rollout).
    pub teacher_forced_accuracy: f32,
    /// Accuracy when previous predictions are model-generated.
    pub free_running_accuracy: f32,
    /// Fraction of entire patterns copied correctly.
    pub exact_match_rate: f32,
    /// Weighted teacher-forced loss when source keys are hidden from echo rows.
    pub source_masked_loss: f32,
    /// Free-running accuracy when source keys are hidden from echo rows.
    pub source_masked_accuracy: f32,
    /// Actual examples for replay inspection.
    pub cases: Vec<Case>,
}
impl Model {
    /// Enter a fixed corpus protocol. Reapplying the same definition preserves progress.
    pub fn start_echo_corpus(&mut self, dataset: Dataset) -> Result<(), String> {
        dataset.validate()?;
        if !self.is_echo() {
            return Err("echo corpus requires a two-window model".into());
        }
        if let Some(state) = &self.corpus {
            if state.dataset != dataset {
                return Err("cannot silently replace a checkpoint's echo corpus definition".into());
            }
        } else {
            self.corpus = Some(TrainingState {
                dataset,
                completed_samples: 0,
            });
        }
        Ok(())
    }
    /// Sampling identity and progress for checkpoint/API diagnostics.
    pub fn echo_corpus(&self) -> Option<&TrainingState> {
        self.corpus.as_ref()
    }
    /// One full-parameter update; advance the sampler only after the update commits.
    pub fn train_echo_corpus_step(&mut self) -> Result<Metrics, String> {
        let state = self
            .corpus
            .as_ref()
            .ok_or("start an echo corpus before training")?;
        let next = state
            .completed_samples
            .checked_add(1)
            .ok_or("echo sampler cursor overflow")?;
        let source = pattern(state.next_seed()?);
        let metric = self.update(self.echo_loss(&source, true)?)?;
        self.corpus
            .as_mut()
            .ok_or("echo corpus disappeared")?
            .completed_samples = next;
        Ok(metric)
    }
    /// Evaluate a bounded held-out prefix without updating weights or sampling state.
    pub fn evaluate_echo_corpus(&self, split: Split, samples: usize) -> Result<Evaluation, String> {
        let state = self.corpus.as_ref().ok_or("no echo corpus registered")?;
        if split == Split::Train || samples == 0 || samples > state.dataset.len(split) {
            return Err("evaluation requires 1..split_size development/test patterns".into());
        }
        let inference = self.inference_snapshot();
        let mut cases = Vec::new();
        let mut forced_correct = 0;
        let mut correct = 0;
        let mut masked_correct = 0;
        let mut exact = 0;
        for i in 0..samples {
            let seed = state.dataset.seed(split, i)?;
            let source = pattern(seed);
            let logits = inference
                .echo_logits(&source, &source[..LETTERS - 1], LETTERS, true)?
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|e| e.to_string())?;
            let vocab = self.config.vocab_size;
            forced_correct += logits
                .chunks_exact(vocab)
                .zip(&source)
                .filter(|(row, target)| {
                    row.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|v| v.0 as u32)
                        == Some(**target)
                })
                .count();
            let replay = inference.replay_echo(&source, true)?;
            let masked = inference.replay_echo(&source, false)?;
            let n = replay
                .echo_token_ids
                .iter()
                .zip(&source)
                .filter(|(a, b)| a == b)
                .count();
            correct += n;
            exact += usize::from(n == LETTERS);
            masked_correct += masked
                .echo_token_ids
                .iter()
                .zip(&source)
                .filter(|(a, b)| a == b)
                .count();
            let loss = inference.evaluate_echo(&source, true)?;
            let source_masked_loss = inference.evaluate_echo(&source, false)?;
            if !loss.is_finite() || !source_masked_loss.is_finite() {
                return Err("nonfinite echo evaluation loss".into());
            }
            cases.push(Case {
                seed: seed.to_string(),
                loss,
                source_masked_loss,
                replay,
                source_masked_accuracy: masked.accuracy,
            });
        }
        let targets = samples * LETTERS;
        Ok(Evaluation {
            format: "echo-evaluation/1",
            dataset_id: state.dataset.identity()?,
            split,
            model_step: self.step,
            samples,
            target_weight: targets,
            loss: cases.iter().map(|c| c.loss).sum::<f32>() / samples as f32,
            teacher_forced_accuracy: forced_correct as f32 / targets as f32,
            free_running_accuracy: correct as f32 / targets as f32,
            exact_match_rate: exact as f32 / samples as f32,
            source_masked_loss: cases.iter().map(|c| c.source_masked_loss).sum::<f32>()
                / samples as f32,
            source_masked_accuracy: masked_correct as f32 / targets as f32,
            cases,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_seeds_are_disjoint_and_epoch_shuffles_are_complete() {
        let d = Dataset::default();
        let mut all = std::collections::BTreeSet::new();
        let mut patterns = std::collections::BTreeSet::new();
        for split in [Split::Train, Split::Development, Split::Test] {
            for i in 0..d.len(split) {
                assert!(all.insert(d.seed(split, i).unwrap()));
                assert!(
                    patterns.insert(pattern(d.seed(split, i).unwrap())),
                    "default corpus has duplicate pattern content"
                );
            }
        }
        for epoch in 0..2 {
            let seeds = (0..d.train_size)
                .map(|i| d.training_seed((epoch * d.train_size + i) as u64).unwrap())
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(seeds, (0..d.train_size as u64).collect());
        }
        assert_eq!(d.training_seed(10), d.clone().training_seed(10));
        assert_ne!(
            d.training_seed(10),
            d.training_seed(10 + d.train_size as u64)
        );
        assert!(d.seed(Split::Development, 16).is_err());
        let mut bad = d.clone();
        bad.format = "future/2".into();
        assert!(bad.validate().is_err());
        assert_ne!(
            d.identity().unwrap(),
            Dataset {
                shuffle_seed: 43,
                ..d.clone()
            }
            .identity()
            .unwrap()
        );
    }
}
