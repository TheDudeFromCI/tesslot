//! Tiny CPU full-parameter SFT with epoch-boundary checkpoints; no persistent jobs yet.
//! Each trajectory retains its graph through every storage chunk. Numerator
//! gradients accumulate across samples and are normalized once per optimizer step.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tesslot_core::ModelConfig;
use tesslot_dataset::Sample;
use tesslot_model::ReferenceModel;
use tesslot_tensor::{Cache, Model, Parameters, Prediction};

/// Versioned, checksummed epoch-boundary checkpoint storage.
pub mod checkpoint;
/// Allocation-free, per-device memory planning for future pretrained training.
pub mod feasibility;

type State = BTreeMap<String, Vec<Vec<f32>>>;

/// Full-parameter optimizer; all named leaves use the same hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Optimizer {
    /// Plain SGD without momentum or weight decay.
    Sgd,
    /// Bias-corrected Adam with decoupled weight decay on every parameter.
    AdamW {
        /// First-moment decay, in [0,1).
        beta1: f32,
        /// Second-moment decay, in [0,1).
        beta2: f32,
        /// Positive denominator epsilon.
        epsilon: f32,
        /// Nonnegative decoupled decay.
        weight_decay: f32,
    },
}

/// Validated run controls, independent of HTTP and model architecture config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Number of complete passes through the export.
    pub epochs: usize,
    /// Maximum complete trajectories per optimizer step; flush tails each epoch.
    pub accumulation_samples: usize,
    /// Positive peak learning rate.
    pub learning_rate: f32,
    /// Linear warmup length in optimizer steps; zero disables warmup.
    pub warmup_steps: u64,
    /// Optional linear decay duration after reaching peak; zero is invalid.
    pub decay_steps: Option<u64>,
    /// Optional positive global L2 gradient norm bound.
    pub clip_norm: Option<f64>,
    /// Optimizer algorithm and hyperparameters.
    pub optimizer: Optimizer,
    /// Split and shuffle seed; parameter initialization uses ModelConfig::seed.
    pub seed: u64,
    /// Training fraction in basis points, assigned by trajectory UUID.
    pub train_basis_points: u16,
    /// Validation fraction; remaining trajectories are held out.
    pub validation_basis_points: u16,
    /// Bounded shuffle buffer of complete samples.
    pub shuffle_buffer: usize,
    /// CPU attached graph row admission bound per trajectory.
    pub max_sample_rows: usize,
    /// Enable the existing operation-level graph recomputation path.
    pub recompute: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            epochs: 1,
            accumulation_samples: 1,
            learning_rate: 0.001,
            warmup_steps: 0,
            decay_steps: None,
            clip_norm: Some(1.0),
            optimizer: Optimizer::AdamW {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: 0.01,
            },
            seed: 42,
            train_basis_points: 9000,
            validation_basis_points: 1000,
            shuffle_buffer: 8,
            max_sample_rows: 512,
            recompute: false,
        }
    }
}
impl Config {
    /// Reject invalid controls before allocating model/optimizer state.
    pub fn validate(&self) -> Result<(), String> {
        if self.epochs == 0
            || self.accumulation_samples == 0
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || self.decay_steps == Some(0)
            || self.clip_norm.is_some_and(|v| !v.is_finite() || v <= 0.0)
            || self.train_basis_points == 0
            || u32::from(self.train_basis_points) + u32::from(self.validation_basis_points) > 10000
            || self.shuffle_buffer == 0
            || self.max_sample_rows == 0
        {
            return Err("invalid training controls: require positive sizes/rate/clip/decay and split total <= 10000".into());
        }
        if let Optimizer::AdamW {
            beta1,
            beta2,
            epsilon,
            weight_decay,
        } = self.optimizer
            && (!(0.0..1.0).contains(&beta1)
                || !(0.0..1.0).contains(&beta2)
                || !epsilon.is_finite()
                || epsilon <= 0.0
                || !weight_decay.is_finite()
                || weight_decay < 0.0)
        {
            return Err("invalid AdamW beta, epsilon or weight decay".into());
        }
        Ok(())
    }
    /// One-based optimizer-step rate; first step (or last warmup step) is peak.
    /// Linear decay reaches zero `decay_steps` steps after that peak.
    pub fn rate(&self, step: u64) -> f32 {
        let peak = self.warmup_steps.max(1);
        let scale = if step <= self.warmup_steps {
            step as f64 / self.warmup_steps as f64
        } else if let Some(duration) = self.decay_steps {
            1.0 - (step.saturating_sub(peak) as f64 / duration as f64).min(1.0)
        } else {
            1.0
        };
        (f64::from(self.learning_rate) * scale) as f32
    }
}

/// Measured teacher-forced statistics, weighted by the export's target weights.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Metrics {
    /// Complete trajectories evaluated.
    pub samples: usize,
    /// Sum of weighted negative log likelihoods before normalization.
    pub loss_numerator: f64,
    /// Sum of effective target weights.
    pub effective_weight: f64,
    /// Sum of weights whose argmax equals the target.
    pub correct_weight: f64,
}
impl Metrics {
    /// Objective uses the same max(weight,1) denominator as training.
    pub fn loss(&self) -> f64 {
        self.loss_numerator / self.effective_weight.max(1.0)
    }
    /// Weighted accuracy; zero if there are no supervised targets.
    pub fn accuracy(&self) -> f64 {
        if self.effective_weight == 0.0 {
            0.0
        } else {
            self.correct_weight / self.effective_weight
        }
    }
    /// Aggregate numerators, never average per-sample means.
    pub fn merge(&mut self, other: &Self) -> Result<(), String> {
        self.samples += other.samples;
        self.loss_numerator += other.loss_numerator;
        self.effective_weight += other.effective_weight;
        self.correct_weight += other.correct_weight;
        if [
            self.loss_numerator,
            self.effective_weight,
            self.correct_weight,
        ]
        .iter()
        .any(|v| !v.is_finite())
        {
            return Err("nonfinite aggregated metrics".into());
        }
        Ok(())
    }
}

/// A completed optimizer boundary, with pre-update objective and unclipped norm.
#[derive(Debug, Serialize)]
pub struct Step {
    /// Number of committed updates.
    pub step: u64,
    /// Actual scheduled learning rate.
    pub learning_rate: f32,
    /// L2 norm after normalization, before clipping.
    pub gradient_norm: f64,
    /// Multiplier applied to every normalized gradient.
    pub clip_scale: f64,
    /// Measured objective and accuracy before this update.
    pub metrics: Metrics,
}

/// CPU trainer with FP32 master parameters and FP32 Adam moments.
/// Host f64 parameter interchange is rounded to FP32 at every committed update.
pub struct Trainer {
    model: ModelConfig,
    config: Config,
    values: Parameters,
    first: State,
    second: State,
    step: u64,
}
impl Trainer {
    /// Seed every trainable leaf from the existing reference initializer.
    pub fn new(model: ModelConfig, config: Config) -> Result<Self, String> {
        config.validate()?;
        let values = ReferenceModel::new(model.clone())
            .map_err(|e| e.to_string())?
            .parameters();
        Self::from_parameters(model, config, values)
    }
    /// Start a fresh optimizer from an explicit complete parameter payload.
    /// This is not checkpoint resume: moments and the step counter start at zero.
    pub fn from_parameters(
        model: ModelConfig,
        config: Config,
        mut values: Parameters,
    ) -> Result<Self, String> {
        config.validate()?;
        ReferenceModel::from_parameters(model.clone(), &values).map_err(|e| e.to_string())?;
        for v in values.values_mut().flatten().flatten() {
            *v = f64::from(*v as f32);
            if !v.is_finite() {
                return Err("parameter outside finite FP32 range".into());
            }
        }
        let zeros = || {
            values
                .iter()
                .map(|(n, m)| (n.clone(), m.iter().map(|r| vec![0.0; r.len()]).collect()))
                .collect()
        };
        let (first, second) = if matches!(config.optimizer, Optimizer::AdamW { .. }) {
            (zeros(), zeros())
        } else {
            (State::new(), State::new())
        };
        Ok(Self {
            model,
            config,
            values,
            first,
            second,
            step: 0,
        })
    }
    /// Current logical parameter payload, including every trainable leaf.
    pub fn parameters(&self) -> &Parameters {
        &self.values
    }
    /// Completed optimizer updates; unsuccessful steps do not advance it.
    pub fn steps(&self) -> u64 {
        self.step
    }

    /// Evaluate a complete trajectory with fresh non-recording cache state.
    pub fn evaluate(&self, sample: &Sample) -> Result<Metrics, String> {
        check_sample(sample, self.config.max_sample_rows)?;
        let model = Model::inference(self.model.clone(), &self.values)?;
        let predictions = forward_sample(&model, sample)?;
        measure(&model, &predictions, sample).map(|(_, m)| m)
    }

    /// Accumulate full-trajectory numerator gradients, normalize once, then update.
    /// Any failure leaves parameters, moments and schedule position unchanged.
    pub fn train_batch(&mut self, samples: &[&Sample]) -> Result<Step, String> {
        if samples.is_empty() || samples.len() > self.config.accumulation_samples {
            return Err(
                "optimizer batch must contain 1..=accumulation_samples trajectories".into(),
            );
        }
        for sample in samples {
            check_sample(sample, self.config.max_sample_rows)?;
        }
        let mut gradients: Parameters = self
            .values
            .iter()
            .map(|(n, m)| (n.clone(), m.iter().map(|r| vec![0.0; r.len()]).collect()))
            .collect();
        let mut metrics = Metrics::default();
        for sample in samples {
            let model = Model::new(self.model.clone(), &self.values, self.config.recompute)?;
            let predictions = forward_sample(&model, sample)?;
            let (loss, measured) = measure(&model, &predictions, sample)?;
            metrics.merge(&measured)?;
            let sample_gradients = model.gradients(&loss)?;
            for (name, matrix) in &mut gradients {
                for (g, v) in matrix
                    .iter_mut()
                    .flatten()
                    .zip(sample_gradients[name].iter().flatten())
                {
                    *g += v;
                }
            }
        }
        if metrics.effective_weight == 0.0 {
            return Err("optimizer batch has no supervised target weight".into());
        }
        let denominator = metrics.effective_weight.max(1.0);
        let mut norm = 0.0_f64;
        for g in gradients.values_mut().flatten().flatten() {
            *g /= denominator;
            if !g.is_finite() {
                return Err("nonfinite accumulated gradient".into());
            }
            norm = norm.hypot(*g);
        }
        if !norm.is_finite() {
            return Err("nonfinite gradient norm".into());
        }
        let scale = self
            .config
            .clip_norm
            .map_or(1.0, |bound| if norm > bound { bound / norm } else { 1.0 });
        let step = self.step.checked_add(1).ok_or("optimizer step overflow")?;
        let rate = self.config.rate(step);
        let mut values = self.values.clone();
        let mut first = self.first.clone();
        let mut second = self.second.clone();
        for (name, matrix) in &mut values {
            for (i, row) in matrix.iter_mut().enumerate() {
                for (j, v) in row.iter_mut().enumerate() {
                    let g = (gradients[name][i][j] * scale) as f32;
                    if !g.is_finite() {
                        return Err(format!("gradient outside finite FP32 range: {name}"));
                    }
                    let old = *v as f32;
                    let next = match self.config.optimizer {
                        Optimizer::Sgd => old - rate * g,
                        Optimizer::AdamW {
                            beta1,
                            beta2,
                            epsilon,
                            weight_decay,
                        } => {
                            let m = beta1 * first[name][i][j] + (1.0 - beta1) * g;
                            let s = beta2 * second[name][i][j] + (1.0 - beta2) * g * g;
                            if !m.is_finite() || !s.is_finite() {
                                return Err(format!("nonfinite AdamW moment: {name}"));
                            }
                            first.get_mut(name).unwrap()[i][j] = m;
                            second.get_mut(name).unwrap()[i][j] = s;
                            let m_hat = f64::from(m) / (1.0 - f64::from(beta1).powf(step as f64));
                            let s_hat = f64::from(s) / (1.0 - f64::from(beta2).powf(step as f64));
                            let direction = (m_hat / (s_hat.sqrt() + f64::from(epsilon))) as f32;
                            old * (1.0 - rate * weight_decay) - rate * direction
                        }
                    };
                    if !next.is_finite() {
                        return Err(format!("nonfinite optimizer update: {name}"));
                    }
                    *v = f64::from(next);
                }
            }
        }
        self.values = values;
        self.first = first;
        self.second = second;
        self.step = step;
        Ok(Step {
            step,
            learning_rate: rate,
            gradient_norm: norm,
            clip_scale: scale,
            metrics,
        })
    }
}

fn check_sample(sample: &Sample, limit: usize) -> Result<(), String> {
    if sample.rows.is_empty()
        || sample.rows.len() > limit
        || sample.chunk_ends.last() != Some(&sample.rows.len())
    {
        return Err("empty, oversized or incomplete trajectory".into());
    }
    let mut start = 0;
    for &end in &sample.chunk_ends {
        if end <= start
            || end > sample.rows.len()
            || (end < sample.rows.len() && sample.rows[end - 1].tick == sample.rows[end].tick)
        {
            return Err("invalid storage boundary or split tick".into());
        }
        start = end;
    }
    if sample.rows.iter().enumerate().any(|(i, r)| r.index != i) {
        return Err("noncontiguous row indices".into());
    }
    Ok(())
}

fn forward_sample(model: &Model, sample: &Sample) -> Result<Vec<Prediction>, String> {
    let mut cache = Cache::default();
    let mut predictions = Vec::new();
    let mut start = 0;
    for &end in &sample.chunk_ends {
        while start < end {
            let next = start
                + sample.rows[start..end].partition_point(|r| r.tick == sample.rows[start].tick);
            predictions.extend(model.tick(
                &sample.rows[start..next],
                &sample.metadata.windows,
                &mut cache,
            )?);
            start = next;
        }
    }
    Ok(predictions)
}

fn measure(
    model: &Model,
    predictions: &[Prediction],
    sample: &Sample,
) -> Result<(tesslot_tensor::Loss, Metrics), String> {
    let (loss, weight) = model.loss_parts(predictions, &sample.rows)?;
    let numerator = loss
        .clone()
        .into_data_as::<f64>()
        .try_to_vec::<f64>()
        .map_err(|e| e.to_string())?[0];
    if !numerator.is_finite() {
        return Err("nonfinite objective".into());
    }
    let mut correct_weight = 0.0;
    for p in predictions {
        let logits = p
            .logits
            .clone()
            .into_data_as::<f64>()
            .try_to_vec::<f64>()
            .map_err(|e| e.to_string())?;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite prediction".into());
        }
        let target = &sample.rows[p.row_index];
        // First maximum wins, giving a deterministic tie policy.
        let best = logits
            .iter()
            .enumerate()
            .fold(0, |best, (i, v)| if *v > logits[best] { i } else { best });
        if best == target.token_id as usize {
            correct_weight += target.loss_weight;
        }
    }
    Ok((
        loss,
        Metrics {
            samples: 1,
            loss_numerator: numerator,
            effective_weight: weight,
            correct_weight,
        },
    ))
}
