//! FP32 Qwen3 smoke runtime, plus a versioned two-window exchange experiment.
mod checks;
/// Procedural multi-pattern echo training and held-out evaluation.
pub mod corpus;
pub mod echo;
/// Explicit experimental optimizer configuration.
pub mod optimizer;
mod recompute;
/// Native full-trajectory Qwen training on validated Trajectory Remix exports.
pub mod remix;
/// Explicit hardware placement and mixed matmul precision.
pub mod runtime;
mod window;
use burn_tensor::{
    Bool, DType, Device, Tensor, TensorData,
    activation::{log_softmax, silu, softmax},
};
use checks::finite;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Cursor, path::Path};
use tesslot_artifacts::{Mapping, Tensors, encode_f32_with_metadata, import, inspect};
use tesslot_model::qwen::Config;
type T = Tensor<2>;
/// Actual optimizer metrics measured at a completed update boundary.
#[derive(Debug, Serialize)]
pub struct Metrics {
    /// Cross entropy before the update.
    pub loss: f32,
    /// L2 gradient norm before clipping.
    pub gradient_norm: f32,
    /// Completed updates.
    pub step: u64,
}
/// Tiny executable HF-shaped Qwen, tied head and GQA included. Sampling is greedy.
pub struct Model {
    id: usize,
    config: Config,
    parameters: BTreeMap<String, T>,
    device: Device,
    plan: runtime::Plan,
    training: bool,
    checkpoint_identity: Option<String>,
    exchange_windows: Option<usize>,
    corpus: Option<corpus::TrainingState>,
    remix: Option<remix::TrainingState>,
    adam: Option<optimizer::Adam>,
    /// Completed optimizer updates.
    pub step: u64,
    /// Constant optimizer learning rate.
    pub learning_rate: f32,
}
impl Model {
    /// Load local single-file tensors within small experimental limits; no network access.
    pub fn load(config: Config, bytes: &[u8], training: bool) -> Result<Self, String> {
        Self::load_architecture(config, bytes, training, None)
    }
    fn load_architecture(
        config: Config,
        bytes: &[u8],
        training: bool,
        exchange_windows: Option<usize>,
    ) -> Result<Self, String> {
        config.validate()?;
        if config.head_dim > 64
            || config.num_attention_heads > 8
            || config.num_key_value_heads > 8
            || config.intermediate_size > 256
            || config.hidden_size > 64
            || config.num_hidden_layers > 4
            || config.vocab_size > 256
            || config.max_position_embeddings > 256
            || bytes.len() > 2 * 1024 * 1024
        {
            return Err("tiny Qwen experiment exceeds model/context/byte limits".into());
        }
        let expected = echo::shapes(&config, exchange_windows)?;
        let sources = expected.keys().map(|n| (n.clone(), n.clone())).collect();
        let weights = import(
            &mut Cursor::new(bytes),
            &Mapping {
                expected,
                sources,
                initialize: Default::default(),
            },
            &Tensors::new(),
            2 * 1024 * 1024,
        )
        .map_err(|e| e.to_string())?;
        let device = if training {
            Device::flex().autodiff()
        } else {
            Device::flex()
        };
        let parameters = weights
            .into_iter()
            .map(|(n, (shape, v))| {
                let shape = if shape.len() == 1 {
                    [1, shape[0]]
                } else {
                    [shape[0], shape[1]]
                };
                let p = T::from_data(TensorData::new(v, shape), (&device, DType::F32));
                (n, if training { p.require_grad() } else { p })
            })
            .collect();
        Ok(Self {
            id: crate::NEXT_MODEL.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            plan: runtime::Plan::cpu(config.num_hidden_layers, 2 * 1024 * 1024),
            config,
            parameters,
            device,
            training,
            checkpoint_identity: Some("ascii/1".into()),
            exchange_windows,
            corpus: None,
            remix: None,
            adam: None,
            step: 0,
            learning_rate: 0.2,
        })
    }
    /// Load the committed random fixture; its tokenizer is explicitly ASCII for this experiment.
    pub fn demo(training: bool) -> Result<Self, String> {
        let c = Config::from_json(include_bytes!("../../../fixtures/tiny-qwen/config.json"))?;
        Self::load(
            c,
            include_bytes!("../../../fixtures/tiny-qwen/model.safetensors"),
            training,
        )
    }
    fn linear(&self, x: T, name: &str) -> T {
        let device = x.device();
        self.matmul(
            x,
            self.parameters[name].clone().to_device(&device).transpose(),
        )
    }
    fn norm(&self, x: T, name: &str) -> T {
        let eps = self.config.rms_norm_eps;
        let device = x.device();
        let y = x.clone() / (x.powf_scalar(2.0).mean_dim(1) + eps).sqrt();
        y * self.parameters[name].clone().to_device(&device)
    }
    fn rope(&self, x: T) -> T {
        let positions = (0..x.dims()[0]).collect::<Vec<_>>();
        self.rope_positions(x, &positions)
    }
    fn rope_positions(&self, x: T, positions: &[usize]) -> T {
        let [seq, d] = x.dims();
        let half = d / 2;
        let device = x.device();
        let mut cosine = Vec::with_capacity(seq * half);
        let mut sine = Vec::with_capacity(seq * half);
        for position in positions {
            for i in 0..half {
                let (sin, cos) = (*position as f64
                    / self.config.rope_theta.powf((2 * i) as f64 / d as f64))
                .sin_cos();
                sine.push(sin as f32);
                cosine.push(cos as f32);
            }
        }
        let cos = T::from_data(TensorData::new(cosine, [seq, half]), (&device, DType::F32));
        let sin = T::from_data(TensorData::new(sine, [seq, half]), (&device, DType::F32));
        let first = x.clone().slice([0..seq, 0..half]);
        let second = x.slice([0..seq, half..d]);
        T::cat(
            vec![
                first.clone() * cos.clone() - second.clone() * sin.clone(),
                second * cos + first * sin,
            ],
            1,
        )
    }

    /// Full causal Qwen forward with trainable RMS scales, split-half RoPE and GQA.
    pub fn forward(&self, ids: &[u32]) -> Result<T, String> {
        if self.exchange_windows.is_some() {
            return Err("Use the architecture-specific window forward method".into());
        }
        let c = &self.config;
        let n = ids.len();
        let d = c.hidden_size;
        if n == 0
            || n > c.max_position_embeddings
            || ids.iter().any(|i| *i as usize >= c.vocab_size)
        {
            return Err("invalid or overlong tiny Qwen token context".into());
        }
        let mut x = T::cat(
            ids.iter()
                .map(|i| {
                    self.parameters["model.embed_tokens.weight"]
                        .clone()
                        .slice([*i as usize..*i as usize + 1, 0..d])
                })
                .collect(),
            0,
        );
        let mask = Tensor::<2, Bool>::from_data(
            TensorData::new(
                (0..n)
                    .flat_map(|i| (0..n).map(move |j| j > i))
                    .collect::<Vec<_>>(),
                [n, n],
            ),
            &self.device,
        );
        let heads = c.num_attention_heads;
        let kv_heads = c.num_key_value_heads;
        let dh = c.head_dim;
        for l in 0..c.num_hidden_layers {
            let device = self.layer_device(l);
            x = x.to_device(&device);
            let prefix = format!("model.layers.{l}.");
            let h = self.norm(x.clone(), &(prefix.clone() + "input_layernorm.weight"));
            let q = self.linear(h.clone(), &(prefix.clone() + "self_attn.q_proj.weight"));
            let k = self.linear(h.clone(), &(prefix.clone() + "self_attn.k_proj.weight"));
            let v = self.linear(h, &(prefix.clone() + "self_attn.v_proj.weight"));
            let mut responses = Vec::new();
            for head in 0..heads {
                let kv = head / (heads / kv_heads);
                let q = self.rope(self.norm(
                    q.clone().slice([0..n, head * dh..(head + 1) * dh]),
                    &(prefix.clone() + "self_attn.q_norm.weight"),
                ));
                let k = self.rope(self.norm(
                    k.clone().slice([0..n, kv * dh..(kv + 1) * dh]),
                    &(prefix.clone() + "self_attn.k_norm.weight"),
                ));
                let v = v.clone().slice([0..n, kv * dh..(kv + 1) * dh]);
                let scores = (self.matmul(q, k.transpose()) / (dh as f32).sqrt())
                    .mask_fill(mask.clone().to_device(&device), f32::NEG_INFINITY);
                responses.push(self.matmul(softmax(scores, 1), v));
            }
            x = x + self.linear(
                T::cat(responses, 1),
                &(prefix.clone() + "self_attn.o_proj.weight"),
            );
            let h = self.norm(
                x.clone(),
                &(prefix.clone() + "post_attention_layernorm.weight"),
            );
            let gate = self.linear(h.clone(), &(prefix.clone() + "mlp.gate_proj.weight"));
            let up = self.linear(h, &(prefix.clone() + "mlp.up_proj.weight"));
            x = x + self.linear(up * silu(gate), &(prefix + "mlp.down_proj.weight"));
        }
        let x = self.norm(x.to_device(&self.device), "model.norm.weight");
        Ok(self.linear(
            x,
            if c.tie_word_embeddings {
                "model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            },
        ))
    }
    fn loss(&self, inputs: &[u32], targets: &[u32]) -> Result<Tensor<1>, String> {
        if inputs.len() != targets.len()
            || targets
                .iter()
                .any(|t| *t as usize >= self.config.vocab_size)
        {
            return Err("invalid next-token targets".into());
        }
        let p = log_softmax(self.forward(inputs)?, 1);
        let terms = targets
            .iter()
            .enumerate()
            .map(|(i, t)| {
                p.clone()
                    .slice([i..i + 1, *t as usize..*t as usize + 1])
                    .sum()
            })
            .collect::<Vec<_>>();
        Ok(-Tensor::<1>::cat(terms, 0).mean())
    }
    /// Measure next-token cross entropy without updating.
    pub fn evaluate(&self, inputs: &[u32], targets: &[u32]) -> Result<f32, String> {
        Ok(self
            .loss(inputs, targets)?
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| e.to_string())?[0])
    }
    /// Full-parameter deterministic SGD, with global gradient clipping and finite checks.
    pub fn train_step(&mut self, inputs: &[u32], targets: &[u32]) -> Result<Metrics, String> {
        if self.exchange_windows.is_some() {
            return Err(
                "Use the echo, corpus or Remix trainer for the configured window architecture"
                    .into(),
            );
        }
        self.update(self.loss(inputs, targets)?)
    }
    fn update(&mut self, loss: Tensor<1>) -> Result<Metrics, String> {
        if !self.training || !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(
                "training requires an autodiff model and positive finite learning rate".into(),
            );
        }
        if self
            .adam
            .as_ref()
            .is_some_and(|a| self.learning_rate * a.config.weight_decay >= 1.0)
        {
            return Err("AdamW learning rate times decay must be below one".into());
        }
        let value = loss
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| e.to_string())?[0];
        if !value.is_finite() {
            return Err("nonfinite loss".into());
        }
        let gradients = loss.backward();
        let mut gs = BTreeMap::new();
        for (name, p) in &self.parameters {
            let g = match p.grad(&gradients) {
                Some(g) => g,
                None if self.remix.is_some() => p.clone().zeros_like().detach(),
                None => return Err(format!("missing gradient {name}")),
            };
            gs.insert(name.clone(), g);
        }
        self.apply_gradients(value, gs)
    }
    fn apply_gradients(&mut self, value: f32, gs: BTreeMap<String, T>) -> Result<Metrics, String> {
        if !self.training
            || !value.is_finite()
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
        {
            return Err("invalid training state or loss".into());
        }
        if self
            .adam
            .as_ref()
            .is_some_and(|a| self.learning_rate * a.config.weight_decay >= 1.0)
        {
            return Err("AdamW learning rate times decay must be below one".into());
        }
        let next_step = self.step.checked_add(1).ok_or("training step overflow")?;
        let mut norm = 0.0f64;
        for (name, g) in &gs {
            if !finite(g).map_err(|e| format!("gradient {name}: {e}"))? {
                return Err(format!("nonfinite gradient {name}"));
            }
            norm += g
                .clone()
                .powf_scalar(2.0)
                .sum()
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|e| e.to_string())?[0] as f64;
        }
        let magnitude = norm.sqrt() as f32;
        if !magnitude.is_finite() {
            return Err("nonfinite gradient norm".into());
        }
        let scale = self.learning_rate / magnitude.max(1.0);
        let adam_next = self
            .adam
            .as_ref()
            .map(|a| a.next(&gs, 1.0 / magnitude.max(1.0)))
            .transpose()?;
        let mut next = BTreeMap::new();
        for (name, p) in &self.parameters {
            let updated = if let Some(adam) = &adam_next {
                adam.parameter(name, p.clone().detach(), self.learning_rate)
            } else {
                p.clone().detach() - gs[name].clone() * scale
            };
            if !finite(&updated).map_err(|e| format!("updated parameter {name}: {e}"))? {
                return Err(format!("nonfinite updated parameter {name}"));
            }
            next.insert(name.clone(), updated.detach().require_grad());
        }
        self.parameters = next;
        self.adam = adam_next;
        self.step = next_step;
        Ok(Metrics {
            loss: value,
            gradient_norm: magnitude,
            step: self.step,
        })
    }
    /// Greedy ASCII experiment. Returned IDs remain authoritative even for nonprinting tokens.
    pub fn generate(&self, prompt: &[u32], count: usize) -> Result<Vec<u32>, String> {
        if count > 32
            || prompt.is_empty()
            || prompt.len() + count > self.config.max_position_embeddings
        {
            return Err("generation requires prompt + <=32 tokens within context".into());
        }
        if self.training {
            let inference = self.inference_snapshot();
            return inference.generate(prompt, count);
        }
        let mut ids = prompt.to_vec();
        let mut output = Vec::new();
        for _ in 0..count {
            let logits = self.forward(&ids)?;
            let n = ids.len();
            let last = logits
                .slice([n - 1..n, 0..self.config.vocab_size])
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|e| e.to_string())?;
            if last.iter().any(|x| !x.is_finite()) {
                return Err("nonfinite generation logits".into());
            }
            let token = last
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .ok_or("empty logits")?
                .0 as u32;
            ids.push(token);
            output.push(token);
            if token as usize == self.config.eos_token_id {
                break;
            }
        }
        Ok(output)
    }
    /// Publish a new checksummed experimental SGD checkpoint directory at an update boundary.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if self.remix.is_none() && self.checkpoint_identity.is_none() {
            return Err("checkpoint requires a bound tokenizer identity; use from_pretrained for converted models".into());
        }

        if path.exists() {
            return Err("checkpoint destination already exists".into());
        }
        let mut tensors = Tensors::new();
        let shapes = echo::shapes(&self.config, self.exchange_windows)?;
        for (name, p) in &self.parameters {
            tensors.insert(
                name.clone(),
                (
                    shapes[name].clone(),
                    p.clone()
                        .into_data()
                        .try_to_vec::<f32>()
                        .map_err(|e| e.to_string())?,
                ),
            );
        }
        let bytes =
            encode_f32_with_metadata(&tensors, BTreeMap::from([("format".into(), "pt".into())]))
                .map_err(|e| e.to_string())?;
        let (optimizer_bytes, adamw) = if let Some(adam) = &self.adam {
            let (bytes, manifest) = adam.encode(&shapes)?;
            (Some(bytes), Some(manifest))
        } else {
            (None, None)
        };
        let manifest = Checkpoint {
            format: if self.remix.is_some() {
                if self.adam.is_some() {
                    "tiny-qwen-remix-adamw/1"
                } else {
                    "tiny-qwen-remix-sgd/1"
                }
            } else if self.corpus.is_some() && self.adam.is_some() {
                "tiny-qwen-echo-corpus-adamw/1"
            } else if self.corpus.is_some() {
                "tiny-qwen-echo-corpus-sgd/1"
            } else if self.exchange_windows.is_some() {
                "tiny-qwen-echo-sgd/1"
            } else {
                "tiny-qwen-sgd/1"
            }
            .into(),
            exchange_windows: self.exchange_windows,
            corpus: self.corpus.clone(),
            remix: self.remix.clone(),
            config: self.config.clone(),
            step: self.step,
            learning_rate: self.learning_rate,
            weights_sha256: format!("{:x}", Sha256::digest(&bytes)),
            tokenizer: self
                .remix
                .as_ref()
                .map(|s| s.tokenizer.sha256.clone())
                .unwrap_or_else(|| self.checkpoint_identity.clone().unwrap_or_default()),
            optimizer: self.optimizer_id().into(),
            adamw,
            runtime: Some(self.plan.clone()),
        };
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let temporary = tempfile::tempdir_in(parent).map_err(|e| e.to_string())?;
        let mut files = vec![
            ("model.safetensors", bytes),
            (
                "manifest.json",
                serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?,
            ),
        ];
        if let Some(bytes) = optimizer_bytes {
            files.push(("optimizer.safetensors", bytes));
        }
        for (name, data) in files {
            let file = temporary.path().join(name);
            std::fs::write(&file, data).map_err(|e| e.to_string())?;
            std::fs::File::open(file)
                .and_then(|f| f.sync_all())
                .map_err(|e| e.to_string())?;
        }
        std::fs::File::open(temporary.path())
            .and_then(|f| f.sync_all())
            .map_err(|e| e.to_string())?;
        std::fs::create_dir(path).map_err(|e| e.to_string())?;
        if let Err(e) = std::fs::rename(temporary.path(), path) {
            let _ = std::fs::remove_dir(path);
            return Err(e.to_string());
        }
        std::fs::File::open(parent)
            .and_then(|f| f.sync_all())
            .map_err(|e| e.to_string())?;
        Ok(())
    }
    /// Restore weights, SGD configuration and step; validate manifest and content checksum.
    pub fn restore(path: &Path, training: bool) -> Result<Self, String> {
        Self::restore_with_budget(path, training, 2 * 1024 * 1024)
    }
    /// Restore on CPU within an explicit per-file decoded budget; relocate separately if needed.
    pub fn restore_with_budget(
        path: &Path,
        training: bool,
        max_bytes: usize,
    ) -> Result<Self, String> {
        let read = |name: &str, limit: u64| -> Result<Vec<u8>, String> {
            use std::io::Read;
            let f = std::fs::File::open(path.join(name)).map_err(|e| e.to_string())?;
            let mut bytes = Vec::new();
            f.take(
                limit
                    .checked_add(1)
                    .ok_or("checkpoint byte limit too large")?,
            )
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
            if bytes.len() as u64 > limit {
                return Err("checkpoint file exceeds limit".into());
            }
            Ok(bytes)
        };
        let m: Checkpoint =
            serde_json::from_slice(&read("manifest.json", 65536)?).map_err(|e| e.to_string())?;
        let supported = matches!(
            (
                m.format.as_str(),
                m.exchange_windows,
                m.corpus.is_some(),
                m.remix.is_some(),
                m.adamw.is_some()
            ),
            ("tiny-qwen-sgd/1", None, false, false, false)
                | ("tiny-qwen-echo-sgd/1", Some(2), false, false, false)
                | ("tiny-qwen-echo-corpus-sgd/1", Some(2), true, false, false)
                | ("tiny-qwen-echo-corpus-adamw/1", Some(2), true, false, true)
                | ("tiny-qwen-remix-sgd/1", Some(_), false, true, false)
                | ("tiny-qwen-remix-adamw/1", Some(_), false, true, true)
        );
        if !supported
            || m.tokenizer
                != m.remix
                    .as_ref()
                    .map(|s| s.tokenizer.sha256.as_str())
                    .unwrap_or("ascii/1")
            || m.optimizer
                != if m.adamw.is_some() {
                    "adamw-f32/1"
                } else {
                    "sgd-no-momentum/1"
                }
            || !m.learning_rate.is_finite()
            || m.learning_rate <= 0.0
        {
            return Err("incompatible checkpoint manifest".into());
        }
        if m.remix.is_none()
            && m.exchange_windows.is_some()
            && (m.config.vocab_size < 123 || m.config.max_position_embeddings < echo::LETTERS)
        {
            return Err("incompatible echo vocabulary/context".into());
        }
        if let Some(state) = &m.remix {
            state.validate(&m.config, m.exchange_windows, m.step)?;
        }
        let bytes = read("model.safetensors", max_bytes as u64)?;
        let _ = inspect(&mut Cursor::new(&bytes)).map_err(|e| e.to_string())?;
        if format!("{:x}", Sha256::digest(&bytes)) != m.weights_sha256 {
            return Err("checkpoint checksum mismatch".into());
        }
        let expected = echo::shapes(&m.config, m.exchange_windows)?;
        let sources = expected.keys().map(|n| (n.clone(), n.clone())).collect();
        let weights = import(
            &mut Cursor::new(&bytes),
            &Mapping {
                expected,
                sources,
                initialize: Default::default(),
            },
            &Tensors::new(),
            max_bytes,
        )
        .map_err(|e| e.to_string())?;
        let cpu = runtime::Plan::cpu(m.config.num_hidden_layers, max_bytes as u64);
        let mut model = Self::from_tensors(m.config, weights, m.exchange_windows, training, cpu)?;
        if let Some(state) = &m.corpus {
            state.dataset.validate()?;
            if state.completed_samples > m.step {
                return Err("echo sampler cursor exceeds completed model updates".into());
            }
        }
        if let Some(a) = m.adamw {
            model.adam = Some(optimizer::Adam::restore(
                a,
                read("optimizer.safetensors", max_bytes as u64)?,
                echo::shapes(&model.config, model.exchange_windows)?,
                &model.device,
                m.step,
                m.learning_rate,
                max_bytes,
            )?);
        }
        model.checkpoint_identity = if m.remix.is_none() {
            Some(m.tokenizer)
        } else {
            None
        };
        model.corpus = m.corpus;
        model.remix = m.remix;
        model.step = m.step;
        model.learning_rate = m.learning_rate;
        Ok(model)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    format: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exchange_windows: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    corpus: Option<corpus::TrainingState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    remix: Option<remix::TrainingState>,
    config: Config,
    step: u64,
    learning_rate: f32,
    weights_sha256: String,
    tokenizer: String,
    optimizer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    adamw: Option<optimizer::Checkpoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime: Option<runtime::Plan>,
}
