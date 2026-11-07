//! FP32 Qwen3 smoke runtime, plus a versioned two-window exchange experiment.
/// Procedural multi-pattern echo training and held-out evaluation.
pub mod corpus;
pub mod echo;
/// Explicit experimental optimizer configuration.
pub mod optimizer;
use burn_tensor::{
    Bool, DType, Device, Tensor, TensorData,
    activation::{log_softmax, softmax},
};
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
    config: Config,
    parameters: BTreeMap<String, T>,
    device: Device,
    training: bool,
    exchange_windows: Option<usize>,
    corpus: Option<corpus::TrainingState>,
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
        if exchange_windows.is_some()
            && (config.vocab_size < 123 || config.max_position_embeddings < echo::LETTERS)
        {
            return Err(
                "echo requires lowercase ASCII vocabulary and at least 12 local positions".into(),
            );
        }
        if config.hidden_size > 64
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
            config,
            parameters,
            device,
            training,
            exchange_windows,
            corpus: None,
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
    fn constant(&self, v: Vec<f32>, shape: [usize; 2]) -> T {
        T::from_data(TensorData::new(v, shape), (&self.device, DType::F32))
    }
    fn linear(&self, x: T, name: &str) -> T {
        x.matmul(self.parameters[name].clone().transpose())
    }
    fn norm(&self, x: T, name: &str) -> T {
        let eps = self.config.rms_norm_eps;
        let y = x.clone() / (x.powf_scalar(2.0).mean_dim(1) + eps).sqrt();
        y * self.parameters[name].clone()
    }
    fn rope(&self, x: T) -> T {
        let positions = (0..x.dims()[0]).collect::<Vec<_>>();
        self.rope_positions(x, &positions)
    }
    fn rope_positions(&self, x: T, positions: &[usize]) -> T {
        let [seq, d] = x.dims();
        let half = d / 2;
        T::cat(
            (0..seq)
                .map(|p| {
                    let mut rotation = vec![0.0; d * d];
                    for i in 0..half {
                        let (s, c) = (positions[p] as f64
                            / self.config.rope_theta.powf((2 * i) as f64 / d as f64))
                        .sin_cos();
                        rotation[i * d + i] = c as f32;
                        rotation[i * d + i + half] = -s as f32;
                        rotation[(i + half) * d + i] = s as f32;
                        rotation[(i + half) * d + i + half] = c as f32;
                    }
                    x.clone()
                        .slice([p..p + 1, 0..d])
                        .matmul(self.constant(rotation, [d, d]).transpose())
                })
                .collect(),
            0,
        )
    }
    /// Full causal Qwen forward with trainable RMS scales, split-half RoPE and GQA.
    pub fn forward(&self, ids: &[u32]) -> Result<T, String> {
        if self.is_echo() {
            return Err("Use echo_logits for the two-window architecture".into());
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
                let scores = (q.matmul(k.transpose()) / (dh as f32).sqrt())
                    .mask_fill(mask.clone(), f32::NEG_INFINITY);
                responses.push(softmax(scores, 1).matmul(v));
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
            x = x + self.linear(
                up * gate.clone() / (1.0 + (-gate).exp()),
                &(prefix + "mlp.down_proj.weight"),
            );
        }
        let x = self.norm(x, "model.norm.weight");
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
            return Err("Use train_echo_step for the two-window model".into());
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
        let next_step = self.step.checked_add(1).ok_or("training step overflow")?;
        let value = loss
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| e.to_string())?[0];
        if !value.is_finite() {
            return Err("nonfinite loss".into());
        }
        let gradients = loss.backward();
        let mut norm = 0.0f64;
        let mut gs = BTreeMap::new();
        for (name, p) in &self.parameters {
            let g = p
                .grad(&gradients)
                .ok_or_else(|| format!("missing gradient {name}"))?;
            let flat = g
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|e| e.to_string())?;
            for v in flat {
                if !v.is_finite() {
                    return Err(format!("nonfinite gradient {name}"));
                }
                norm += f64::from(v).powi(2);
            }
            gs.insert(name.clone(), g);
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
            if updated
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|e| e.to_string())?
                .iter()
                .any(|x| !x.is_finite())
            {
                return Err("nonfinite updated parameters".into());
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
            // Generation uses a detached CPU inference snapshot, never a backward tape.
            let device = Device::flex();
            let parameters = self
                .parameters
                .iter()
                .map(|(name, p)| {
                    (
                        name.clone(),
                        T::from_data(p.clone().into_data(), (&device, DType::F32)),
                    )
                })
                .collect();
            let inference = Self {
                config: self.config.clone(),
                parameters,
                device,
                training: false,
                exchange_windows: self.exchange_windows,
                corpus: self.corpus.clone(),
                adam: None,
                step: self.step,
                learning_rate: self.learning_rate,
            };
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
            format: if self.corpus.is_some() && self.adam.is_some() {
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
            config: self.config.clone(),
            step: self.step,
            learning_rate: self.learning_rate,
            weights_sha256: format!("{:x}", Sha256::digest(&bytes)),
            tokenizer: "ascii/1".into(),
            optimizer: self.optimizer_id().into(),
            adamw,
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
        let read = |name: &str, limit: u64| -> Result<Vec<u8>, String> {
            use std::io::Read;
            let f = std::fs::File::open(path.join(name)).map_err(|e| e.to_string())?;
            let mut bytes = Vec::new();
            f.take(limit + 1)
                .read_to_end(&mut bytes)
                .map_err(|e| e.to_string())?;
            if bytes.len() as u64 > limit {
                return Err("checkpoint file exceeds limit".into());
            }
            Ok(bytes)
        };
        let m: Checkpoint =
            serde_json::from_slice(&read("manifest.json", 65536)?).map_err(|e| e.to_string())?;
        if !matches!(
            (
                m.format.as_str(),
                m.exchange_windows,
                m.corpus.is_some(),
                m.adamw.is_some()
            ),
            ("tiny-qwen-sgd/1", None, false, false)
                | ("tiny-qwen-echo-sgd/1", Some(2), false, false)
                | ("tiny-qwen-echo-corpus-sgd/1", Some(2), true, false)
                | ("tiny-qwen-echo-corpus-adamw/1", Some(2), true, true)
        ) || m.tokenizer != "ascii/1"
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
        let bytes = read("model.safetensors", 2 * 1024 * 1024)?;
        let _ = inspect(&mut Cursor::new(&bytes)).map_err(|e| e.to_string())?;
        if format!("{:x}", Sha256::digest(&bytes)) != m.weights_sha256 {
            return Err("checkpoint checksum mismatch".into());
        }
        let mut model = Self::load_architecture(m.config, &bytes, training, m.exchange_windows)?;
        if let Some(state) = &m.corpus {
            state.dataset.validate()?;
            if state.completed_samples > m.step {
                return Err("echo sampler cursor exceeds completed model updates".into());
            }
        }
        if let Some(a) = m.adamw {
            model.adam = Some(optimizer::Adam::restore(
                a,
                read("optimizer.safetensors", 2 * 1024 * 1024)?,
                echo::shapes(&model.config, model.exchange_windows)?,
                &model.device,
                m.step,
                m.learning_rate,
            )?);
        }
        model.corpus = m.corpus;
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
    config: Config,
    step: u64,
    learning_rate: f32,
    weights_sha256: String,
    tokenizer: String,
    optimizer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    adamw: Option<optimizer::Checkpoint>,
}
