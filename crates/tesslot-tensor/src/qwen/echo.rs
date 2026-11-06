//! Two-window echo protocol and full-rank GQA specialization of ProjectKidra exchange.
//! Source letters are runtime RNG decisions, not model predictions or supervised targets.
use super::*;

/// Fixed length for this bounded experiment, not the model's window capacity.
pub const LETTERS: usize = 12;
/// Explicit architecture/protocol identity stored with experiment results.
pub const PROTOCOL: &str = "kidra-qwen-echo/1";

/// Deterministic random lowercase ASCII letters; no pretrained model or external service.
pub fn pattern(seed: u64) -> Vec<u32> {
    let mut state = seed;
    (0..LETTERS)
        .map(|_| {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            u32::from(b'a') + ((z ^ (z >> 31)) % 26) as u32
        })
        .collect()
}

pub(super) fn shapes(
    config: &Config,
    windows: Option<usize>,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    let mut shapes = tesslot_artifacts::qwen::source_shapes(config).map_err(|e| e.to_string())?;
    if let Some(windows) = windows {
        if windows != 2 {
            return Err("echo experiment requires exactly two windows".into());
        }
        let d = config.hidden_size;
        let q = config.query_width()?;
        let kv = config.kv_width()?;
        for (name, shape) in [
            ("kidra.window_embeddings.weight", vec![windows, d]),
            ("kidra.query_embedding.weight", vec![1, d]),
            ("kidra.exchange.identities.weight", vec![windows, kv]),
            ("kidra.exchange.query.weight", vec![q, d]),
            ("kidra.exchange.out.weight", vec![d, q]),
            ("kidra.exchange.norm.weight", vec![d]),
            ("kidra.exchange.norm.bias", vec![d]),
        ] {
            shapes.insert(name.into(), shape);
        }
    }
    Ok(shapes)
}

#[derive(Clone)]
struct Row {
    window: usize,
    position: usize,
    tick: usize,
    token: Option<u32>,
}

/// Greedy replay of actual model outputs. Source and echo commit only after each decision.
#[derive(Debug, Serialize, PartialEq)]
pub struct Replay {
    /// Architecture identity independent of application version.
    pub protocol: &'static str,
    /// Automatically generated source letters (window 0).
    pub source: String,
    /// Greedy echo, displayed as ASCII or an escaped nonprinting token (window 1).
    pub echo: String,
    /// Authoritative model-generated token IDs.
    pub echo_token_ids: Vec<u32>,
    /// Fraction of correct freely generated letters, not teacher-forced accuracy.
    pub accuracy: f32,
    /// Logical timeline; current-tick source output is never visible to the echo decision.
    pub ticks: Vec<Tick>,
}
/// Parallel decision phase, committed together.
#[derive(Debug, Serialize, PartialEq)]
pub struct Tick {
    /// Logical model tick.
    pub tick: usize,
    /// Runtime-random source decision, absent at the drain tick.
    pub source_token: Option<u32>,
    /// Echo model prediction, absent at the initial tick.
    pub echo_token: Option<u32>,
}
fn text(ids: &[u32]) -> String {
    ids.iter()
        .map(|t| {
            if (32..127).contains(t) {
                char::from_u32(*t).unwrap().to_string()
            } else {
                format!("\\x{t:02x}")
            }
        })
        .collect()
}
impl Model {
    /// Convert the tiny base weights and seed the seven trainable Kidra components.
    pub fn echo_demo(training: bool) -> Result<Self, String> {
        let c = Config::from_json(include_bytes!("../../../../fixtures/tiny-qwen/config.json"))?;
        Self::echo_from_base(
            c,
            include_bytes!("../../../../fixtures/tiny-qwen/model.safetensors"),
            20261107,
            training,
        )
    }
    /// Convert tiny local Qwen weights into this explicitly versioned two-window architecture.
    pub fn echo_from_base(
        c: Config,
        bytes: &[u8],
        seed: u64,
        training: bool,
    ) -> Result<Self, String> {
        let converted =
            tesslot_artifacts::qwen::convert(&mut Cursor::new(bytes), &c, 2, seed, 128 * 1024)
                .map_err(|e| e.to_string())?;
        let bytes = encode_f32_with_metadata(&converted.tensors, BTreeMap::new())
            .map_err(|e| e.to_string())?;
        Self::load_architecture(c, &bytes, training, Some(2))
    }
    /// Whether checkpoint/model belongs to the separate two-window experiment.
    pub fn is_echo(&self) -> bool {
        self.exchange_windows == Some(2)
    }

    fn mask(
        &self,
        rows: &[Row],
        memory: usize,
        local: bool,
        source_visible: bool,
    ) -> Tensor<2, Bool> {
        let data = rows
            .iter()
            .flat_map(|r| {
                rows[..memory].iter().map(move |k| {
                    k.tick >= r.tick
                        || (local && k.window != r.window)
                        || (!source_visible && r.window == 1 && k.window == 0)
                })
            })
            .collect::<Vec<_>>();
        Tensor::from_data(TensorData::new(data, [rows.len(), memory]), &self.device)
    }
    fn attend(&self, q: T, k: T, v: T, mask: Tensor<2, Bool>) -> T {
        // Empty visible sets yield zero exactly, without NaN softmax(-inf,...,-inf).
        let weights = softmax(
            (q.matmul(k.transpose()) / (self.config.head_dim as f32).sqrt())
                .mask_fill(mask.clone(), f32::NEG_INFINITY)
                .mask_fill(mask.clone().all_dim(1), 0.0),
            1,
        )
        .mask_fill(mask, 0.0);
        weights.matmul(v)
    }
    fn forward_rows(&self, rows: &[Row], memory: usize, source_visible: bool) -> Result<T, String> {
        if !self.is_echo() {
            return Err("two-window model required".into());
        }
        let c = &self.config;
        let d = c.hidden_size;
        let n = rows.len();
        let positions = rows.iter().map(|r| r.position).collect::<Vec<_>>();
        let mut x = T::cat(
            rows.iter()
                .map(|r| {
                    let input = if let Some(token) = r.token {
                        self.parameters["model.embed_tokens.weight"]
                            .clone()
                            .slice([token as usize..token as usize + 1, 0..d])
                    } else {
                        self.parameters["kidra.query_embedding.weight"].clone()
                    };
                    input
                        + self.parameters["kidra.window_embeddings.weight"]
                            .clone()
                            .slice([r.window..r.window + 1, 0..d])
                })
                .collect(),
            0,
        );
        let local_mask = self.mask(rows, memory, true, source_visible);
        let global_mask = self.mask(rows, memory, false, source_visible);
        for l in 0..c.num_hidden_layers {
            let prefix = format!("model.layers.{l}.");
            let h = self.norm(x.clone(), &(prefix.clone() + "input_layernorm.weight"));
            let query = self.linear(h.clone(), &(prefix.clone() + "self_attn.q_proj.weight"));
            let keys = self.linear(h.clone(), &(prefix.clone() + "self_attn.k_proj.weight"));
            let values = self.linear(h, &(prefix.clone() + "self_attn.v_proj.weight"));
            let mut local = Vec::new();
            let mut kv = Vec::new();
            for head in 0..c.num_key_value_heads {
                let start = head * c.head_dim;
                let key = self
                    .rope_positions(
                        self.norm(
                            keys.clone().slice([0..n, start..start + c.head_dim]),
                            &(prefix.clone() + "self_attn.k_norm.weight"),
                        ),
                        &positions,
                    )
                    .slice([0..memory, 0..c.head_dim]);
                let value = values.clone().slice([0..memory, start..start + c.head_dim]);
                kv.push((key, value));
            }
            for head in 0..c.num_attention_heads {
                let start = head * c.head_dim;
                let q = self.rope_positions(
                    self.norm(
                        query.clone().slice([0..n, start..start + c.head_dim]),
                        &(prefix.clone() + "self_attn.q_norm.weight"),
                    ),
                    &positions,
                );
                let (k, v) = &kv[head / (c.num_attention_heads / c.num_key_value_heads)];
                local.push(self.attend(q, k.clone(), v.clone(), local_mask.clone()));
            }
            x = x + self.linear(
                T::cat(local, 1),
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
            // Original GlobalExchange: LayerNorm query, shared Q/out, cached local K/V.
            // No global Q RoPE, extra K/V projections, or value identity embeddings.
            let centered = x.clone() - x.clone().mean_dim(1);
            let h = centered.clone() / (centered.powf_scalar(2.0).mean_dim(1) + 1e-5).sqrt();
            let h = h * self.parameters["kidra.exchange.norm.weight"].clone()
                + self.parameters["kidra.exchange.norm.bias"].clone();
            let q = self.linear(h, "kidra.exchange.query.weight");
            let mut global = Vec::new();
            for head in 0..c.num_attention_heads {
                let kh = head / (c.num_attention_heads / c.num_key_value_heads);
                let (k, v) = &kv[kh];
                let identities = T::cat(
                    rows[..memory]
                        .iter()
                        .map(|r| {
                            self.parameters["kidra.exchange.identities.weight"]
                                .clone()
                                .slice([
                                    r.window..r.window + 1,
                                    kh * c.head_dim..(kh + 1) * c.head_dim,
                                ])
                        })
                        .collect(),
                    0,
                );
                global.push(
                    self.attend(
                        q.clone()
                            .slice([0..n, head * c.head_dim..(head + 1) * c.head_dim]),
                        k.clone() + identities,
                        v.clone(),
                        global_mask.clone(),
                    ),
                );
            }
            x = x + self.linear(T::cat(global, 1), "kidra.exchange.out.weight");
        }
        Ok(self.linear(
            self.norm(x, "model.norm.weight"),
            if c.tie_word_embeddings {
                "model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            },
        ))
    }
    fn echo_rows(
        &self,
        source: &[u32],
        history: &[u32],
        predictions: usize,
    ) -> Result<(Vec<Row>, usize), String> {
        if source.len() != LETTERS
            || source.iter().any(|t| !(97..=122).contains(t))
            || history.len() > LETTERS
            || predictions == 0
            || predictions > LETTERS
            || history.len() < predictions - 1
        {
            return Err("echo requires 12 lowercase letters and valid causal echo history".into());
        }
        if history
            .iter()
            .any(|t| *t as usize >= self.config.vocab_size)
        {
            return Err("invalid echo token".into());
        }
        let mut rows = source
            .iter()
            .enumerate()
            .map(|(p, t)| Row {
                window: 0,
                position: p,
                tick: p,
                token: Some(*t),
            })
            .collect::<Vec<_>>();
        rows.extend(history.iter().enumerate().map(|(p, t)| Row {
            window: 1,
            position: p,
            tick: p + 1,
            token: Some(*t),
        }));
        let memory = rows.len();
        rows.extend((0..predictions).map(|p| Row {
            window: 1,
            position: p,
            tick: p + 1,
            token: None,
        }));
        Ok((rows, memory))
    }
    /// Label-free echo queries. Targets are used only as earlier durable teacher-forced history.
    /// Source ablation blocks source keys at every layer, preventing indirect contamination.
    pub fn echo_logits(
        &self,
        source: &[u32],
        history: &[u32],
        predictions: usize,
        source_visible: bool,
    ) -> Result<T, String> {
        let (rows, memory) = self.echo_rows(source, history, predictions)?;
        Ok(self
            .forward_rows(&rows, memory, source_visible)?
            .slice([memory..rows.len(), 0..self.config.vocab_size]))
    }
    fn echo_loss(&self, source: &[u32], source_visible: bool) -> Result<Tensor<1>, String> {
        let logits = self.echo_logits(
            source,
            &source[..source.len().saturating_sub(1)],
            LETTERS,
            source_visible,
        )?;
        let p = log_softmax(logits, 1);
        Ok(-Tensor::<1>::cat(
            source
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    p.clone()
                        .slice([i..i + 1, *t as usize..*t as usize + 1])
                        .sum()
                })
                .collect(),
            0,
        )
        .mean())
    }
    /// Echo-only average cross entropy (source window has no supervised loss).
    pub fn evaluate_echo(&self, source: &[u32], source_visible: bool) -> Result<f32, String> {
        Ok(self
            .echo_loss(source, source_visible)?
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| e.to_string())?[0])
    }
    /// One attached full-parameter update over all 12 echo targets and source memory.
    pub fn train_echo_step(&mut self, source: &[u32]) -> Result<Metrics, String> {
        self.update(self.echo_loss(source, true)?)
    }
    /// Free-running replay; no teacher-forced echo labels enter model context.
    pub fn replay_echo(&self, source: &[u32], source_visible: bool) -> Result<Replay, String> {
        let inference = self.inference_snapshot();
        let mut echo = Vec::new();
        for p in 0..LETTERS {
            let logits = inference
                .echo_logits(source, &echo, p + 1, source_visible)?
                .slice([p..p + 1, 0..self.config.vocab_size])
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|e| e.to_string())?;
            if logits.iter().any(|x| !x.is_finite()) {
                return Err("nonfinite echo logits".into());
            }
            echo.push(
                logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .ok_or("empty echo logits")?
                    .0 as u32,
            );
        }
        Ok(Replay {
            protocol: PROTOCOL,
            source: text(source),
            echo: text(&echo),
            accuracy: echo.iter().zip(source).filter(|(a, b)| a == b).count() as f32
                / LETTERS as f32,
            ticks: (0..=LETTERS)
                .map(|tick| Tick {
                    tick,
                    source_token: source.get(tick).copied(),
                    echo_token: tick.checked_sub(1).and_then(|p| echo.get(p)).copied(),
                })
                .collect(),
            echo_token_ids: echo,
        })
    }
    fn inference_snapshot(&self) -> Self {
        let device = Device::flex();
        Self {
            config: self.config.clone(),
            parameters: self
                .parameters
                .iter()
                .map(|(n, p)| {
                    (
                        n.clone(),
                        T::from_data(p.clone().into_data(), (&device, DType::F32)),
                    )
                })
                .collect(),
            device,
            training: false,
            exchange_windows: self.exchange_windows,
            step: self.step,
            learning_rate: self.learning_rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "opt-in source-to-echo attached gradient gate"]
    fn source_memory_has_attached_gradients() {
        let model = Model::echo_demo(true).unwrap();
        let source = pattern(42);
        let gradient = |visible| {
            let logits = model.echo_logits(&source, &[], 1, visible).unwrap();
            let loss = -log_softmax(logits, 1)
                .slice([0..1, source[0] as usize..source[0] as usize + 1])
                .sum();
            let gradients = loss.backward();
            let g = model.parameters["model.layers.0.self_attn.v_proj.weight"]
                .grad(&gradients)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            g.iter().map(|x| x.abs()).sum::<f32>()
        };
        assert!(gradient(true) > 1e-7);
        assert_eq!(gradient(false), 0.0);
    }
}
