//! Differentiable f32 CPU execution with an explicit f64 validation path of kidra-reference/2 using Burn/Flex.
//! Independent of HTTP and trainers. Caches retain the attached graph across ticks.
use burn_tensor::{
    DType, Device, Tensor, TensorData,
    activation::{log_softmax, softmax},
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT_MODEL: AtomicUsize = AtomicUsize::new(1);
use tesslot_core::{ModelConfig, Phase, Row, Window, visible_to};
use tesslot_model::ReferenceModel;
/// Backend-independent named parameter payload (output-major matrices).
pub type Parameters = BTreeMap<String, Vec<Vec<f64>>>;
/// Attached scalar objective, available to trainers without a direct backend dependency.
pub type Loss = Tensor<1>;
type T = Tensor<2>;
struct Entry {
    row: Row,
    states: Vec<T>,
}
/// Attached numerical memory; create a fresh cache for each trajectory/model.
#[derive(Default)]
pub struct Cache {
    entries: Vec<Entry>,
    last: Option<u64>,
    owner: Option<usize>,
}
/// A differentiable query and its unshifted source identity.
pub struct Prediction {
    /// Sample row index.
    pub row_index: usize,
    /// Logits of shape [1,vocabulary].
    pub logits: T,
}
/// All reference parameters are differentiable leaves, including original embeddings.
pub struct Model {
    config: ModelConfig,
    id: usize,
    device: Device,
    parameters: BTreeMap<String, T>,
    dtype: DType,
    training: bool,
}
impl Model {
    /// Load validated reference parameters; choose operation-level recomputation explicitly.
    pub fn new(config: ModelConfig, values: &Parameters, checkpoint: bool) -> Result<Self, String> {
        Self::with_dtype(config, values, checkpoint, DType::F32, true)
    }
    /// High precision diagnostic path for oracle/finite-difference verification only.
    pub fn validation_f64(
        config: ModelConfig,
        values: &Parameters,
        checkpoint: bool,
    ) -> Result<Self, String> {
        Self::with_dtype(config, values, checkpoint, DType::F64, true)
    }
    /// FP32 inference without recording a backward graph or retaining parameter gradients.
    pub fn inference(config: ModelConfig, values: &Parameters) -> Result<Self, String> {
        Self::with_dtype(config, values, false, DType::F32, false)
    }
    fn with_dtype(
        config: ModelConfig,
        values: &Parameters,
        checkpoint: bool,
        dtype: DType,
        training: bool,
    ) -> Result<Self, String> {
        if dtype == DType::F32
            && values
                .values()
                .flatten()
                .flatten()
                .any(|v| !(*v as f32).is_finite())
        {
            return Err("parameter cannot be represented as finite f32".into());
        }

        ReferenceModel::from_parameters(config.clone(), values).map_err(|e| e.to_string())?;
        let device = if training {
            Device::flex().autodiff()
        } else {
            Device::flex()
        };
        let device = if checkpoint {
            device.gradient_checkpointing()
        } else {
            device
        };
        let parameters = values
            .iter()
            .map(|(n, v)| {
                let data = TensorData::new(
                    v.iter().flatten().copied().collect::<Vec<_>>(),
                    [v.len(), v[0].len()],
                );
                (n.clone(), {
                    let tensor = T::from_data(data, (&device, dtype));
                    if training {
                        tensor.require_grad()
                    } else {
                        tensor
                    }
                })
            })
            .collect();
        Ok(Self {
            config,
            id: NEXT_MODEL.fetch_add(1, Ordering::Relaxed),
            device,
            parameters,
            dtype,
            training,
        })
    }
    /// Actual storage/compute dtype for parameters and activations.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    fn constant(&self, v: Vec<f64>, shape: [usize; 2]) -> T {
        T::from_data(TensorData::new(v, shape), (&self.device, self.dtype))
    }
    fn linear(&self, x: T, name: &str) -> T {
        x.matmul(self.parameters[name].clone().transpose())
    }
    fn embed(&self, r: &Row, windows: &[Window], query: bool) -> T {
        let d = self.config.hidden_size;
        let x = if query {
            self.parameters["query"].clone()
        } else {
            self.parameters["tokens"]
                .clone()
                .slice([r.token_id as usize..r.token_id as usize + 1, 0..d])
        };
        if self.config.window_embeddings {
            let slot = windows
                .iter()
                .find(|w| w.id == r.window)
                .expect("validated window")
                .slot;
            x + self.parameters["windows"]
                .clone()
                .slice([slot..slot + 1, 0..d])
        } else {
            x
        }
    }
    fn rope(&self, x: T, p: usize) -> T {
        let d = self.config.hidden_size;
        let mut rotation = vec![0.0; d * d];
        for i in (0..d).step_by(2) {
            let (s, c) = (p as f64 / self.config.rope_theta.powf(i as f64 / d as f64)).sin_cos();
            rotation[i * d + i] = c;
            rotation[i * d + i + 1] = -s;
            rotation[(i + 1) * d + i] = s;
            rotation[(i + 1) * d + i + 1] = c;
        }
        x.matmul(self.constant(rotation, [d, d]).transpose())
    }
    fn attention(&self, q: T, keys: Vec<T>, values: Vec<T>) -> Option<T> {
        if keys.is_empty() {
            return None;
        }
        let k = T::cat(keys, 0);
        let v = T::cat(values, 0);
        Some(
            softmax(
                q.matmul(k.transpose()) / (self.config.hidden_size as f64).sqrt(),
                1,
            )
            .matmul(v),
        )
    }
    fn local(&self, x: T, r: &Row, windows: &[Window], memory: &[Entry], l: usize) -> T {
        let prefix = format!("layers.{l}.");
        let q = self.rope(
            self.linear(rms(x.clone()), &(prefix.clone() + "q")),
            r.position,
        );
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for e in memory
            .iter()
            .filter(|e| e.row.window == r.window && visible_to(r, &e.row, windows))
        {
            let h = rms(e.states[l].clone());
            keys.push(self.rope(
                self.linear(h.clone(), &(prefix.clone() + "k")),
                e.row.position,
            ));
            values.push(self.linear(h, &(prefix.clone() + "v")));
        }
        let u = match self.attention(q, keys, values) {
            Some(a) => x + self.linear(a, &(prefix.clone() + "o")),
            None => x,
        };
        let h = rms(u.clone());
        let gate = self.linear(h.clone(), &(prefix.clone() + "gate"));
        let up = self.linear(h, &(prefix.clone() + "up"));
        u + self.linear(
            up * gate.clone() / (1.0 + (-gate).exp()),
            &(prefix + "down"),
        )
    }
    fn global(&self, x: T, r: &Row, windows: &[Window], memory: &[Entry], l: usize) -> T {
        if !self.config.global_attention {
            return x;
        }
        let centered = x.clone() - x.clone().mean_dim(1);
        let normalized = centered.clone() / (centered.powf_scalar(2.0).mean_dim(1) + 1e-5).sqrt();
        let q = self.linear(normalized, "exchange.query");
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for e in memory.iter().filter(|e| visible_to(r, &e.row, windows)) {
            let h = rms(e.states[l].clone());
            let mut k = self.rope(
                self.linear(h.clone(), &format!("layers.{l}.k")),
                e.row.position,
            );
            if self.config.window_embeddings {
                let slot = windows
                    .iter()
                    .find(|w| w.id == e.row.window)
                    .expect("validated window")
                    .slot;
                k = k + self.parameters["windows"]
                    .clone()
                    .slice([slot..slot + 1, 0..self.config.hidden_size]);
            }
            keys.push(k);
            values.push(self.linear(h, &format!("layers.{l}.v")));
        }
        match self.attention(q, keys, values) {
            Some(a) => x + self.linear(a, "exchange.out"),
            None => x,
        }
    }
    fn entry(&self, r: &Row, windows: &[Window], memory: &[Entry]) -> Entry {
        let mut states = vec![self.embed(r, windows, false)];
        for l in 0..self.config.layers {
            let x = self.local(states[l].clone(), r, windows, memory, l);
            states.push(self.global(x, r, windows, memory, l));
        }
        Entry {
            row: r.clone(),
            states,
        }
    }
    fn query(&self, r: &Row, windows: &[Window], memory: &[Entry]) -> Prediction {
        let mut x = self.embed(r, windows, true);
        for l in 0..self.config.layers {
            x = self.local(x, r, windows, memory, l);
            x = self.global(x, r, windows, memory, l);
        }
        Prediction {
            row_index: r.index,
            logits: self.linear(rms(x), "output"),
        }
    }
    /// Execute a validated full tick without detaching any retained activations.
    pub fn tick(
        &self,
        rows: &[Row],
        windows: &[Window],
        cache: &mut Cache,
    ) -> Result<Vec<Prediction>, String> {
        let tick = rows.first().ok_or("empty tick")?.tick;
        if cache.owner.is_some_and(|id| id != self.id) {
            return Err("cache belongs to a different model; create a fresh cache".into());
        }
        if cache.last.is_some_and(|t| tick <= t)
            || rows.iter().any(|r| {
                r.tick != tick
                    || r.token_id as usize >= self.config.vocab_size
                    || !windows
                        .iter()
                        .any(|w| w.id == r.window && w.slot < self.config.max_windows)
            })
            || rows.windows(2).any(|p| p[0].phase > p[1].phase)
        {
            return Err("invalid tick, vocabulary or window mapping".into());
        }
        cache.owner = Some(self.id);
        let mut result = Vec::new();
        for r in rows.iter().filter(|r| r.phase != Phase::D) {
            if r.phase == Phase::B {
                result.push(self.query(r, windows, &cache.entries));
            }
            if r.durable {
                cache.entries.push(self.entry(r, windows, &cache.entries));
            }
        }
        result.extend(
            rows.iter()
                .filter(|r| r.phase == Phase::D)
                .map(|r| self.query(r, windows, &cache.entries)),
        );
        let pending: Vec<_> = rows
            .iter()
            .filter(|r| r.phase == Phase::D && r.durable)
            .map(|r| self.entry(r, windows, &cache.entries))
            .collect();
        cache.entries.extend(pending);
        cache.entries.retain(|e| {
            !e.row.removed_at.as_ref().is_some_and(|at| at.tick <= tick)
                && windows
                    .iter()
                    .any(|w| w.id == e.row.window && !w.close.is_some_and(|t| t <= tick))
        });
        cache.last = Some(tick);
        Ok(result)
    }
    /// Replay complete ticks; storage boundaries never trigger backward or detach.
    pub fn forward(&self, rows: &[Row], windows: &[Window]) -> Result<Vec<Prediction>, String> {
        let mut cache = Cache::default();
        let mut outputs = Vec::new();
        let mut start = 0;
        while start < rows.len() {
            let end = start + rows[start..].partition_point(|r| r.tick == rows[start].tick);
            outputs.extend(self.tick(&rows[start..end], windows, &mut cache)?);
            start = end;
        }
        Ok(outputs)
    }
    /// Stable weighted objective; denominator is the effective weight clamped to one.
    pub fn loss(&self, predictions: &[Prediction], rows: &[Row]) -> Result<Tensor<1>, String> {
        let (total, weight) = self.loss_parts(predictions, rows)?;
        Ok(total / weight.max(1.0))
    }
    /// Unnormalized attached objective and effective weight, for sample accumulation.
    /// Normalize once across the optimizer batch, never at storage-chunk boundaries.
    pub fn loss_parts(
        &self,
        predictions: &[Prediction],
        rows: &[Row],
    ) -> Result<(Tensor<1>, f64), String> {
        let mut total = self.parameters["output"].clone().sum() * 0.0;
        let mut weight = 0.0;
        for p in predictions {
            let r = rows
                .get(p.row_index)
                .ok_or("prediction row outside sample")?;
            if !r.loss_weight.is_finite()
                || r.loss_weight < 0.0
                || (self.dtype == DType::F32 && !(r.loss_weight as f32).is_finite())
            {
                return Err("invalid effective weight".into());
            }
            if r.loss_weight > 0.0 {
                let logp = log_softmax(p.logits.clone(), 1);
                let id = r.token_id as usize;
                if id >= self.config.vocab_size {
                    return Err("target outside vocabulary".into());
                }
                total = total - logp.slice([0..1, id..id + 1]).sum() * r.loss_weight;
                weight += r.loss_weight;
            }
        }
        if !weight.is_finite() || (self.dtype == DType::F32 && !(weight as f32).is_finite()) {
            return Err("effective weight overflow".into());
        }
        Ok((total, weight))
    }
    /// Backpropagate once and return all named gradients; unused leaves get exact zeros.
    pub fn gradients(&self, loss: &Tensor<1>) -> Result<Parameters, String> {
        if !self.training {
            return Err("inference model does not record gradients".into());
        }
        let grads = loss.backward();
        self.parameters
            .iter()
            .map(|(n, p)| {
                let [a, b] = p.dims();
                let flat = match p.grad(&grads) {
                    Some(g) => g
                        .into_data_as::<f64>()
                        .try_to_vec::<f64>()
                        .map_err(|e| e.to_string())?,
                    None => vec![0.0; a * b],
                };
                if flat.iter().any(|v| !v.is_finite()) {
                    return Err(format!("nonfinite gradient: {n}"));
                }
                Ok((n.clone(), flat.chunks(b).map(|r| r.to_vec()).collect()))
            })
            .collect()
    }
}
fn rms(x: T) -> T {
    x.clone() / (x.powf_scalar(2.0).mean_dim(1) + 1e-6).sqrt()
}
