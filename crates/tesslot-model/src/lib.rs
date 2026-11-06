//! Deterministic f64 CPU oracle, not a production Qwen3 implementation or trainer.
//!
//! Separate prediction queries never embed their labels. Durable memory is built
//! causally at every layer. The batched and tick-cache paths share parameters but
//! construct memory independently, enabling meaningful equivalence tests.

use tesslot_core::{ModelConfig, Phase, Row, Window, visible_to};

/// Hugging Face Qwen3 configuration inspection.
pub mod qwen;

type Vector = Vec<f64>;

#[derive(Clone)]
struct Matrix {
    rows: Vec<Vector>,
}
impl Matrix {
    fn new(output: usize, input: usize, rng: &mut Rng) -> Self {
        Self {
            rows: (0..output)
                .map(|_| {
                    (0..input)
                        .map(|_| rng.next() / (input as f64).sqrt())
                        .collect()
                })
                .collect(),
        }
    }
    fn apply(&self, x: &[f64]) -> Vector {
        self.rows.iter().map(|r| dot(r, x)).collect()
    }
}
#[derive(Clone)]
struct Attention {
    q: Matrix,
    k: Matrix,
    v: Matrix,
    o: Matrix,
}
impl Attention {
    fn new(d: usize, rng: &mut Rng) -> Self {
        Self {
            q: Matrix::new(d, d, rng),
            k: Matrix::new(d, d, rng),
            v: Matrix::new(d, d, rng),
            o: Matrix::new(d, d, rng),
        }
    }
    fn apply(
        &self,
        x: &[f64],
        row: &Row,
        keys: impl Iterator<Item = (Vector, usize)>,
        theta: f64,
    ) -> Vector {
        let q = rope(&self.q.apply(&norm(x)), row.position, theta);
        let mut scores = Vec::new();
        let mut values = Vec::new();
        for (key, position) in keys {
            let normalized = norm(&key);
            let k = rope(&self.k.apply(&normalized), position, theta);
            scores.push(dot(&q, &k) / (x.len() as f64).sqrt());
            values.push(self.v.apply(&normalized));
        }
        if scores.is_empty() {
            return x.to_vec();
        }
        let weights = softmax(&scores);
        let mut value = vec![0.0; x.len()];
        for (w, v) in weights.iter().zip(values) {
            for (a, b) in value.iter_mut().zip(v) {
                *a += w * b;
            }
        }
        add(x, &self.o.apply(&value))
    }
}
#[derive(Clone)]
struct Layer {
    local: Attention,
    up: Matrix,
    gate: Matrix,
    down: Matrix,
}
impl Layer {
    fn new(d: usize, rng: &mut Rng) -> Self {
        Self {
            local: Attention::new(d, rng),
            up: Matrix::new(2 * d, d, rng),
            gate: Matrix::new(2 * d, d, rng),
            down: Matrix::new(d, 2 * d, rng),
        }
    }
    fn feed_forward(&self, x: &[f64]) -> Vector {
        let normalized = norm(x);
        let hidden: Vector = self
            .up
            .apply(&normalized)
            .into_iter()
            .zip(self.gate.apply(&normalized))
            .map(|(u, g)| u * g / (1.0 + (-g).exp()))
            .collect();
        add(x, &self.down.apply(&hidden))
    }
}

// Single-head specialization of ProjectKidra model/exchange.py::GlobalExchange.
// Linear biases are initialized to zero; LayerNorm affine is identity in this oracle.
#[derive(Clone)]
struct GlobalExchange {
    query: Matrix,
    out: Matrix,
}
impl GlobalExchange {
    fn apply(&self, x: &[f64], keys_values: impl Iterator<Item = (Vector, Vector)>) -> Vector {
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
        let normalized: Vector = x
            .iter()
            .map(|v| (v - mean) / (variance + 1e-5).sqrt())
            .collect();
        let q = self.query.apply(&normalized);
        let mut scores = Vec::new();
        let mut values = Vec::new();
        for (k, v) in keys_values {
            scores.push(dot(&q, &k) / (x.len() as f64).sqrt());
            values.push(v);
        }
        if scores.is_empty() {
            return x.to_vec();
        }
        let mut response = vec![0.0; x.len()];
        for (weight, value) in softmax(&scores).into_iter().zip(values) {
            for (r, v) in response.iter_mut().zip(value) {
                *r += weight * v;
            }
        }
        add(x, &self.out.apply(&response))
    }
}

/// Tiny configurable multi-window transformer with local RoPE and global attention.
#[derive(Clone)]
pub struct ReferenceModel {
    config: ModelConfig,
    tokens: Matrix,
    windows: Matrix,
    query_embedding: Vector,
    layers: Vec<Layer>,
    output: Matrix,
    exchange: GlobalExchange,
}

/// Prediction before target commit, including excluded observations for inspection.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Source row index, retaining tick/window identity through the caller's plan.
    pub row_index: usize,
    /// Unnormalized next-token scores.
    pub logits: Vec<f64>,
}

#[derive(Clone)]
struct Entry {
    row: Row,
    states: Vec<Vector>,
    local: Vec<Vector>,
}

/// Incremental per-layer memory. Non-durable predictions never become entries.
#[derive(Default)]
pub struct Cache {
    entries: Vec<Entry>,
    last_tick: Option<u64>,
    model_config: Option<ModelConfig>,
}
impl Cache {
    /// Number of live durable tokens (including BOS/headers), not a VRAM estimate.
    pub fn durable_tokens(&self) -> usize {
        self.entries.len()
    }
    /// Window-local cache occupancy for diagnostics.
    pub fn window_tokens(&self, id: &str) -> usize {
        self.entries.iter().filter(|e| e.row.window == id).count()
    }
}

impl ReferenceModel {
    /// Seeded parameters use a fixed integer generator, independent of global RNG state.
    pub fn new(config: ModelConfig) -> Result<Self, Error> {
        config.validate().map_err(|e| Error::Input(e.to_string()))?;
        let mut rng = Rng(config.seed);
        let d = config.hidden_size;
        Ok(Self {
            tokens: Matrix::new(config.vocab_size, d, &mut rng),
            windows: Matrix::new(config.max_windows, d, &mut rng),
            query_embedding: (0..d).map(|_| rng.next()).collect(),
            layers: (0..config.layers)
                .map(|_| Layer::new(d, &mut rng))
                .collect(),
            exchange: GlobalExchange {
                query: Matrix::new(d, d, &mut rng),
                out: Matrix::new(d, d, &mut rng),
            },
            output: Matrix::new(config.vocab_size, d, &mut rng),
            config,
        })
    }

    /// Named f64 parameter matrices, independent of the tensor backend.
    pub fn parameters(&self) -> std::collections::BTreeMap<String, Vec<Vec<f64>>> {
        let mut copy = self.clone();
        let mut result = std::collections::BTreeMap::new();
        copy.visit_matrices(|name, m| {
            result.insert(name, m.rows.clone());
        });
        result.insert("query".into(), vec![self.query_embedding.clone()]);
        result
    }
    /// Load all parameters transactionally; unknown names/shapes/nonfinite values fail.
    pub fn from_parameters(
        config: ModelConfig,
        values: &std::collections::BTreeMap<String, Vec<Vec<f64>>>,
    ) -> Result<Self, Error> {
        let mut model = Self::new(config)?;
        let expected = model.parameters();
        if expected.keys().ne(values.keys())
            || expected.iter().any(|(n, v)| {
                values[n].len() != v.len()
                    || values[n]
                        .iter()
                        .zip(v)
                        .any(|(a, b)| a.len() != b.len() || a.iter().any(|x| !x.is_finite()))
            })
        {
            return Err(Error::Input(
                "parameter names, shapes or finite values incompatible with reference architecture"
                    .into(),
            ));
        }
        model.visit_matrices(|name, m| {
            m.rows = values[&name].clone();
        });
        model.query_embedding = values["query"][0].clone();
        Ok(model)
    }
    fn visit_matrices(&mut self, mut visit: impl FnMut(String, &mut Matrix)) {
        visit("tokens".into(), &mut self.tokens);
        visit("windows".into(), &mut self.windows);
        visit("output".into(), &mut self.output);
        visit("exchange.query".into(), &mut self.exchange.query);
        visit("exchange.out".into(), &mut self.exchange.out);
        for (i, l) in self.layers.iter_mut().enumerate() {
            for (name, m) in [
                ("q", &mut l.local.q),
                ("k", &mut l.local.k),
                ("v", &mut l.local.v),
                ("o", &mut l.local.o),
                ("up", &mut l.up),
                ("gate", &mut l.gate),
                ("down", &mut l.down),
            ] {
                visit(format!("layers.{i}.{name}"), m);
            }
        }
    }

    fn check(&self, rows: &[Row], windows: &[Window]) -> Result<(), Error> {
        for r in rows {
            let Some(w) = windows.iter().find(|w| w.id == r.window) else {
                return Err(Error::Input(format!("row {} has unknown window", r.index)));
            };
            if w.slot >= self.config.max_windows
                || r.token_id as usize >= self.config.vocab_size
                || r.position > 500000
            {
                return Err(Error::Input(format!(
                    "row {} exceeds model vocabulary, window capacity or reference position bound",
                    r.index
                )));
            }
        }
        Ok(())
    }

    fn embedding(&self, row: &Row, windows: &[Window], prediction: bool) -> Vector {
        let mut v = if prediction {
            self.query_embedding.clone()
        } else {
            self.tokens.rows[row.token_id as usize].clone()
        };
        if self.config.window_embeddings {
            let slot = windows
                .iter()
                .find(|w| w.id == row.window)
                .expect("checked window")
                .slot;
            v = add(&v, &self.windows.rows[slot]);
        }
        v
    }

    fn local(
        &self,
        x: &[f64],
        row: &Row,
        memory: &[Entry],
        windows: &[Window],
        layer: usize,
    ) -> Vector {
        let attention = self.layers[layer].local.apply(
            x,
            row,
            memory
                .iter()
                .filter(|e| e.row.window == row.window && visible_to(row, &e.row, windows))
                .map(|e| (e.states[layer].clone(), e.row.position)),
            self.config.rope_theta,
        );
        self.layers[layer].feed_forward(&attention)
    }
    fn global(
        &self,
        x: &[f64],
        row: &Row,
        memory: &[Entry],
        windows: &[Window],
        layer: usize,
    ) -> Vector {
        if !self.config.global_attention {
            return x.to_vec();
        }
        self.exchange.apply(
            x,
            memory
                .iter()
                .filter(|e| visible_to(row, &e.row, windows))
                .map(|e| {
                    // Reuse exactly the local attention projections, including local RoPE on K.
                    let input = norm(&e.states[layer]);
                    let mut key = rope(
                        &self.layers[layer].local.k.apply(&input),
                        e.row.position,
                        self.config.rope_theta,
                    );
                    if self.config.window_embeddings {
                        let slot = windows
                            .iter()
                            .find(|w| w.id == e.row.window)
                            .expect("checked window")
                            .slot;
                        key = add(&key, &self.windows.rows[slot]);
                    }
                    (key, self.layers[layer].local.v.apply(&input))
                }),
        )
    }

    fn query(&self, row: &Row, memory: &[Entry], windows: &[Window]) -> Prediction {
        let mut state = self.embedding(row, windows, true);
        for l in 0..self.layers.len() {
            state = self.local(&state, row, memory, windows, l);
            state = self.global(&state, row, memory, windows, l);
        }
        Prediction {
            row_index: row.index,
            logits: self.output.apply(&norm(&state)),
        }
    }

    /// Full-sequence memory construction, layer by layer with exact causal masks.
    pub fn forward(&self, rows: &[Row], windows: &[Window]) -> Result<Vec<Prediction>, Error> {
        self.check(rows, windows)?;
        let mut memory: Vec<Entry> = rows
            .iter()
            .filter(|r| r.durable)
            .map(|r| Entry {
                row: r.clone(),
                states: vec![self.embedding(r, windows, false)],
                local: Vec::new(),
            })
            .collect();
        for l in 0..self.layers.len() {
            let locals: Vec<Vector> = memory
                .iter()
                .map(|e| self.local(&e.states[l], &e.row, &memory, windows, l))
                .collect();
            for (e, v) in memory.iter_mut().zip(locals) {
                e.local.push(v);
            }
            let outputs: Vec<Vector> = memory
                .iter()
                .map(|e| self.global(&e.local[l], &e.row, &memory, windows, l))
                .collect();
            for (e, v) in memory.iter_mut().zip(outputs) {
                e.states.push(v);
            }
        }
        Ok(rows
            .iter()
            .filter(|r| r.phase != Phase::A)
            .map(|r| self.query(r, &memory, windows))
            .collect())
    }

    fn commit(&self, row: &Row, windows: &[Window], cache: &Cache) -> Entry {
        let mut entry = Entry {
            row: row.clone(),
            states: vec![self.embedding(row, windows, false)],
            local: Vec::new(),
        };
        for l in 0..self.layers.len() {
            let local = self.local(&entry.states[l], row, &cache.entries, windows, l);
            let global = self.global(&local, row, &cache.entries, windows, l);
            entry.local.push(local);
            entry.states.push(global);
        }
        entry
    }

    /// Execute one complete tick: A, ordered B, shared predictions, D, eviction.
    /// Rows must have been validated by the dataset reader. No chunk detach occurs.
    pub fn tick(
        &self,
        rows: &[Row],
        windows: &[Window],
        cache: &mut Cache,
    ) -> Result<Vec<Prediction>, Error> {
        self.check(rows, windows)?;
        if cache
            .model_config
            .as_ref()
            .is_some_and(|c| c != &self.config)
        {
            return Err(Error::Input(
                "cache belongs to a different model configuration or seed; create a fresh cache"
                    .into(),
            ));
        }
        let tick = rows
            .first()
            .ok_or_else(|| Error::Input("tick group is empty".into()))?
            .tick;
        if cache.last_tick.is_some_and(|t| tick <= t)
            || rows.iter().any(|r| r.tick != tick)
            || rows.windows(2).any(|pair| pair[0].phase > pair[1].phase)
        {
            return Err(Error::Input(
                "expected one complete, ordered, strictly increasing tick group".into(),
            ));
        }
        cache.model_config = Some(self.config.clone());
        let mut outputs = Vec::new();
        for row in rows.iter().filter(|r| r.phase != Phase::D) {
            if row.phase == Phase::B {
                outputs.push(self.query(row, &cache.entries, windows));
            }
            if row.durable {
                let e = self.commit(row, windows, cache);
                cache.entries.push(e);
            }
        }
        // All D logits and all D memory states are computed before any D commit.
        outputs.extend(
            rows.iter()
                .filter(|r| r.phase == Phase::D)
                .map(|r| self.query(r, &cache.entries, windows)),
        );
        let committed: Vec<Entry> = rows
            .iter()
            .filter(|r| r.phase == Phase::D && r.durable)
            .map(|r| self.commit(r, windows, cache))
            .collect();
        cache.entries.extend(committed);
        cache.entries.retain(|e| {
            !e.row.removed_at.as_ref().is_some_and(|r| r.tick <= tick)
                && windows
                    .iter()
                    .any(|w| w.id == e.row.window && !w.close.is_some_and(|close| close <= tick))
        });
        cache.last_tick = Some(tick);
        Ok(outputs)
    }
}

/// Aggregated loss retains numerator and effective denominator across chunks.
#[derive(Debug, Clone, Copy, Default)]
pub struct WeightedLoss {
    /// Sum of cross-entropies times effective target weights.
    pub numerator: f64,
    /// Sum of effective weights; never token count or average of chunk means.
    pub effective_weight: f64,
}
impl WeightedLoss {
    /// Stable log-sum-exp cross entropy, with no label shifting.
    pub fn add(&mut self, logits: &[f64], target: usize, weight: f64) -> Result<(), Error> {
        if target >= logits.len()
            || !weight.is_finite()
            || weight < 0.0
            || logits.iter().any(|x| !x.is_finite())
        {
            return Err(Error::Input(
                "loss requires finite logits, valid unshifted target and nonnegative finite weight"
                    .into(),
            ));
        }
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let loss = max + logits.iter().map(|x| (x - max).exp()).sum::<f64>().ln() - logits[target];
        let numerator = self.numerator + loss * weight;
        let effective_weight = self.effective_weight + weight;
        if !numerator.is_finite() || !effective_weight.is_finite() {
            return Err(Error::Input(
                "loss accumulation overflowed; reduce target magnitude or effective weights".into(),
            ));
        }
        self.numerator = numerator;
        self.effective_weight = effective_weight;
        Ok(())
    }
    /// Upstream denominator is clamp_min(1), including zero/sub-unit weights.
    pub fn mean(&self) -> f64 {
        self.numerator / self.effective_weight.max(1.0)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn add(a: &[f64], b: &[f64]) -> Vector {
    a.iter().zip(b).map(|(a, b)| a + b).collect()
}
fn norm(a: &[f64]) -> Vector {
    let scale = (dot(a, a) / a.len() as f64 + 1e-6).sqrt();
    a.iter().map(|x| x / scale).collect()
}
fn softmax(scores: &[f64]) -> Vector {
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = scores.iter().map(|x| (x - max).exp()).sum();
    scores.iter().map(|x| (x - max).exp() / sum).collect()
}
fn rope(x: &[f64], position: usize, theta: f64) -> Vector {
    let mut out = x.to_vec();
    for i in (0..x.len()).step_by(2) {
        let angle = position as f64 / theta.powf(i as f64 / x.len() as f64);
        let (sin, cos) = angle.sin_cos();
        out[i] = x[i] * cos - x[i + 1] * sin;
        out[i + 1] = x[i] * sin + x[i + 1] * cos;
    }
    out
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        // SplitMix64: fixed arithmetic, including seed zero.
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        (z >> 11) as f64 / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Invalid model inputs fail before indexing parameter arrays.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Unsupported configuration or malformed execution group.
    #[error("{0}")]
    Input(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rope_is_local_and_preserves_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(rope(&x, 0, 10000.0), x);
        assert!((dot(&rope(&x, 7, 10000.0), &rope(&x, 7, 10000.0)) - dot(&x, &x)).abs() < 1e-12);
        assert_ne!(rope(&x, 1, 10000.0), rope(&x, 2, 10000.0));
    }
    #[test]
    fn weighted_loss_and_finite_difference() {
        let mut l = WeightedLoss::default();
        l.add(&[0.0, 0.0], 1, 0.25).unwrap();
        assert!((l.mean() - 0.25 * 2f64.ln()).abs() < 1e-14);
        l.add(&[0.0, 0.0], 0, 1.75).unwrap();
        assert!((l.mean() - 2f64.ln()).abs() < 1e-14);
        let x = [0.2, -0.3, 1.0];
        for i in 0..3 {
            let mut plus = x;
            plus[i] += 1e-6;
            let mut minus = x;
            minus[i] -= 1e-6;
            let mut a = WeightedLoss::default();
            a.add(&plus, 1, 2.0).unwrap();
            let mut b = WeightedLoss::default();
            b.add(&minus, 1, 2.0).unwrap();
            let derivative = (a.mean() - b.mean()) / 2e-6;
            assert!((derivative - (softmax(&x)[i] - if i == 1 { 1.0 } else { 0.0 })).abs() < 1e-9);
        }
        assert!(l.add(&[f64::NAN], 0, 1.0).is_err());
        let before = l.numerator;
        assert!(l.add(&[f64::MAX, -f64::MAX], 1, 1.0).is_err());
        assert_eq!(l.numerator, before);
    }
    #[test]
    fn exchange_matches_cached_kv_equation() {
        let identity = Matrix {
            rows: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let exchange = GlobalExchange {
            query: identity.clone(),
            out: identity,
        };
        let x = [1.0, -1.0];
        // K already contains local RoPE and identity offsets; V is unmodified local V.
        let pairs = vec![
            (vec![1.5, 0.0], vec![2.0, 3.0]),
            (vec![0.0, 0.75], vec![4.0, 5.0]),
        ];
        let result = exchange.apply(&x, pairs.into_iter());
        let q = 1.0 / (1.0_f64 + 1e-5).sqrt();
        let weight = 1.0 / (1.0 + (-2.25 * q / 2.0_f64.sqrt()).exp());
        assert!((result[0] - (1.0 + weight * 2.0 + (1.0 - weight) * 4.0)).abs() < 1e-12);
        assert!((result[1] - (-1.0 + weight * 3.0 + (1.0 - weight) * 5.0)).abs() < 1e-12);
        assert_eq!(exchange.apply(&x, std::iter::empty()), x);
        // Changing post-local representations must not change the exchange bank.
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../../fixtures/canonical-single.jsonl")).unwrap();
        let rows: Vec<Row> = serde_json::from_value(fixture["rows"].clone()).unwrap();
        let windows: Vec<Window> = serde_json::from_value(fixture["windows"].clone()).unwrap();
        let model = ReferenceModel::new(ModelConfig::default()).unwrap();
        let row = rows[0].clone();
        let mut query = row.clone();
        query.tick = 1;
        query.phase = Phase::D;
        let mut memory = vec![Entry {
            states: vec![model.embedding(&row, &windows, false)],
            local: vec![vec![0.0; 8]],
            row,
        }];
        let state = vec![0.2; 8];
        let before = model.global(&state, &query, &memory, &windows, 0);
        memory[0].local[0] = vec![999.0; 8];
        assert_eq!(before, model.global(&state, &query, &memory, &windows, 0));
    }
    #[test]
    fn parameter_payloads_fail_closed() {
        let config = ModelConfig::default();
        let model = ReferenceModel::new(config.clone()).unwrap();
        let values = model.parameters();
        let mut bad = values.clone();
        bad.remove("output");
        assert!(ReferenceModel::from_parameters(config.clone(), &bad).is_err());
        let mut bad = values.clone();
        bad.get_mut("output").unwrap()[0].clear();
        assert!(ReferenceModel::from_parameters(config.clone(), &bad).is_err());
        let mut bad = values;
        bad.get_mut("output").unwrap()[0][0] = f64::NAN;
        assert!(ReferenceModel::from_parameters(config, &bad).is_err());
    }
}
