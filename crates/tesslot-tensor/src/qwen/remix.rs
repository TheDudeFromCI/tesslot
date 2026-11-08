//! Native Qwen/Kidra full-trajectory training. Storage chunks never detach a graph.
use super::{
    window::{Frame, Memory, StateRow},
    *,
};
use tesslot_core::{Phase, Row, Window, visible_to};
use tesslot_dataset::{Limits, Sample, TokenizerIdentity};
/// Architecture identity for the general external-first query/memory execution path.
pub const ARCHITECTURE: &str = "kidra-qwen-remix/1";
/// Deliberate bound for this CPU correctness implementation, independent of reader limits.
pub const MAX_ROWS: usize = 512;
/// Provenance of the source vocabulary; this is never inferred from equal dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    /// Random reference weights explicitly initialized for the target tokenizer.
    Synthetic,
    /// Operator-supplied tiny Qwen weights already aligned with the declared tokenizer.
    TokenAligned,
    /// Append-only control/tokenizer conversion of a user-supplied pretrained checkpoint.
    Pretrained,
}
/// Checkpointed training contract; one committed update consumes one complete sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingState {
    /// `kidra-qwen-remix/1`.
    pub architecture: String,
    /// `kidra-token-chunk/1`.
    pub export_format: String,
    /// Exact target tokenizer, including native controls.
    pub tokenizer: TokenizerIdentity,
    /// Initialization seed for synthetic/reference parameters.
    pub seed: u64,
    /// Source weight provenance; no implicit ASCII/pretrained token remapping.
    pub source: Source,
    /// Complete imported source SafeTensors hash.
    pub source_sha256: String,
    /// Versioned tensor mapping and initialization contract.
    pub conversion_format: String,
    /// Complete dataset file hash, if bound by the CLI.
    pub dataset_sha256: Option<String>,
    /// Successful whole-sample optimizer updates, also the resumed sequential sample cursor.
    pub completed_samples: u64,
}
impl TrainingState {
    pub(super) fn validate(
        &self,
        config: &Config,
        windows: Option<usize>,
        step: u64,
    ) -> Result<(), String> {
        self.tokenizer.validate().map_err(|e| e.to_string())?;
        if self.architecture != ARCHITECTURE
            || self.export_format != tesslot_dataset::FORMAT
            || self.tokenizer.vocab_size != config.vocab_size
            || config.bos_token_id != self.tokenizer.special_tokens.bos as usize
            || config.eos_token_id != self.tokenizer.special_tokens.eos as usize
            || !windows.is_some_and(|w| (1..=64).contains(&w))
            || self.completed_samples != step
            || self.conversion_format
                != if matches!(self.source, Source::Pretrained) {
                    "kidra-pretrained/1"
                } else {
                    tesslot_artifacts::qwen::FORMAT
                }
            || self.source_sha256.len() != 64
            || !self
                .source_sha256
                .bytes()
                .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
            || self.dataset_sha256.as_ref().is_some_and(|s| {
                s.len() != 64
                    || !s
                        .bytes()
                        .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
            })
        {
            return Err(
                "incompatible Qwen/Remix architecture, tokenizer, capacity, cursor or dataset hash"
                    .into(),
            );
        }
        Ok(())
    }
}
/// Causal prediction computed before its target's insertion/commit.
pub struct Prediction {
    /// Sample-global row index; no target shifting.
    pub row_index: usize,
    /// FP32 vocabulary logits, shape [1,vocabulary].
    pub logits: T,
}
struct Entry {
    row: Row,
    memory: Memory,
}
/// Ephemeral attached projected KV. It cannot be reused after an update or across samples.
#[derive(Default)]
pub struct Cache {
    entries: Vec<Entry>,
    owner: Option<(usize, u64)>,
    sample: Option<String>,
    last: Option<u64>,
    next_row: usize,
}
impl Cache {
    /// Number of live durable rows; NOOP never adds an entry.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// Whether there are any live keys.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Logical identities of live cache rows, for lifecycle inspection.
    pub fn windows(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.row.window.as_str()).collect()
    }
}
/// Reference execution modes used for training/incremental parity gates.
#[derive(Clone, Copy)]
pub enum Execution {
    /// One dense full-sequence graph with masks on memory and queries in every layer.
    Dense,
    /// Real projected KV carried across ticks and original storage chunks.
    Chunked,
}
impl Model {
    /// Initialize a tiny synthetic Qwen with an exactly matched tokenizer vocabulary.
    /// This does not reinterpret or transfer ASCII/pretrained embedding rows.
    pub fn remix_demo(
        tokenizer: TokenizerIdentity,
        windows: usize,
        seed: u64,
        training: bool,
    ) -> Result<Self, String> {
        tokenizer.validate().map_err(|e| e.to_string())?;
        let mut config =
            Config::from_json(include_bytes!("../../../../fixtures/tiny-qwen/config.json"))?;
        config.max_position_embeddings = 256;
        config.vocab_size = tokenizer.vocab_size;
        config.bos_token_id = tokenizer.special_tokens.bos as usize;
        config.eos_token_id = tokenizer.special_tokens.eos as usize;
        let tensors = tesslot_artifacts::qwen::random_source(&config, seed, 128 * 1024)
            .map_err(|e| e.to_string())?;
        let bytes =
            encode_f32_with_metadata(&tensors, BTreeMap::new()).map_err(|e| e.to_string())?;
        let mut model =
            Self::remix_from_aligned(config, &bytes, tokenizer, windows, seed, training)?;
        model
            .remix
            .as_mut()
            .ok_or("missing synthetic provenance")?
            .source = Source::Synthetic;
        Ok(model)
    }
    /// Convert tiny token-aligned Qwen weights; caller must supply the exact weight tokenizer.
    /// Vocabulary/control mismatches are rejected, never automatically remapped.
    pub fn remix_from_aligned(
        config: Config,
        bytes: &[u8],
        tokenizer: TokenizerIdentity,
        windows: usize,
        seed: u64,
        training: bool,
    ) -> Result<Self, String> {
        if bytes.len() > 2 * 1024 * 1024 {
            return Err("tiny Qwen source exceeds 2 MiB".into());
        }
        let state = TrainingState {
            architecture: ARCHITECTURE.into(),
            export_format: tesslot_dataset::FORMAT.into(),
            tokenizer,
            seed,
            source: Source::TokenAligned,
            source_sha256: format!("{:x}", Sha256::digest(bytes)),
            conversion_format: tesslot_artifacts::qwen::FORMAT.into(),
            dataset_sha256: None,
            completed_samples: 0,
        };
        state.validate(&config, Some(windows), 0)?;
        let converted = tesslot_artifacts::qwen::convert(
            &mut Cursor::new(bytes),
            &config,
            windows,
            seed,
            128 * 1024,
        )
        .map_err(|e| e.to_string())?;
        let bytes = encode_f32_with_metadata(&converted.tensors, BTreeMap::new())
            .map_err(|e| e.to_string())?;
        let mut model = Self::load_architecture(config, &bytes, training, Some(windows))?;
        model.remix = Some(state);
        Ok(model)
    }
    /// Load a complete local Kidra conversion and bind its exact export/training identity.
    pub fn from_pretrained(
        path: &std::path::Path,
        training: bool,
        plan: super::runtime::Plan,
        max_bytes: usize,
    ) -> Result<Self, String> {
        let artifact = tesslot_artifacts::pretrained::load_converted(path, max_bytes)
            .map_err(|e| e.to_string())?;
        let state = TrainingState {
            architecture: ARCHITECTURE.into(),
            export_format: tesslot_dataset::FORMAT.into(),
            tokenizer: artifact.tokenizer,
            seed: artifact.seed,
            source: Source::Pretrained,
            source_sha256: artifact.sha256,
            conversion_format: "kidra-pretrained/1".into(),
            dataset_sha256: None,
            completed_samples: 0,
        };
        state.validate(&artifact.config, Some(artifact.windows), 0)?;
        let mut model = Self::from_tensors(
            artifact.config,
            artifact.weights,
            Some(artifact.windows),
            training,
            plan,
        )?;
        model.remix = Some(state);
        Ok(model)
    }
    /// Checkpointed tokenizer/dataset contract and whole-sample update cursor.
    pub fn remix_state(&self) -> Option<&TrainingState> {
        self.remix.as_ref()
    }
    /// Bind an exact file hash for sequential CLI resumption; a changed file is rejected.
    pub fn bind_remix_dataset(&mut self, hash: String) -> Result<(), String> {
        let state = self.remix.as_mut().ok_or("Qwen/Remix model required")?;
        if state.dataset_sha256.is_none() && state.completed_samples > 0 {
            return Err("cannot attach a sequential dataset cursor after unbound training".into());
        }
        if state.dataset_sha256.as_ref().is_some_and(|s| s != &hash) {
            return Err("checkpoint dataset hash differs from the supplied export".into());
        }
        let old = state.dataset_sha256.replace(hash);
        if let Err(e) = state.validate(&self.config, self.exchange_windows, self.step) {
            state.dataset_sha256 = old;
            return Err(e);
        }
        Ok(())
    }
    /// Check tokenizer, export structure, lifecycle, slots and CPU allocation bounds without a forward pass.
    pub fn validate_remix_sample(&self, sample: &Sample) -> Result<(), String> {
        let state = self.remix.as_ref().ok_or("Qwen/Remix model required")?;
        state.validate(&self.config, self.exchange_windows, self.step)?;
        sample
            .validate(
                &state.tokenizer,
                &Limits {
                    sample_rows: MAX_ROWS,
                    window_slots: self.exchange_windows.unwrap_or(0),
                    ..Limits::default()
                },
            )
            .map_err(|e| e.to_string())?;
        if sample
            .rows
            .iter()
            .any(|r| r.position >= self.config.max_position_embeddings)
        {
            return Err("sample exceeds tiny Qwen local context limit".into());
        }
        let weight = sample.rows.iter().map(|r| r.loss_weight).sum::<f64>();
        if !weight.is_finite()
            || sample
                .rows
                .iter()
                .any(|r| r.loss_weight > 0.0 && (r.loss_weight / weight.max(1.0)) as f32 == 0.0)
        {
            return Err(
                "effective weight sum or normalized weight is not representable for FP32 training"
                    .into(),
            );
        }
        Ok(())
    }
    pub(super) fn state_row(
        &self,
        row: &Row,
        windows: &[Window],
        query: bool,
    ) -> Result<StateRow, String> {
        let w = windows
            .iter()
            .find(|w| w.id == row.window)
            .ok_or("unknown logical window")?;
        if row.token_id as usize >= self.config.vocab_size
            || row.position >= self.config.max_position_embeddings
            || w.slot >= self.exchange_windows.unwrap_or(0)
        {
            return Err("row token/position/slot exceeds model bounds".into());
        }
        Ok(StateRow {
            slot: w.slot,
            position: row.position,
            token: if query { None } else { Some(row.token_id) },
        })
    }
    fn encode(
        &self,
        rows: &[(&Row, bool)],
        windows: &[Window],
        memory: &[Entry],
    ) -> Result<Frame, String> {
        let states = rows
            .iter()
            .map(|(r, q)| self.state_row(r, windows, *q))
            .collect::<Result<Vec<_>, _>>()?;
        let allowed = rows
            .iter()
            .flat_map(|(q, _)| memory.iter().map(move |e| visible_to(q, &e.row, windows)))
            .collect::<Vec<_>>();
        self.window_frame(
            &states,
            0,
            &memory.iter().map(|e| &e.memory).collect::<Vec<_>>(),
            &allowed,
        )
    }
    fn entry(
        &self,
        row: &Row,
        windows: &[Window],
        frame: &Frame,
        index: usize,
    ) -> Result<Entry, String> {
        let slot = self.state_row(row, windows, false)?.slot;
        let width = self.config.kv_width()?;
        Ok(Entry {
            row: row.clone(),
            memory: Memory {
                slot,
                keys: frame
                    .keys
                    .iter()
                    .map(|t| t.clone().slice([index..index + 1, 0..width]))
                    .collect(),
                values: frame
                    .values
                    .iter()
                    .map(|t| t.clone().slice([index..index + 1, 0..width]))
                    .collect(),
            },
        })
    }
    /// Execute one validated tick; same-tick D decisions/commits share the after-B snapshot.
    /// The sample identity prevents accidental cross-trajectory cache packing.
    pub fn remix_tick(
        &self,
        sample_id: &str,
        rows: &[Row],
        windows: &[Window],
        cache: &mut Cache,
    ) -> Result<Vec<Prediction>, String> {
        if self.remix.is_none() || sample_id.is_empty() {
            return Err("Qwen/Remix model and nonempty sample identity required".into());
        }
        let tick = rows.first().ok_or("empty tick")?.tick;
        let next_row = cache
            .next_row
            .checked_add(rows.len())
            .ok_or("sample row index overflow")?;
        if rows
            .iter()
            .enumerate()
            .any(|(i, r)| r.index != cache.next_row + i)
        {
            return Err("missing/repeated prefix rows; retain one cache across all chunks".into());
        }
        if rows.len() > MAX_ROWS
            || cache.entries.len() + rows.iter().filter(|r| r.durable).count() > MAX_ROWS
        {
            return Err("tiny projected-KV budget exceeded".into());
        }
        if cache
            .owner
            .is_some_and(|owner| owner != (self.id, self.step))
            || cache.sample.as_ref().is_some_and(|s| s != sample_id)
        {
            return Err(
                "cache belongs to another model, update or sample; create a fresh cache".into(),
            );
        }
        if cache.last.is_some_and(|last| tick <= last)
            || rows.iter().any(|r| r.tick != tick)
            || rows.windows(2).any(|r| r[0].phase > r[1].phase)
        {
            return Err("tick order must be strictly increasing and external-first".into());
        }
        for row in rows {
            self.state_row(row, windows, false)?;
        }
        cache.owner = Some((self.id, self.step));
        cache.sample = Some(sample_id.into());
        cache.entries.retain(|e| {
            windows
                .iter()
                .any(|w| w.id == e.row.window && !w.close.is_some_and(|close| close < tick))
        });
        let mut output = Vec::new();
        for row in rows.iter().filter(|r| r.phase != Phase::D) {
            let query = row.phase == Phase::B && row.loss_weight > 0.0;
            let mut frame_rows = Vec::new();
            if query {
                frame_rows.push((row, true));
            }
            if row.durable {
                frame_rows.push((row, false));
            }
            if !frame_rows.is_empty() {
                let frame = self.encode(&frame_rows, windows, &cache.entries)?;
                if query {
                    output.push(Prediction {
                        row_index: row.index,
                        logits: frame
                            .logits
                            .clone()
                            .slice([0..1, 0..self.config.vocab_size]),
                    });
                }
                if row.durable {
                    cache
                        .entries
                        .push(self.entry(row, windows, &frame, usize::from(query))?);
                }
            }
        }
        let decisions = rows
            .iter()
            .filter(|r| r.phase == Phase::D)
            .collect::<Vec<_>>();
        if !decisions.is_empty() {
            let mut frame_rows = decisions.iter().map(|r| (*r, true)).collect::<Vec<_>>();
            let durable = decisions
                .iter()
                .copied()
                .filter(|r| r.durable)
                .collect::<Vec<_>>();
            frame_rows.extend(durable.iter().map(|r| (*r, false)));
            let frame = self.encode(&frame_rows, windows, &cache.entries)?;
            output.extend(decisions.iter().enumerate().map(|(i, r)| {
                Prediction {
                    row_index: r.index,
                    logits: frame
                        .logits
                        .clone()
                        .slice([i..i + 1, 0..self.config.vocab_size]),
                }
            }));
            for (i, row) in durable.iter().enumerate() {
                cache
                    .entries
                    .push(self.entry(row, windows, &frame, decisions.len() + i)?);
            }
        }
        // Graphs used by surviving entries/losses remain attached after physical eviction.
        cache.entries.retain(|e| {
            !e.row.removed_at.as_ref().is_some_and(|at| at.tick <= tick)
                && windows
                    .iter()
                    .any(|w| w.id == e.row.window && !w.close.is_some_and(|close| close <= tick))
        });
        cache.last = Some(tick);
        cache.next_row = next_row;
        Ok(output)
    }
    /// Replay a complete sample. Chunked execution carries the same attached cache throughout.
    pub fn remix_forward(
        &self,
        sample: &Sample,
        execution: Execution,
    ) -> Result<Vec<Prediction>, String> {
        self.validate_remix_sample(sample)?;
        match execution {
            Execution::Dense => {
                if sample.rows.is_empty() {
                    return Ok(Vec::new());
                }
                let memory = sample.rows.iter().filter(|r| r.durable).collect::<Vec<_>>();
                let queries = sample
                    .rows
                    .iter()
                    .filter(|r| r.phase == Phase::D || (r.phase == Phase::B && r.loss_weight > 0.0))
                    .collect::<Vec<_>>();
                let rows = memory
                    .iter()
                    .map(|r| (*r, false))
                    .chain(queries.iter().map(|r| (*r, true)))
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
                let frame = self.window_frame(&states, memory.len(), &[], &allowed)?;
                Ok(queries
                    .iter()
                    .enumerate()
                    .map(|(i, r)| Prediction {
                        row_index: r.index,
                        logits: frame.logits.clone().slice([
                            memory.len() + i..memory.len() + i + 1,
                            0..self.config.vocab_size,
                        ]),
                    })
                    .collect())
            }
            Execution::Chunked => {
                let mut cache = Cache::default();
                let mut output = Vec::new();
                let mut begin = 0;
                for &end in &sample.chunk_ends {
                    let mut i = begin;
                    while i < end {
                        let next = i + sample.rows[i..end]
                            .partition_point(|r| r.tick == sample.rows[i].tick);
                        output.extend(self.remix_tick(
                            &sample.metadata.trajectory_uuid,
                            &sample.rows[i..next],
                            &sample.metadata.windows,
                            &mut cache,
                        )?);
                        i = next;
                    }
                    begin = end;
                }
                Ok(output)
            }
        }
    }
    /// Upstream unshifted objective: sum(weight × CE) / max(total effective weight, 1).
    pub fn remix_loss(
        &self,
        predictions: &[Prediction],
        sample: &Sample,
    ) -> Result<Tensor<1>, String> {
        self.validate_remix_sample(sample)?;
        let expected = sample
            .rows
            .iter()
            .filter(|r| r.loss_weight > 0.0)
            .map(|r| r.index)
            .collect::<std::collections::BTreeSet<_>>();
        let weight = sample.rows.iter().map(|r| r.loss_weight).sum::<f64>();
        if !weight.is_finite() {
            return Err("effective target weight sum overflow".into());
        }
        let denominator = weight.max(1.0);
        let mut seen = std::collections::BTreeSet::new();
        let mut total = self.parameters["model.embed_tokens.weight"].clone().sum() * 0.0;
        for p in predictions {
            let row = sample
                .rows
                .get(p.row_index)
                .ok_or("prediction index outside sample")?;
            if !seen.insert(p.row_index)
                || (row.phase != Phase::B && row.phase != Phase::D)
                || p.logits.dims() != [1, self.config.vocab_size]
            {
                return Err("duplicate or malformed prediction row".into());
            }
            if row.loss_weight > 0.0 {
                let normalized = (row.loss_weight / denominator) as f32;
                if normalized == 0.0 {
                    return Err("normalized target weight underflows FP32".into());
                }
                total = total
                    - log_softmax(p.logits.clone(), 1)
                        .slice([0..1, row.token_id as usize..row.token_id as usize + 1])
                        .sum()
                        * normalized;
            }
        }
        if !expected.is_subset(&seen) {
            return Err("missing effective training targets".into());
        }
        Ok(total)
    }
    /// CPU diagnostic gradient snapshot; unused trainable coordinates receive exact zeros.
    pub fn remix_gradients(&self, loss: &Tensor<1>) -> Result<BTreeMap<String, Vec<f32>>, String> {
        if !self.training || self.remix.is_none() {
            return Err("gradient diagnostics require a training Qwen/Remix model".into());
        }
        let gradients = loss.backward();
        self.parameters
            .iter()
            .map(|(name, p)| {
                let values = match p.grad(&gradients) {
                    Some(g) => g
                        .into_data()
                        .try_to_vec::<f32>()
                        .map_err(|e| e.to_string())?,
                    None => vec![0.0; p.dims().iter().product()],
                };
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(format!("nonfinite gradient {name}"));
                }
                Ok((name.clone(), values))
            })
            .collect()
    }
    /// One optimizer update per complete trajectory, with one backward across all chunks.
    pub fn train_remix_step(&mut self, sample: &Sample) -> Result<Metrics, String> {
        self.validate_remix_sample(sample)?;
        if !sample.rows.iter().any(|r| r.loss_weight > 0.0) {
            return Err("sample has no effective training targets".into());
        }
        let output = self.remix_forward(sample, Execution::Chunked)?;
        let metrics = self.update(self.remix_loss(&output, sample)?)?;
        self.remix
            .as_mut()
            .ok_or("missing Remix training state")?
            .completed_samples += 1;
        Ok(metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample() -> Sample {
        let identity: TokenizerIdentity = serde_json::from_slice(include_bytes!(
            "../../../../fixtures/tokenizer-identity.json"
        ))
        .unwrap();
        tesslot_dataset::Reader::new(
            Cursor::new(include_bytes!("../../../../fixtures/autograd.jsonl")),
            identity,
            Limits::default(),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
    }
    fn model(training: bool) -> Model {
        let s = sample();
        let mut c = Config::from_json(include_bytes!("../../../../fixtures/tiny-qwen/config.json"))
            .unwrap();
        c.hidden_size = 4;
        c.intermediate_size = 8;
        c.head_dim = 2;
        c.num_attention_heads = 2;
        c.num_key_value_heads = 1;
        c.num_hidden_layers = 2;
        c.vocab_size = 101;
        c.bos_token_id = 1;
        c.eos_token_id = 2;
        let source = tesslot_artifacts::qwen::random_source(&c, 42, 128 * 1024).unwrap();
        let bytes = encode_f32_with_metadata(&source, BTreeMap::new()).unwrap();
        let identity = TokenizerIdentity {
            sha256: s.metadata.tokenizer_sha256.clone(),
            vocab_size: 101,
            special_tokens: s.metadata.special_tokens.clone(),
        };
        Model::remix_from_aligned(c, &bytes, identity, 7, 42, training).unwrap()
    }
    #[test]
    #[ignore = "opt-in two-layer finite differences and retained-prefix gradient proof"]
    fn finite_differences_and_replaced_prefix_graph_remain_correct() {
        let s = sample();
        let m = model(true);
        let p = m.remix_forward(&s, Execution::Dense).unwrap();
        let gradients = m.remix_gradients(&m.remix_loss(&p, &s).unwrap()).unwrap();
        let coordinates = [
            ("model.embed_tokens.weight", 92 * 4),
            ("model.layers.0.self_attn.k_proj.weight", 0),
            ("model.layers.0.self_attn.v_proj.weight", 0),
            ("model.layers.0.self_attn.q_norm.weight", 0),
            ("kidra.exchange.query.weight", 2),
            ("kidra.exchange.out.weight", 1),
            ("kidra.window_embeddings.weight", 4 * 4),
            ("kidra.query_embedding.weight", 0),
            ("model.norm.weight", 0),
        ];
        for (name, index) in coordinates {
            let eval = |delta: f32| {
                let mut model = model(false);
                let tensor = &model.parameters[name];
                let shape = tensor.dims();
                let mut values = tensor.clone().into_data().try_to_vec::<f32>().unwrap();
                values[index] += delta;
                model.parameters.insert(
                    name.into(),
                    T::from_data(TensorData::new(values, shape), (&model.device, DType::F32)),
                );
                let p = model.remix_forward(&s, Execution::Dense).unwrap();
                model
                    .remix_loss(&p, &s)
                    .unwrap()
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap()[0]
            };
            let numeric = (eval(1e-3) - eval(-1e-3)) / 2e-3;
            let analytic = gradients[name][index];
            assert!(
                (numeric - analytic).abs() < 2e-3 + 0.02 * analytic.abs(),
                "finite difference {name}[{index}]: {numeric} vs {analytic}"
            );
        }
        let later_gradient = |detach: bool| {
            let model = model(true);
            let mut cache = Cache::default();
            for tick in s.ticks().take_while(|rows| rows[0].tick < 2) {
                model
                    .remix_tick("sample", tick, &s.metadata.windows, &mut cache)
                    .unwrap();
            }
            if detach {
                for e in &mut cache.entries {
                    for t in e.memory.keys.iter_mut().chain(&mut e.memory.values) {
                        *t = t.clone().detach();
                    }
                }
            }
            let out = model
                .remix_tick(
                    "sample",
                    s.ticks().find(|rows| rows[0].tick == 2).unwrap(),
                    &s.metadata.windows,
                    &mut cache,
                )
                .unwrap();
            assert!(
                cache.entries.iter().all(|e| e.row.token_id != 92),
                "replaced prefix key must be physically evicted"
            );
            let p = &out[0];
            let target = s.rows[p.row_index].token_id as usize;
            let loss = -log_softmax(p.logits.clone(), 1)
                .slice([0..1, target..target + 1])
                .sum();
            model.remix_gradients(&loss).unwrap()["model.embed_tokens.weight"][92 * 4..93 * 4]
                .to_vec()
        };
        let attached = later_gradient(false);
        let detached = later_gradient(true);
        assert!(
            attached
                .iter()
                .zip(detached)
                .any(|(a, b)| (a - b).abs() > 1e-7),
            "later loss must retain gradient through a replaced prefix's surviving dependent representation"
        );
    }
}
