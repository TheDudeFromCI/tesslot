//! Native, bounded JSONL reader for the pinned Trajectory Remix export contract.
//!
//! One complete trajectory is materialized at a time. Storage chunks are never
//! treated as independent training samples. No tokenizer or Python runtime needed.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{BufRead, Read},
};
use tesslot_core::{Phase, Row, SpecialTokens, Window};

mod sampling;
pub use sampling::{Shuffled, shuffle_samples};

/// Supported export schema.
pub const FORMAT: &str = "kidra-token-chunk/1";
/// Exact cache policy emitted by the pinned exporter.
pub const CACHE_POLICY: &str = "durable-only; evict closed windows after phase D; full-gradients requires attached prefix graph";

/// A storage chunk. Row-aligned arrays are validated against the authoritative rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Chunk {
    /// Export format version.
    pub format: String,
    /// Human-readable sample identity.
    pub sample_id: String,
    /// Immutable trajectory UUID and slot seed.
    pub trajectory_uuid: String,
    /// Project slot limit; null means no explicit exporter limit.
    pub max_window_slots: Option<usize>,
    /// Versioned slot policy.
    pub slot_policy: String,
    /// Exact normalized tokenizer fingerprint from the exporter.
    pub tokenizer_sha256: String,
    /// Dedicated native controls.
    pub special_tokens: SpecialTokens,
    /// Zero-based contiguous chunk ordinal.
    pub chunk_index: usize,
    /// Declared number of chunks in this sample.
    pub chunk_count: usize,
    /// Declared number of rows in this sample.
    pub sample_row_count: usize,
    /// Inclusive start tick.
    pub tick_start: u64,
    /// Inclusive end tick.
    pub tick_end: u64,
    /// Required preceding row, -1 initially.
    pub requires_prefix_through_row: i64,
    /// Window lifetimes; order is not slot order.
    pub windows: Vec<Window>,
    /// Explicit logical-to-numeric mapping.
    pub window_slots: BTreeMap<String, usize>,
    /// Replay records, preserving parallel ticks.
    pub rows: Vec<Row>,
    /// Redundant row-aligned IDs.
    pub token_ids: Vec<u32>,
    /// Unshifted target IDs or -100.
    pub labels: Vec<i64>,
    /// Integer 0/1 row-aligned masks.
    pub loss_mask: Vec<u8>,
    /// Effective target weights.
    pub loss_weights: Vec<f64>,
    /// Exact causal predicate version.
    pub attention_rule: String,
    /// Exact cache semantics.
    pub cache_policy: String,
}

/// Expected training tokenizer; this must come from trusted model metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerIdentity {
    /// Fingerprint of tokenizer.to_str() after no-padding/no-truncation normalization.
    pub sha256: String,
    /// Vocabulary upper bound, not inferred from a sample's largest token.
    pub vocab_size: usize,
    /// Native control mapping.
    pub special_tokens: SpecialTokens,
}

impl TokenizerIdentity {
    /// Reject malformed hashes and impossible controls before reading exports.
    pub fn validate(&self) -> Result<(), Error> {
        if self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        {
            return Err(Error::Contract(
                "expected lowercase 64-digit tokenizer SHA-256".into(),
            ));
        }
        self.special_tokens
            .validate(self.vocab_size)
            .map_err(|e| Error::Contract(e.to_string()))
    }
}

/// Explicit bounds for untrusted input. Memory is bounded per trajectory.
#[derive(Debug, Clone)]
pub struct Limits {
    /// Maximum encoded line length.
    pub line_bytes: usize,
    /// Maximum rows per sample (upstream compiler limit).
    pub sample_rows: usize,
    /// Maximum slot capacity accepted by this consumer.
    pub window_slots: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            line_bytes: 64 * 1024 * 1024,
            sample_rows: 500_000,
            window_slots: 4096,
        }
    }
}

/// Reassembled trajectory. Boundaries preserve future attached-graph execution.
#[derive(Debug, Clone)]
pub struct Sample {
    /// First chunk contains immutable sample metadata; its row arrays remain first-chunk-only.
    pub metadata: Chunk,
    /// Complete ordered replay rows.
    pub rows: Vec<Row>,
    /// Exclusive cumulative row count at each storage boundary.
    pub chunk_ends: Vec<usize>,
}
impl Sample {
    /// Iterate complete parallel tick groups without flattening window decisions.
    pub fn ticks(&self) -> impl Iterator<Item = &[Row]> {
        let mut remaining = self.rows.as_slice();
        std::iter::from_fn(move || {
            let tick = remaining.first()?.tick;
            let n = remaining.partition_point(|r| r.tick == tick);
            let (group, rest) = remaining.split_at(n);
            remaining = rest;
            Some(group)
        })
    }
    /// Stable trajectory-level split/shuffle key; chunks always stay together.
    pub fn order_key(&self, seed: u64) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(seed.to_le_bytes());
        hash.update(self.metadata.trajectory_uuid.as_bytes());
        hash.finalize().into()
    }
    /// Deterministic train/dev/test assignment in basis points (10000 total).
    pub fn split(&self, seed: u64, train: u16, development: u16) -> Result<Split, Error> {
        if u32::from(train) + u32::from(development) > 10000 {
            return fail("split weights exceed 10000 basis points");
        }
        let key = self.order_key(seed);
        let value = u16::from_le_bytes([key[0], key[1]]) as u32 * 10000 / 65536;
        Ok(if value < u32::from(train) {
            Split::Train
        } else if value < u32::from(train) + u32::from(development) {
            Split::Development
        } else {
            Split::Test
        })
    }
}

/// Trajectory-level partition, independent of storage chunk sizes.
#[derive(Debug, PartialEq, Eq)]
pub enum Split {
    /// Training partition.
    Train,
    /// Validation partition.
    Development,
    /// Held-out partition.
    Test,
}

/// Streaming reader. Iterate to EOF to certify whole-file completeness.
pub struct Reader<R> {
    input: R,
    expected: TokenizerIdentity,
    limits: Limits,
    line: usize,
    pending: Option<Chunk>,
    samples: BTreeSet<String>,
    trajectories: BTreeSet<String>,
    ended: bool,
}
impl<R: BufRead> Reader<R> {
    /// Construct with an explicit trusted tokenizer contract.
    pub fn new(input: R, expected: TokenizerIdentity, limits: Limits) -> Result<Self, Error> {
        expected.validate()?;
        if limits.line_bytes == 0 || limits.sample_rows == 0 || limits.window_slots == 0 {
            return fail("reader limits must be positive");
        }
        Ok(Self {
            input,
            expected,
            limits,
            line: 0,
            pending: None,
            samples: BTreeSet::new(),
            trajectories: BTreeSet::new(),
            ended: false,
        })
    }

    fn chunk(&mut self) -> Result<Option<Chunk>, Error> {
        if let Some(c) = self.pending.take() {
            return Ok(Some(c));
        }
        loop {
            let mut bytes = Vec::new();
            let n = self
                .input
                .by_ref()
                .take(self.limits.line_bytes as u64 + 1)
                .read_until(b'\n', &mut bytes)?;
            if n == 0 {
                return Ok(None);
            }
            self.line += 1;
            if n > self.limits.line_bytes {
                return fail(format!(
                    "line {} exceeds {} bytes",
                    self.line, self.limits.line_bytes
                ));
            }
            if bytes.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let c: Chunk = serde_json::from_slice(&bytes)
                .map_err(|e| Error::Contract(format!("line {}: {e}", self.line)))?;
            validate_chunk(&c, &self.expected, &self.limits).map_err(|e| {
                Error::Contract(format!(
                    "line {}, sample {} chunk {}: {e}",
                    self.line, c.sample_id, c.chunk_index
                ))
            })?;
            return Ok(Some(c));
        }
    }

    fn sample(&mut self) -> Result<Option<Sample>, Error> {
        let Some(first) = self.chunk()? else {
            return Ok(None);
        };
        if first.chunk_index != 0
            || first.tick_start != 0
            || first.requires_prefix_through_row != -1
        {
            return fail("sample must start with chunk 0, tick 0 and prefix -1");
        }
        if !self.samples.insert(first.sample_id.clone())
            || !self.trajectories.insert(first.trajectory_uuid.clone())
        {
            return fail("duplicate/noncontiguous sample or trajectory UUID");
        }
        let mut rows = first.rows.clone();
        let mut ends = vec![rows.len()];
        let mut tick_end = first.tick_end;
        for index in 1..first.chunk_count {
            let c = self.chunk()?.ok_or_else(|| {
                Error::Contract(format!(
                    "sample {}: missing suffix chunk {index}",
                    first.sample_id
                ))
            })?;
            if c.sample_id != first.sample_id
                || c.chunk_index != index
                || c.requires_prefix_through_row != rows.len() as i64 - 1
                || c.tick_start != tick_end + 1
            {
                return fail("missing/duplicate/reordered chunk, prefix or tick interval");
            }
            if !same_metadata(&first, &c) {
                return fail("sample metadata changed between chunks");
            }
            if rows.len() + c.rows.len() > first.sample_row_count {
                return fail("rows exceed declared sample_row_count");
            }
            tick_end = c.tick_end;
            rows.extend(c.rows);
            ends.push(rows.len());
        }
        if rows.len() != first.sample_row_count {
            return fail("sample_row_count mismatch or missing rows");
        }
        validate_rows(&rows, &first)?;
        let next = self.chunk()?;
        if let Some(c) = &next
            && (c.sample_id == first.sample_id || c.chunk_index != 0)
        {
            return fail("duplicate chunk or noncontiguous next sample");
        }
        self.pending = next;
        Ok(Some(Sample {
            metadata: first,
            rows,
            chunk_ends: ends,
        }))
    }
}
impl<R: BufRead> Iterator for Reader<R> {
    type Item = Result<Sample, Error>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.ended {
            return None;
        }
        match self.sample() {
            Ok(Some(s)) => Some(Ok(s)),
            Ok(None) => {
                self.ended = true;
                None
            }
            Err(e) => {
                self.ended = true;
                Some(Err(e))
            }
        }
    }
}

fn same_metadata(a: &Chunk, b: &Chunk) -> bool {
    a.trajectory_uuid == b.trajectory_uuid
        && a.tokenizer_sha256 == b.tokenizer_sha256
        && a.special_tokens == b.special_tokens
        && a.chunk_count == b.chunk_count
        && a.sample_row_count == b.sample_row_count
        && a.windows == b.windows
        && a.window_slots == b.window_slots
        && a.slot_policy == b.slot_policy
        && a.max_window_slots == b.max_window_slots
}

fn validate_chunk(c: &Chunk, expected: &TokenizerIdentity, limits: &Limits) -> Result<(), Error> {
    if c.format != FORMAT
        || c.attention_rule != "external-first-v1"
        || c.cache_policy != CACHE_POLICY
    {
        return fail("unsupported export, attention or cache policy version");
    }
    if c.tokenizer_sha256 != expected.sha256 || c.special_tokens != expected.special_tokens {
        return fail("tokenizer identity or special-token mapping incompatible with model");
    }
    if c.sample_id.is_empty() || !valid_uuid(&c.trajectory_uuid) {
        return fail("sample identity or trajectory UUID is invalid");
    }
    if c.chunk_count == 0
        || c.chunk_count > 100000
        || c.chunk_index >= c.chunk_count
        || c.sample_row_count > limits.sample_rows
        || c.rows.len() > c.sample_row_count
        || c.tick_start > c.tick_end
        || c.tick_end >= 100000
    {
        return fail("invalid chunk counts, row counts or tick bounds");
    }
    if !matches!(c.slot_policy.as_str(), "uuid-hash-v1" | "declaration-v1") {
        return fail("unsupported slot policy");
    }
    let capacity = c.max_window_slots.unwrap_or(c.windows.len());
    if capacity == 0
        || capacity > limits.window_slots
        || c.windows.len() > capacity
        || c.window_slots.len() != c.windows.len()
    {
        return fail("invalid window capacity or slot mapping");
    }
    let mut ids = BTreeSet::new();
    let mut slots = BTreeSet::new();
    for w in &c.windows {
        if w.id.is_empty()
            || w.kind.is_empty()
            || !ids.insert(&w.id)
            || !slots.insert(w.slot)
            || w.slot >= capacity
            || c.window_slots.get(&w.id) != Some(&w.slot)
            || w.open >= 100000
            || w.close.is_some_and(|t| t < w.open || t >= 100000)
        {
            return fail("duplicate/invalid window, slot, mapping or lifetime");
        }
    }
    let n = c.rows.len();
    if [
        c.token_ids.len(),
        c.labels.len(),
        c.loss_mask.len(),
        c.loss_weights.len(),
    ]
    .iter()
    .any(|&len| len != n)
    {
        return fail("row-aligned arrays have different lengths");
    }
    for (i, r) in c.rows.iter().enumerate() {
        if r.tick < c.tick_start
            || r.tick > c.tick_end
            || r.token_id as usize >= expected.vocab_size
            || !r.loss_weight.is_finite()
            || r.loss_weight < 0.0
            || r.loss_mask > 1
            || (r.loss_weight > 0.0) != (r.loss_mask == 1)
        {
            return fail(format!(
                "row {}: tick, token ID or effective loss mask/weight invalid",
                r.index
            ));
        }
        if c.token_ids[i] != r.token_id
            || c.labels[i]
                != if r.loss_mask == 1 {
                    i64::from(r.token_id)
                } else {
                    -100
                }
            || c.loss_mask[i] != r.loss_mask
            || c.loss_weights[i] != r.loss_weight
        {
            return fail(format!(
                "row {}: redundant arrays disagree; targets are unshifted",
                r.index
            ));
        }
    }
    Ok(())
}

fn valid_uuid(s: &str) -> bool {
    s.len() == 36
        && s.bytes().enumerate().all(|(i, b)| {
            if [8, 13, 18, 23].contains(&i) {
                b == b'-'
            } else {
                b.is_ascii_hexdigit()
            }
        })
}

fn validate_rows(rows: &[Row], c: &Chunk) -> Result<(), Error> {
    let lives: BTreeMap<_, _> = c.windows.iter().map(|w| (w.id.as_str(), w)).collect();
    let mut opened = BTreeSet::new();
    let mut closed = BTreeSet::new();
    let mut positions: BTreeMap<&str, usize> = BTreeMap::new();
    let mut removals: BTreeMap<(u64, u64), Vec<usize>> = BTreeMap::new();
    let mut active = vec![false; rows.len()];
    for r in rows {
        if let Some(at) = &r.removed_at {
            if !r.durable
                || matches!(r.op.as_str(), "open" | "header")
                || at.tick < r.tick
                || at.tick >= 100000
                || (at.tick == r.tick && (r.phase != Phase::B || at.input_rank <= r.input_rank))
            {
                return fail(format!("row {}: invalid replacement boundary", r.index));
            }
            removals
                .entry((at.tick, at.input_rank))
                .or_default()
                .push(r.index);
        }
    }
    let mut decisions = BTreeSet::new();
    let mut previous: Option<&Row> = None;
    for (index, r) in rows.iter().enumerate() {
        if r.index != index {
            return fail("missing/reordered row indices");
        }
        let Some(w) = lives.get(r.window.as_str()) else {
            return fail(format!("row {index}: unknown window"));
        };
        if r.tick < w.open
            || w.close.is_some_and(|t| r.tick > t)
            || closed.contains(r.window.as_str())
        {
            return fail(format!("row {index}: write outside window lifetime"));
        }
        if let Some(p) = previous
            && ((r.tick, r.phase) < (p.tick, p.phase)
                || (r.tick == p.tick
                    && r.phase == Phase::B
                    && p.phase == Phase::B
                    && r.input_rank <= p.input_rank))
        {
            return fail("rows violate tick/phase/external input_rank ordering");
        }
        previous = Some(r);
        // A is atomic; replacements are applied only at their B boundary or D.
        while let Some((&(tick, rank), _)) = removals.first_key_value() {
            let due = tick < r.tick
                || (tick == r.tick
                    && (r.phase == Phase::D || (r.phase == Phase::B && rank <= r.input_rank)));
            if !due {
                break;
            }
            let (_, removed) = removals.pop_first().expect("entry just inspected");
            for old in removed {
                if old >= index || !active[old] {
                    return fail("replacement removes an unavailable row");
                }
                active[old] = false;
                let count = positions.entry(&rows[old].window).or_default();
                *count = count
                    .checked_sub(1)
                    .ok_or_else(|| Error::Contract("invalid replacement position".into()))?;
            }
        }
        if r.op == "open" {
            if r.phase != Phase::A
                || r.tick != w.open
                || r.token_id != c.special_tokens.bos
                || !opened.insert(r.window.as_str())
                || !r.durable
                || r.loss_mask != 0
            {
                return fail("invalid or duplicate BOS/open row");
            }
        } else if !opened.contains(r.window.as_str()) {
            return fail("row precedes BOS opening");
        }
        if r.phase == Phase::A
            && (r.tick != w.open
                || !matches!(r.op.as_str(), "open" | "header")
                || !r.durable
                || r.loss_mask != 0)
        {
            return fail("phase A must contain only masked durable BOS/headers");
        }
        if r.phase == Phase::B && r.op != "append" {
            return fail("phase B must contain external append rows");
        }
        if r.phase == Phase::D
            && (!matches!(r.op.as_str(), "generate" | "noop")
                || !decisions.insert((r.tick, r.window.as_str())))
        {
            return fail("multiple or invalid same-tick decisions for one window");
        }
        if r.durable == (r.token_id == c.special_tokens.noop)
            || (r.op == "noop" && r.token_id != c.special_tokens.noop)
        {
            return fail("NOOP must be non-durable; other tokens must be durable");
        }
        let position = positions.entry(&r.window).or_default();
        if r.position != *position {
            return fail(format!(
                "row {index}: local position {}, expected {position}",
                r.position
            ));
        }
        if r.durable {
            *position += 1;
            active[index] = true;
        }
        if r.phase == Phase::D && r.token_id == c.special_tokens.eos {
            if w.close != Some(r.tick) {
                return fail("EOS must match window closure tick");
            }
            closed.insert(r.window.as_str());
        }
    }
    if opened.len() != lives.len()
        || c.windows
            .iter()
            .any(|w| w.close.is_some() && !closed.contains(w.id.as_str()))
    {
        return fail("missing window open or EOS closure");
    }
    // Empty replacement appends can invalidate context without producing a row.
    // Remaining future removals are allowed only within the exported horizon.
    if removals
        .keys()
        .any(|(tick, _)| *tick > rows.last().map_or(0, |r| r.tick))
    {
        return fail("replacement lies beyond sample horizon");
    }
    Ok(())
}

fn fail<T>(message: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(message.into()))
}

/// Dataset errors retain line/sample/row context whenever available.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// File or stream read error.
    #[error("dataset I/O: {0}")]
    Io(#[from] std::io::Error),
    /// Malformed or unsupported export contract.
    #[error("{0}")]
    Contract(String),
}
