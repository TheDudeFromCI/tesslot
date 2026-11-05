//! Backend-independent replay semantics, configuration and placement contracts.

use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, net::SocketAddr};

/// Kidra architecture implemented by the tiny correctness oracle.
pub const ARCHITECTURE_VERSION: &str = "kidra-reference/2";
/// Native public protocol, independent of the application version.
pub const API_VERSION: &str = "window-v1";

/// A phase in an external-first tick. Prediction is computed before D commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Phase {
    /// Atomic window initialization.
    A,
    /// Ordered external observations.
    B,
    /// Parallel model decisions.
    D,
}

/// Lifetime and embedding slot of a logical window. IDs cannot be reused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Window {
    /// Logical identity, independent of array order and physical placement.
    pub id: String,
    /// Numeric identity embedding index from the exporter.
    pub slot: usize,
    /// Purpose of the window.
    #[serde(rename = "type")]
    pub kind: String,
    /// Inclusive opening tick.
    pub open: u64,
    /// Inclusive final tick; evicted after D.
    pub close: Option<u64>,
    /// Resolved automatic header text.
    pub header: String,
}

/// External replacement invalidates an old memory row at this B boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Removal {
    /// Replacement tick.
    pub tick: u64,
    /// First replacement insertion rank.
    pub input_rank: u64,
}

/// An unshifted observation or prediction; loss and memory membership are separate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    /// Sample-global contiguous row index.
    pub index: usize,
    /// Logical tick, unrelated to wall time.
    pub tick: u64,
    /// External-first phase.
    pub phase: Phase,
    /// Logical window ID.
    pub window: String,
    /// Observation ID or unshifted prediction target.
    pub token_id: u32,
    /// Window-local insertion position. NOOP does not advance it.
    pub position: usize,
    /// Only durable rows enter memory.
    pub durable: bool,
    /// Export uses integer 0/1, independently of visibility.
    pub loss_mask: u8,
    /// Effective nonnegative target weight, mask already applied.
    pub loss_weight: f64,
    /// Exporter operation (open/header/append/generate/noop).
    pub op: String,
    /// Source replay event path, or implicit.
    pub event: String,
    /// Ordering of external tokens within a tick.
    pub input_rank: u64,
    /// Optional replacement boundary; retains historical visibility.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub removed_at: Option<Removal>,
}

/// Native distinct control IDs; never inferred from a base model's BOS/EOS.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialTokens {
    /// Beginning of window.
    pub bos: u32,
    /// Close after the shared D commit.
    pub eos: u32,
    /// Non-durable idle prediction.
    pub noop: u32,
}

impl SpecialTokens {
    /// Check dedicated, distinct, vocabulary-bounded controls.
    pub fn validate(&self, vocab_size: usize) -> Result<(), Error> {
        let ids = [self.bos, self.eos, self.noop];
        if ids.iter().any(|&id| id as usize >= vocab_size)
            || self.bos == self.eos
            || self.bos == self.noop
            || self.eos == self.noop
        {
            return Err(Error::Invalid(
                "BOS/EOS/NOOP must be distinct IDs inside the vocabulary".into(),
            ));
        }
        Ok(())
    }
}

/// Exact port of Trajectory Remix compiler.visible_to; true means allowed.
///
/// Apply at every layer, to memory as well as queries. Unknown windows fail closed.
pub fn visible_to(query: &Row, key: &Row, windows: &[Window]) -> bool {
    if !key.durable {
        return false;
    }
    if let Some(removed) = &key.removed_at
        && (query.tick > removed.tick
            || (query.tick == removed.tick
                && (query.phase == Phase::D
                    || (query.phase == Phase::B && query.input_rank >= removed.input_rank))))
    {
        return false;
    }
    let Some(life) = windows.iter().find(|w| w.id == key.window) else {
        return false;
    };
    if life.open > query.tick || life.close.is_some_and(|close| close < query.tick) {
        return false;
    }
    if key.tick < query.tick {
        return true;
    }
    if key.tick > query.tick {
        return false;
    }
    match query.phase {
        Phase::D => matches!(key.phase, Phase::A | Phase::B),
        Phase::B => {
            key.phase == Phase::A || (key.phase == Phase::B && key.input_rank < query.input_rank)
        }
        Phase::A => false,
    }
}

/// Tiny model configuration, deliberately separate from pretrained Qwen3 config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    /// Token embedding table size.
    pub vocab_size: usize,
    /// One reference attention head, even dimension for RoPE.
    pub hidden_size: usize,
    /// Local attention then global attention in every layer.
    pub layers: usize,
    /// Configurable embedding slot capacity.
    pub max_windows: usize,
    /// Window identity embedding ablation.
    pub window_embeddings: bool,
    /// Global cross-window attention ablation.
    pub global_attention: bool,
    /// Window-local RoPE frequency base.
    pub rope_theta: f64,
    /// Reproducible parameter seed.
    pub seed: u64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 128,
            hidden_size: 8,
            layers: 2,
            max_windows: 8,
            window_embeddings: true,
            global_attention: true,
            rope_theta: 10000.0,
            seed: 42,
        }
    }
}

impl ModelConfig {
    /// Bound oracle allocations and reject unsupported dimensions before execution.
    pub fn validate(&self) -> Result<(), Error> {
        if !(4..=65536).contains(&self.vocab_size)
            || !(2..=128).contains(&self.hidden_size)
            || !self.hidden_size.is_multiple_of(2)
            || !(1..=16).contains(&self.layers)
            || !(1..=4096).contains(&self.max_windows)
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 1.0
        {
            return Err(Error::Invalid("reference model requires vocabulary 4..65536, even hidden size 2..128, layers 1..16, window slots 1..4096, finite RoPE theta > 1".into()));
        }
        Ok(())
    }
}

/// Logical device selector, independent of a tensor library.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Device {
    /// CPU correctness backend.
    Cpu,
    /// NVIDIA ordinal; not available in milestone one.
    Cuda {
        /// CUDA device ordinal.
        ordinal: u32,
    },
}

/// Explicit layer groups with distinct memory budgets. Budgets are not pooled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerPlacement {
    /// First layer, inclusive.
    pub start: usize,
    /// End layer, exclusive.
    pub end: usize,
    /// Execution device.
    pub device: Device,
    /// Operator-defined capacity ceiling in bytes.
    pub memory_budget_bytes: u64,
}

/// Server and reference configuration. No secret is stored here.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Localhost by default; remote serving is intentionally disabled until scoped auth exists.
    pub bind: SocketAddr,
    /// Reference model settings.
    pub model: ModelConfig,
    /// Ordered contiguous layer partition.
    pub placement: Vec<LayerPlacement>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8080".parse().expect("literal address"),
            model: ModelConfig::default(),
            placement: vec![LayerPlacement {
                start: 0,
                end: 2,
                device: Device::Cpu,
                memory_budget_bytes: 1 << 30,
            }],
        }
    }
}

impl Config {
    /// Deserialize JSON and validate all execution constraints.
    pub fn from_json(bytes: &[u8]) -> Result<Self, Error> {
        let config: Self = serde_json::from_slice(bytes)?;
        config.validate()?;
        Ok(config)
    }
    /// Reject unsupported hardware explicitly, without silent CPU fallback.
    pub fn validate(&self) -> Result<(), Error> {
        self.model.validate()?;
        if !self.bind.ip().is_loopback() {
            return Err(Error::Invalid("remote binding requires scoped authentication, unavailable in milestone 1; bind a loopback address".into()));
        }
        let mut next = 0;
        for group in &self.placement {
            if group.start != next
                || group.end <= group.start
                || group.end > self.model.layers
                || group.memory_budget_bytes == 0
            {
                return Err(Error::Invalid(
                    "placement must cover every layer exactly once with nonzero budgets".into(),
                ));
            }
            if group.device != Device::Cpu {
                return Err(Error::Invalid(
                    "CUDA execution unavailable: use CPU until backend capability gates pass"
                        .into(),
                ));
            }
            next = group.end;
        }
        if next != self.model.layers {
            return Err(Error::Invalid("placement does not cover all layers".into()));
        }
        Ok(())
    }
}

/// Explicit subsystem availability for API and dashboard clients.
pub fn capabilities() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([
        ("cpu_reference", "available"),
        ("dataset_validation", "available"),
        ("inference", "unimplemented"),
        ("training", "unimplemented"),
        ("agents", "unimplemented"),
        ("evaluation", "unimplemented"),
        ("cuda", "unavailable"),
        ("tools", "unimplemented"),
    ])
}

/// Actionable errors for operator-controlled inputs.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Unsupported or inconsistent configuration.
    #[error("{0}")]
    Invalid(String),
    /// Malformed configuration JSON.
    #[error("configuration JSON: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn config_security_and_placement() {
        Config::default().validate().unwrap();
        assert!(Config::from_json(br#"{"bind":"0.0.0.0:8080"}"#).is_err());
        assert!(Config::from_json(br#"{"model":{"hidden_size":3}}"#).is_err());
        assert!(Config::from_json(br#"{"typo":true}"#).is_err());
        let mut c = Config::default();
        c.placement[0].device = Device::Cuda { ordinal: 0 };
        assert!(c.validate().is_err());
        c.placement[0].device = Device::Cpu;
        c.placement[0].start = 1;
        assert!(c.validate().is_err());
    }
}
