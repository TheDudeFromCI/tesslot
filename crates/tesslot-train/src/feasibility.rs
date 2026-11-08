//! Allocation-free Qwen3 parameter/state accounting. This does not execute Qwen.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tesslot_model::qwen;

fn add(a: u64, b: u64) -> Result<u64, String> {
    a.checked_add(b)
        .ok_or_else(|| "memory accounting overflow".into())
}
fn mul(a: u64, b: u64) -> Result<u64, String> {
    a.checked_mul(b)
        .ok_or_else(|| "memory accounting overflow".into())
}

/// Explicit planning precision, independent of the currently implemented FP32 trainer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    /// FP32 parameters and gradients, without a separate master copy.
    F32,
    /// Hypothetical BF16 parameters, FP32 gradients and separate FP32 master.
    Bf16,
    /// Hypothetical FP16 parameters, FP32 gradients and separate FP32 master.
    F16,
}
/// Persistent optimizer state counted by the planning model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Optimizer {
    /// Plain SGD, no momentum state.
    Sgd,
    /// Two FP32 moments for every trainable parameter.
    AdamW,
}
/// Explicit device budget; capacities are supplied assumptions, never pooled VRAM.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Device {
    /// Unique logical label, not a claim of discovered hardware.
    pub id: String,
    /// Budget after any external reservations, in bytes.
    pub budget_bytes: u64,
    /// Additional trainable Kidra parameters on this device; no converter is implied.
    pub additional_parameters: u64,
    /// Estimated peak saved/recomputed activation bytes, or null if unknown.
    pub activation_bytes: Option<u64>,
    /// Estimated temporary kernel/transfer buffers, or null if unknown.
    pub temporary_bytes: Option<u64>,
    /// Extra device checkpoint staging bytes, or null if unknown.
    pub checkpoint_staging_bytes: Option<u64>,
    /// Explicit unallocated safety reserve.
    pub margin_bytes: u64,
}
/// Proposed layer placement for accounting only, not an executable layer scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerRange {
    /// First layer, inclusive.
    pub start: usize,
    /// Last layer, exclusive.
    pub end: usize,
    /// Declared device label.
    pub device: String,
}
/// Explicit planning scenario. All parameters are trainable, including tied embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    /// Parameter storage precision assumed for a future backend.
    pub precision: Precision,
    /// Optimizer persistent state assumption.
    pub optimizer: Optimizer,
    /// Per-device budgets and workload allowances.
    pub devices: Vec<Device>,
    /// Complete nonoverlapping local-layer coverage.
    pub layers: Vec<LayerRange>,
    /// Embedding table owner; also owns the output projection when weights are tied.
    pub embedding_device: String,
    /// Output projection/final norm compute device. Tied weights require the same owner.
    pub output_device: String,
}
/// Per-device estimated bytes. Optional overheads remain null when unknown.
#[derive(Debug, Serialize)]
pub struct DeviceReport {
    /// Logical scenario device label.
    pub id: String,
    /// Supplied budget.
    pub budget_bytes: u64,
    /// Trainable scalar count assigned to this device.
    pub parameters: u64,
    /// Parameter storage bytes.
    pub weight_bytes: u64,
    /// FP32 gradient bytes.
    pub gradient_bytes: u64,
    /// Separate FP32 master copy for reduced precision, zero for FP32.
    pub master_bytes: u64,
    /// Optimizer moment bytes.
    pub optimizer_bytes: u64,
    /// Sum of weights, gradients, master copy and moments.
    pub persistent_bytes: u64,
    /// Assumed activation bytes; not measured.
    pub activation_bytes: Option<u64>,
    /// Assumed temporary bytes; not measured.
    pub temporary_bytes: Option<u64>,
    /// Assumed checkpoint staging bytes; not measured.
    pub checkpoint_staging_bytes: Option<u64>,
    /// Supplied margin, in bytes.
    pub margin_bytes: u64,
    /// Persistent bytes plus known allowances and margin; incomplete when any is null.
    pub known_required_bytes: u64,
    /// Total estimate only if every workload allowance was supplied.
    pub estimated_required_bytes: Option<u64>,
    /// insufficient_budget, incomplete_estimate, or estimate_within_budget_unverified.
    pub status: String,
}
/// Auditable preflight report, never a measured allocation or training result.
#[derive(Debug, Serialize)]
pub struct Report {
    /// Scope of the calculation; excludes current CPU host/interchange/candidate copies.
    pub accounting_scope: String,
    /// Validated source configuration used to derive the parameter manifest.
    pub model_config: qwen::Config,
    /// Independently versioned report format.
    pub format: String,
    /// Base Qwen3 parameter count; tied source parameters counted once.
    pub base_parameters: u64,
    /// Explicit additional Kidra parameter assumption, not inferred conversion.
    pub additional_parameters: u64,
    /// All trainable parameters in this scenario.
    pub total_parameters: u64,
    /// Estimated FP32 model plus optimizer/master tensor payload on checkpoint storage.
    /// Excludes JSON encoding, manifests, staging copies and filesystem overhead.
    pub checkpoint_tensor_payload_bytes: u64,
    /// Estimated per-device memory without sharing budgets.
    pub devices: Vec<DeviceReport>,
    /// False if any known per-device requirement exceeds its budget.
    pub budgets_sufficient_for_known_bytes: bool,
    /// False while any workload allowance is missing.
    pub estimate_complete: bool,
    /// Always false until model execution, conversion and hardware gates are implemented.
    pub training_ready: bool,
    /// Concrete unresolved execution requirements, independent of budget arithmetic.
    pub blockers: Vec<String>,
    /// Precision, optimizer, placement and overhead assumptions used in this report.
    pub scenario: Scenario,
}

/// Enumerate unique source tensors for supported Qwen3 shapes; no tensor allocation.
/// The manifest names HF norms/projections, including Q/K head-dimension norms.
pub fn parameter_shapes(config: &qwen::Config) -> Result<BTreeMap<String, Vec<usize>>, String> {
    config.validate()?;
    let h = config.hidden_size;
    let q = config.query_width()?;
    let kv = config.kv_width()?;
    let mut shapes = BTreeMap::from([
        (
            "model.embed_tokens.weight".into(),
            vec![config.vocab_size, h],
        ),
        ("model.norm.weight".into(), vec![h]),
    ]);
    if !config.tie_word_embeddings {
        shapes.insert("lm_head.weight".into(), vec![config.vocab_size, h]);
    }
    // Keep hostile dimensions from generating an unbounded metadata map.
    if config.num_hidden_layers > 1024 {
        return Err("preflight supports at most 1024 layers".into());
    }
    for layer in 0..config.num_hidden_layers {
        for (suffix, shape) in [
            ("input_layernorm.weight", vec![h]),
            ("post_attention_layernorm.weight", vec![h]),
            ("self_attn.q_proj.weight", vec![q, h]),
            ("self_attn.k_proj.weight", vec![kv, h]),
            ("self_attn.v_proj.weight", vec![kv, h]),
            ("self_attn.o_proj.weight", vec![h, q]),
            ("self_attn.q_norm.weight", vec![config.head_dim]),
            ("self_attn.k_norm.weight", vec![config.head_dim]),
            ("mlp.up_proj.weight", vec![config.intermediate_size, h]),
            ("mlp.gate_proj.weight", vec![config.intermediate_size, h]),
            ("mlp.down_proj.weight", vec![h, config.intermediate_size]),
        ] {
            shapes.insert(format!("model.layers.{layer}.{suffix}"), shape);
        }
    }
    Ok(shapes)
}

/// Count unique parameters and persistent state, reject overflow/invalid placements.
/// Unknown activation/buffer costs prevent a complete memory estimate.
pub fn estimate(config: &qwen::Config, scenario: Scenario) -> Result<Report, String> {
    let shapes = parameter_shapes(config)?;
    let mut counts = BTreeMap::new();
    for device in &scenario.devices {
        if device.id.is_empty()
            || device.budget_bytes == 0
            || counts
                .insert(device.id.clone(), device.additional_parameters)
                .is_some()
        {
            return Err("device labels must be unique/nonempty with positive budgets".into());
        }
    }
    if !counts.contains_key(&scenario.embedding_device)
        || !counts.contains_key(&scenario.output_device)
    {
        return Err("embedding/output placement references an undeclared device".into());
    }
    if config.tie_word_embeddings && scenario.embedding_device != scenario.output_device {
        return Err(
            "tied embeddings/output must share an owner; replication is not implemented".into(),
        );
    }
    let mut owners = vec![None; config.num_hidden_layers];
    for range in &scenario.layers {
        if range.start >= range.end
            || range.end > owners.len()
            || !counts.contains_key(&range.device)
        {
            return Err("invalid layer range or undeclared device".into());
        }
        for owner in &mut owners[range.start..range.end] {
            if owner.replace(range.device.as_str()).is_some() {
                return Err("overlapping layer placement".into());
            }
        }
    }
    if owners.contains(&None) {
        return Err("layer placement has gaps".into());
    }
    let mut base = 0;
    for (name, shape) in shapes {
        let size = shape.into_iter().try_fold(1, |n, d| mul(n, d as u64))?;
        base = add(base, size)?;
        let device = if let Some(rest) = name.strip_prefix("model.layers.") {
            let index: usize = rest
                .split('.')
                .next()
                .unwrap()
                .parse()
                .map_err(|_| "invalid layer tensor name")?;
            owners[index].unwrap()
        } else if name == "model.embed_tokens.weight" {
            scenario.embedding_device.as_str()
        } else {
            scenario.output_device.as_str()
        };
        let count = counts.get_mut(device).unwrap();
        *count = add(*count, size)?;
    }
    let weight_width = if matches!(scenario.precision, Precision::F32) {
        4
    } else {
        2
    };
    let master_width = if weight_width == 4 { 0 } else { 4 };
    let optimizer_width = if matches!(scenario.optimizer, Optimizer::AdamW) {
        8
    } else {
        0
    };
    let mut reports = Vec::new();
    let mut additional = 0;
    let mut total = 0;
    for device in &scenario.devices {
        additional = add(additional, device.additional_parameters)?;
        let n = counts[&device.id];
        total = add(total, n)?;
        let persistent = mul(n, weight_width + 4 + master_width + optimizer_width)?;
        let mut known = add(persistent, device.margin_bytes)?;
        for overhead in [
            device.activation_bytes,
            device.temporary_bytes,
            device.checkpoint_staging_bytes,
        ]
        .into_iter()
        .flatten()
        {
            known = add(known, overhead)?;
        }
        let complete = device.activation_bytes.is_some()
            && device.temporary_bytes.is_some()
            && device.checkpoint_staging_bytes.is_some();
        let status = if known > device.budget_bytes {
            "insufficient_budget"
        } else if !complete {
            "incomplete_estimate"
        } else {
            "estimate_within_budget_unverified"
        };
        reports.push(DeviceReport {
            id: device.id.clone(),
            budget_bytes: device.budget_bytes,
            parameters: n,
            weight_bytes: mul(n, weight_width)?,
            gradient_bytes: mul(n, 4)?,
            master_bytes: mul(n, master_width)?,
            optimizer_bytes: mul(n, optimizer_width)?,
            persistent_bytes: persistent,
            activation_bytes: device.activation_bytes,
            temporary_bytes: device.temporary_bytes,
            checkpoint_staging_bytes: device.checkpoint_staging_bytes,
            margin_bytes: device.margin_bytes,
            known_required_bytes: known,
            estimated_required_bytes: complete.then_some(known),
            status: status.into(),
        });
    }
    Ok(Report {
        accounting_scope:
            "hypothetical_device_resident_full_parameter_state; not_current_cpu_trainer_peak_memory"
                .into(),
        model_config: config.clone(),
        format: "tesslot-feasibility/1".into(),
        base_parameters: base,
        additional_parameters: additional,
        total_parameters: total,
        checkpoint_tensor_payload_bytes: mul(total, 4 + optimizer_width + master_width)?,
        budgets_sufficient_for_known_bytes: reports
            .iter()
            .all(|d| d.known_required_bytes <= d.budget_bytes),
        estimate_complete: reports.iter().all(|d| d.estimated_required_bytes.is_some()),
        devices: reports,
        training_ready: false,
        blockers: vec![
            "pretrained_qwen_execution_and_kidra_conversion_unimplemented".into(),
            "model_level_cuda_and_precision_gates_unverified".into(),
            "large_model_checkpoint_and_device_resume_unimplemented".into(),
        ],
        scenario,
    })
}
