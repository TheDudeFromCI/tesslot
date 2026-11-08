//! Explicit device placement and compute precision; parameters and optimizer masters stay FP32.
use super::*;
/// Physical execution target. No implicit CPU offload or unified GPU memory.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Target {
    /// Portable reference backend.
    Cpu,
    /// CUDA ordinal after CUDA_VISIBLE_DEVICES filtering.
    Cuda {
        /// Visible ordinal.
        index: usize,
    },
}
/// GEMM precision. Norms, residual streams, logits, gradients and optimizer masters use FP32.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    /// Strict FP32 baseline: native CPU GEMM, slower multiply/reduce reference on CUDA.
    #[default]
    F32,
    /// Permit CubeCL TF32 tensor-core GEMMs with FP32 storage/accumulation on CUDA.
    Tf32,
    /// NVIDIA BF16 compute (hardware gate required).
    Bf16,
    /// NVIDIA FP16 compute; static unscaled gradients, fail on overflow.
    F16,
}
/// Layer ownership is independent of window identity and checkpoint storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    /// Embedding, output and shared exchange parameter owner.
    pub root: Target,
    /// Exactly one device per transformer layer.
    pub layers: Vec<Target>,
    /// Matmul arithmetic policy; TF32/BF16/FP16 require CUDA throughout.
    /// CUDA FP32 is a slower numerical reference, not a production GEMM.
    pub precision: Precision,
    /// Explicit parameter storage budget, per unique physical target.
    /// Excludes activations/gradients/moments; use `estimate` before training.
    pub parameter_budget_bytes: u64,
}
impl Plan {
    /// Single CPU owner and full precision.
    pub fn cpu(layers: usize, parameter_budget_bytes: u64) -> Self {
        Self {
            root: Target::Cpu,
            layers: vec![Target::Cpu; layers],
            precision: Precision::F32,
            parameter_budget_bytes,
        }
    }
    pub(super) fn validate(&self, count: usize) -> Result<(), String> {
        if self.layers.len() != count || self.parameter_budget_bytes == 0 {
            return Err(
                "placement requires exactly one target per layer and a nonzero budget".into(),
            );
        }
        if std::iter::once(&self.root)
            .chain(&self.layers)
            .any(|t| matches!(t, Target::Cpu))
            && std::iter::once(&self.root)
                .chain(&self.layers)
                .any(|t| matches!(t, Target::Cuda { .. }))
        {
            return Err("mixed CPU/CUDA plans are unsupported: distributed backward requires one compute backend".into());
        }
        for target in std::iter::once(&self.root)
            .chain(&self.layers)
            .collect::<std::collections::BTreeSet<_>>()
        {
            if self.precision != Precision::F32 && *target == Target::Cpu {
                return Err("TF32/BF16/FP16 matmuls require CUDA; choose FP32 on CPU".into());
            }
            target.device(false)?;
        }
        Ok(())
    }
    fn devices(&self, training: bool) -> Result<BTreeMap<Target, Device>, String> {
        std::iter::once(&self.root)
            .chain(&self.layers)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|t| Ok((t.clone(), t.device(training)?)))
            .collect()
    }
}
impl Target {
    fn device(&self, training: bool) -> Result<Device, String> {
        let device = match self {
            Self::Cpu => Device::flex(),
            Self::Cuda { index } => {
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = index;
                    return Err(
                        "CUDA backend unavailable; compile tesslot-tensor with feature cuda".into(),
                    );
                }
                #[cfg(feature = "cuda")]
                {
                    let output = std::process::Command::new("nvidia-smi")
                        .args(["--query-gpu=index", "--format=csv,noheader"])
                        .output()
                        .map_err(|_| "NVIDIA discovery unavailable")?;
                    if !output.status.success() {
                        return Err("NVIDIA driver discovery failed".into());
                    }
                    let count = String::from_utf8_lossy(&output.stdout)
                        .lines()
                        .filter(|l| !l.trim().is_empty())
                        .count();
                    let visible = match std::env::var("CUDA_VISIBLE_DEVICES") {
                        Ok(v) if v.trim().is_empty() || v.trim() == "-1" => 0,
                        Ok(v) => count.min(v.split(',').count()),
                        Err(_) => count,
                    };
                    if *index >= visible {
                        return Err(format!(
                            "CUDA ordinal {index} unavailable ({visible} visible devices)"
                        ));
                    }
                    Device::cuda(*index)
                }
            }
        };
        Ok(if training { device.autodiff() } else { device })
    }
}
/// Persistent-state estimate; explicitly excludes sequence-dependent activations and allocator buffers.
#[derive(Debug, Serialize)]
pub struct Estimate {
    /// FP32 trainable weights.
    pub weights: u64,
    /// Full parameter gradients.
    pub gradients: u64,
    /// Two FP32 AdamW moments (zero for SGD/inference).
    pub optimizer: u64,
    /// Candidate weights and (for AdamW) two new moment arrays during transactional updates.
    pub update_temporaries: u64,
}
impl Model {
    /// Load already validated named tensors without tiny-demo dimensional caps.
    /// The caller supplies a memory budget and placement; no weights are downloaded.
    pub fn from_tensors(
        config: Config,
        weights: Tensors,
        windows: Option<usize>,
        training: bool,
        plan: Plan,
    ) -> Result<Self, String> {
        config.validate()?;
        plan.validate(config.num_hidden_layers)?;
        let expected = if windows.is_some() {
            echo::shapes(&config, windows)?
        } else {
            tesslot_artifacts::qwen::source_shapes(&config).map_err(|e| e.to_string())?
        };
        if weights.keys().ne(expected.keys())
            || weights.iter().any(|(n, (s, v))| {
                *s != expected[n]
                    || v.len() != s.iter().product::<usize>()
                    || v.iter().any(|x| !x.is_finite())
            })
        {
            return Err(
                "tensor names, shapes or finite payloads disagree with executable architecture"
                    .into(),
            );
        }
        let devices = plan.devices(training)?;
        let device = devices[&plan.root].clone();
        let mut model = Self {
            id: crate::NEXT_MODEL.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            config,
            parameters: BTreeMap::new(),
            device,
            training,
            checkpoint_identity: None,
            exchange_windows: windows,
            corpus: None,
            remix: None,
            adam: None,
            step: 0,
            learning_rate: 0.2,
            plan: plan.clone(),
        };
        model.check_budget(&expected, &plan)?;
        for (name, (shape, values)) in weights {
            let dims = if shape.len() == 1 {
                [1, shape[0]]
            } else {
                [shape[0], shape[1]]
            };
            let owner = devices[model.owner(&name, &plan)].clone();
            let p = T::from_data(TensorData::new(values, dims), (&owner, DType::F32));
            model
                .parameters
                .insert(name, if training { p.require_grad() } else { p });
        }
        Ok(model)
    }
    fn owner<'a>(&self, name: &str, plan: &'a Plan) -> &'a Target {
        name.strip_prefix("model.layers.")
            .and_then(|s| s.split('.').next())
            .and_then(|s| s.parse::<usize>().ok())
            .and_then(|l| plan.layers.get(l))
            .unwrap_or(&plan.root)
    }
    fn check_budget(
        &self,
        shapes: &BTreeMap<String, Vec<usize>>,
        plan: &Plan,
    ) -> Result<(), String> {
        let mut usage: BTreeMap<String, u64> = BTreeMap::new();
        for (name, shape) in shapes {
            let bytes = shape
                .iter()
                .try_fold(4u64, |n, d| n.checked_mul(*d as u64))
                .ok_or("weight size overflow")?;
            let owner = format!("{:?}", self.owner(name, plan));
            let total = usage.entry(owner).or_default();
            *total = total.checked_add(bytes).ok_or("weight budget overflow")?;
            if *total > plan.parameter_budget_bytes {
                return Err("per-device FP32 parameter budget exceeded".into());
            }
        }
        Ok(())
    }
    /// Relocate masters and optimizer moments together. Invalidate existing caches.
    /// Call only between optimizer updates; no graph is retained by the model itself.
    pub fn configure_runtime(&mut self, plan: Plan) -> Result<(), String> {
        plan.validate(self.config.num_hidden_layers)?;
        self.check_budget(&echo::shapes(&self.config, self.exchange_windows)?, &plan)?;
        let devices = plan.devices(self.training)?;
        let owners = self
            .parameters
            .keys()
            .map(|n| (n.clone(), devices[self.owner(n, &plan)].clone()))
            .collect::<BTreeMap<_, _>>();
        self.parameters = self
            .parameters
            .iter()
            .map(|(n, p)| {
                let p = p.clone().to_device(&owners[n]).detach();
                (n.clone(), if self.training { p.require_grad() } else { p })
            })
            .collect();
        if let Some(adam) = &mut self.adam {
            for (n, (m, v)) in &mut adam.moments {
                *m = m.clone().to_device(&owners[n]);
                *v = v.clone().to_device(&owners[n]);
            }
        }
        self.device = devices[&plan.root].clone();
        self.plan = plan;
        self.id = crate::NEXT_MODEL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
    /// Estimate persistent training state before execution, without pretending activations are known.
    pub fn estimate(&self) -> Estimate {
        let weights = self
            .parameters
            .values()
            .map(|p| p.dims().iter().product::<usize>() as u64 * 4)
            .sum::<u64>();
        Estimate {
            weights,
            gradients: if self.training { weights } else { 0 },
            optimizer: if self.adam.is_some() { 2 * weights } else { 0 },
            update_temporaries: if self.training {
                if self.adam.is_some() {
                    3 * weights
                } else {
                    weights
                }
            } else {
                0
            },
        }
    }
    pub(super) fn layer_device(&self, layer: usize) -> Device {
        self.parameters[&format!("model.layers.{layer}.input_layernorm.weight")].device()
    }
    pub(super) fn matmul(&self, x: T, y: T) -> T {
        match self.plan.precision {
            Precision::F32 if self.plan.root == Target::Cpu => x.matmul(y),
            Precision::F32 => crate::matmul::reference_f32(x, y)
                .expect("validated FP32 model operands and placement"),
            Precision::Tf32 => x.matmul(y),
            Precision::Bf16 => x
                .cast(DType::BF16)
                .matmul(y.cast(DType::BF16))
                .cast(DType::F32),
            Precision::F16 => x
                .cast(DType::F16)
                .matmul(y.cast(DType::F16))
                .cast(DType::F32),
        }
    }
}
