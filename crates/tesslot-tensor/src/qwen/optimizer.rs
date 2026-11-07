//! Tiny FP32 AdamW with explicit state; SGD remains the legacy experiment default.
use super::*;
/// Versioned optimizer hyperparameters. All moments remain FP32 on the CPU backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdamWConfig {
    /// First-moment decay in [0,1).
    pub beta1: f32,
    /// Second-moment decay in [0,1).
    pub beta2: f32,
    /// Positive denominator stabilizer.
    pub epsilon: f32,
    /// Decoupled weight decay, applied to every eligible parameter.
    pub weight_decay: f32,
}
impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        }
    }
}
impl AdamWConfig {
    /// Validate before configuring or restoring optimizer state.
    pub fn validate(&self) -> Result<(), String> {
        if ![self.beta1, self.beta2, self.epsilon, self.weight_decay]
            .iter()
            .all(|v| v.is_finite())
            || !(0.0..1.0).contains(&self.beta1)
            || !(0.0..1.0).contains(&self.beta2)
            || self.epsilon <= 0.0
            || self.weight_decay < 0.0
        {
            return Err("invalid AdamW beta/epsilon/decay".into());
        }
        Ok(())
    }
}
pub(super) struct Adam {
    pub config: AdamWConfig,
    pub updates: u64,
    pub moments: BTreeMap<String, (T, T)>,
}
impl Adam {
    pub fn encode(
        &self,
        shapes: &BTreeMap<String, Vec<usize>>,
    ) -> Result<(Vec<u8>, Checkpoint), String> {
        let mut payload = Tensors::new();
        for (name, (m, v)) in &self.moments {
            for (prefix, tensor) in [("first", m), ("second", v)] {
                payload.insert(
                    format!("{prefix}.{name}"),
                    (
                        shapes[name].clone(),
                        tensor
                            .clone()
                            .into_data()
                            .try_to_vec::<f32>()
                            .map_err(|e| e.to_string())?,
                    ),
                );
            }
        }
        let bytes =
            encode_f32_with_metadata(&payload, BTreeMap::new()).map_err(|e| e.to_string())?;
        let manifest = Checkpoint {
            config: self.config.clone(),
            updates: self.updates,
            sha256: format!("{:x}", Sha256::digest(&bytes)),
        };
        Ok((bytes, manifest))
    }
    pub fn restore(
        manifest: Checkpoint,
        bytes: Vec<u8>,
        shapes: BTreeMap<String, Vec<usize>>,
        device: &Device,
        step: u64,
        learning_rate: f32,
    ) -> Result<Self, String> {
        manifest.config.validate()?;
        if manifest.updates != step || learning_rate * manifest.config.weight_decay >= 1.0 {
            return Err("incompatible AdamW update count or decay".into());
        }
        if format!("{:x}", Sha256::digest(&bytes)) != manifest.sha256 {
            return Err("optimizer checkpoint checksum mismatch".into());
        }
        if inspect(&mut Cursor::new(&bytes))
            .map_err(|e| e.to_string())?
            .tensors
            .values()
            .any(|tensor| tensor.dtype != "F32")
        {
            return Err("adamw-f32/1 requires FP32 optimizer moments".into());
        }
        let expected = if manifest.updates == 0 {
            BTreeMap::new()
        } else {
            shapes
                .iter()
                .flat_map(|(name, shape)| {
                    ["first", "second"].map(|p| (format!("{p}.{name}"), shape.clone()))
                })
                .collect::<BTreeMap<_, _>>()
        };
        let sources = expected
            .keys()
            .map(|name| (name.clone(), name.clone()))
            .collect();
        let payload = import(
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
        let mut moments = BTreeMap::new();
        if manifest.updates > 0 {
            for (name, shape) in shapes {
                let tensor = |prefix: &str| -> Result<T, String> {
                    let values = payload[&format!("{prefix}.{name}")].1.clone();
                    if prefix == "second" && values.iter().any(|v| *v < 0.0) {
                        return Err("negative AdamW second moment".into());
                    }
                    let dims = if shape.len() == 1 {
                        [1, shape[0]]
                    } else {
                        [shape[0], shape[1]]
                    };
                    Ok(T::from_data(
                        TensorData::new(values, dims),
                        (device, DType::F32),
                    ))
                };
                let pair = (tensor("first")?, tensor("second")?);
                moments.insert(name, pair);
            }
        }
        Ok(Self {
            config: manifest.config,
            updates: manifest.updates,
            moments,
        })
    }
    pub fn next(&self, gradients: &BTreeMap<String, T>, clip: f32) -> Result<Self, String> {
        let mut moments = BTreeMap::new();
        for (name, raw) in gradients {
            let g = raw.clone().detach() * clip;
            let zero = g.clone().zeros_like();
            let (m, v) = self
                .moments
                .get(name)
                .cloned()
                .unwrap_or((zero.clone(), zero));
            let m = (m * self.config.beta1 + g.clone() * (1.0 - self.config.beta1)).detach();
            let v =
                (v * self.config.beta2 + g.powf_scalar(2.0) * (1.0 - self.config.beta2)).detach();
            for tensor in [&m, &v] {
                if tensor
                    .clone()
                    .into_data()
                    .try_to_vec::<f32>()
                    .map_err(|e| e.to_string())?
                    .iter()
                    .any(|x| !x.is_finite())
                {
                    return Err("nonfinite AdamW moment".into());
                }
            }
            moments.insert(name.clone(), (m, v));
        }
        Ok(Self {
            config: self.config.clone(),
            updates: self.updates.checked_add(1).ok_or("AdamW update overflow")?,
            moments,
        })
    }
    pub fn parameter(&self, name: &str, p: T, lr: f32) -> T {
        let (m, v) = &self.moments[name];
        let m = m.clone() / (1.0 - self.config.beta1.powf(self.updates as f32));
        let v = v.clone() / (1.0 - self.config.beta2.powf(self.updates as f32));
        p * (1.0 - lr * self.config.weight_decay) - m / (v.sqrt() + self.config.epsilon) * lr
    }
}
/// Manifest portion paired with the independently checksummed moment file.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Checkpoint {
    pub config: AdamWConfig,
    pub updates: u64,
    pub sha256: String,
}
impl Model {
    /// Select AdamW for a registered corpus before its first update; never reset moments silently.
    pub fn enable_adamw(&mut self, config: AdamWConfig, learning_rate: f32) -> Result<(), String> {
        config.validate()?;
        if self.corpus.is_none()
            || self.step != 0
            || self.adam.is_some()
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
            || learning_rate * config.weight_decay >= 1.0
        {
            return Err("configure AdamW with a valid learning rate for an echo corpus before the first model update".into());
        }
        self.adam = Some(Adam {
            config,
            updates: 0,
            moments: BTreeMap::new(),
        });
        self.learning_rate = learning_rate;
        Ok(())
    }
    /// Actual optimizer identity, independent of model architecture.
    pub fn optimizer_id(&self) -> &'static str {
        if self.adam.is_some() {
            "adamw-f32/1"
        } else {
            "sgd-no-momentum/1"
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adamw_matches_known_two_step_vector_and_decoupled_decay() {
        let device = Device::flex();
        let t = |v| T::from_data(TensorData::new(v, [1, 2]), (&device, DType::F32));
        let adam = Adam {
            config: AdamWConfig {
                weight_decay: 0.0,
                ..Default::default()
            },
            updates: 0,
            moments: BTreeMap::new(),
        };
        let a = adam
            .next(&BTreeMap::from([("p".into(), t(vec![2.0, -4.0]))]), 1.0)
            .unwrap();
        let p = a.parameter("p", t(vec![1.0, -2.0]), 0.01);
        let b = a
            .next(&BTreeMap::from([("p".into(), t(vec![-2.0, -4.0]))]), 1.0)
            .unwrap();
        let values = b
            .parameter("p", p, 0.01)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!((values[0] - 0.9905263).abs() < 1e-6);
        assert!((values[1] + 1.98).abs() < 1e-6);
        let decay = Adam {
            config: AdamWConfig {
                weight_decay: 0.1,
                ..Default::default()
            },
            updates: 0,
            moments: BTreeMap::new(),
        }
        .next(&BTreeMap::from([("p".into(), t(vec![0.0, 0.0]))]), 1.0)
        .unwrap();
        let values = decay
            .parameter("p", t(vec![1.0, -2.0]), 0.01)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!((values[0] - 0.999).abs() < 1e-7);
        assert!((values[1] + 1.998).abs() < 1e-7);
    }
    #[test]
    fn adamw_rejects_negative_second_moment_with_valid_checksum() {
        let device = Device::flex();
        let payload = Tensors::from([
            ("first.p".into(), (vec![1, 2], vec![0.0, 0.0])),
            ("second.p".into(), (vec![1, 2], vec![-1.0, 1.0])),
        ]);
        let bytes = encode_f32_with_metadata(&payload, BTreeMap::new()).unwrap();
        let manifest = Checkpoint {
            config: Default::default(),
            updates: 1,
            sha256: format!("{:x}", Sha256::digest(&bytes)),
        };
        assert!(
            matches!(Adam::restore(manifest,bytes,BTreeMap::from([("p".into(),vec![1,2])]),&device,1,0.003),Err(e) if e.contains("negative AdamW second moment"))
        );
    }
}
