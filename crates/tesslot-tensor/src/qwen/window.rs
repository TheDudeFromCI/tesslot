//! Shared Qwen local blocks/exchange for dense and attached projected-KV execution.
use super::*;
use burn_tensor::Int;
pub(super) struct StateRow {
    pub slot: usize,
    pub position: usize,
    pub token: Option<u32>,
}
pub(super) struct Memory {
    pub slot: usize,
    pub keys: Vec<T>,
    pub values: Vec<T>,
}
pub(super) struct Frame {
    pub logits: T,
    pub keys: Vec<T>,
    pub values: Vec<T>,
}
impl Model {
    // One select per embedding table/frame avoids a full embedding-table gradient
    // allocation for every individual sliced token in the same frame.
    fn frame_embeddings(&self, rows: &[StateRow]) -> T {
        let n = rows.len();
        let d = self.config.hidden_size;
        let query = self.parameters["kidra.query_embedding.weight"]
            .clone()
            .repeat_dim(0, n);
        let token = if rows.iter().all(|r| r.token.is_none()) {
            // Do not attach the huge token table to query-only frames via dummy IDs.
            query
        } else {
            let ids = rows
                .iter()
                .map(|r| r.token.unwrap_or(0) as i64)
                .collect::<Vec<_>>();
            let input = self.parameters["model.embed_tokens.weight"].clone().select(
                0,
                Tensor::<1, Int>::from_data(TensorData::new(ids, [n]), &self.device),
            );
            if rows.iter().all(|r| r.token.is_some()) {
                input
            } else {
                let mask = rows
                    .iter()
                    .flat_map(|r| std::iter::repeat_n(r.token.is_none(), d))
                    .collect::<Vec<_>>();
                input.mask_where(
                    Tensor::<2, Bool>::from_data(TensorData::new(mask, [n, d]), &self.device),
                    query,
                )
            }
        };
        let slots = rows.iter().map(|r| r.slot as i64).collect::<Vec<_>>();
        token
            + self.parameters["kidra.window_embeddings.weight"]
                .clone()
                .select(
                    0,
                    Tensor::<1, Int>::from_data(TensorData::new(slots, [n]), &self.device),
                )
    }
    fn attend(&self, q: T, k: T, v: T, mask: Tensor<2, Bool>) -> T {
        let weights = softmax(
            (self.matmul(q, k.transpose()) / (self.config.head_dim as f32).sqrt())
                .mask_fill(mask.clone(), f32::NEG_INFINITY)
                .mask_fill(mask.clone().all_dim(1), 0.0),
            1,
        )
        .mask_fill(mask, 0.0);
        self.matmul(weights, v)
    }
    pub(super) fn window_frame(
        &self,
        rows: &[StateRow],
        current_keys: usize,
        prefix: &[&Memory],
        allowed: &[bool],
    ) -> Result<Frame, String> {
        self.window_frame_logits(rows, current_keys, prefix, allowed, 0..rows.len())
    }
    pub(super) fn window_frame_logits(
        &self,
        rows: &[StateRow],
        current_keys: usize,
        prefix: &[&Memory],
        allowed: &[bool],
        logit_rows: std::ops::Range<usize>,
    ) -> Result<Frame, String> {
        if logit_rows.start >= logit_rows.end || logit_rows.end > rows.len() {
            return Err("invalid logit row range".into());
        }
        let c = &self.config;
        let capacity = self
            .exchange_windows
            .ok_or("window exchange not configured")?;
        let n = rows.len();
        let count = prefix.len() + current_keys;
        if n == 0
            || current_keys > n
            || allowed.len() != n * count
            || rows.iter().any(|r| {
                r.slot >= capacity
                    || r.position >= c.max_position_embeddings
                    || r.token.is_some_and(|t| t as usize >= c.vocab_size)
            })
        {
            return Err("invalid window frame, slot, position, token or mask dimensions".into());
        }
        let positions = rows.iter().map(|r| r.position).collect::<Vec<_>>();
        let slots = prefix
            .iter()
            .map(|m| m.slot)
            .chain(rows[..current_keys].iter().map(|r| r.slot))
            .collect::<Vec<_>>();
        let mask = |local: bool| {
            Tensor::<2, Bool>::from_data(
                TensorData::new(
                    rows.iter()
                        .enumerate()
                        .flat_map(|(i, r)| {
                            slots.iter().enumerate().map(move |(j, slot)| {
                                !allowed[i * count + j] || (local && r.slot != *slot)
                            })
                        })
                        .collect::<Vec<_>>(),
                    [n, count],
                ),
                &self.device,
            )
        };
        let masks = if count > 0 {
            Some((mask(true), mask(false)))
        } else {
            None
        };
        let d = c.hidden_size;
        let mut x = self.frame_embeddings(rows);
        let mut own_keys = Vec::new();
        let mut own_values = Vec::new();
        for layer in 0..c.num_hidden_layers {
            let device = self.layer_device(layer);
            x = x.to_device(&device);
            let masks = masks.as_ref().map(|(local, global)| {
                (
                    local.clone().to_device(&device),
                    global.clone().to_device(&device),
                )
            });
            let p = format!("model.layers.{layer}.");
            let h = self.norm(x.clone(), &(p.clone() + "input_layernorm.weight"));
            let q = self.linear(h.clone(), &(p.clone() + "self_attn.q_proj.weight"));
            let raw_k = self.linear(h.clone(), &(p.clone() + "self_attn.k_proj.weight"));
            let values = self.linear(h, &(p.clone() + "self_attn.v_proj.weight"));
            let mut key_heads = Vec::new();
            for head in 0..c.num_key_value_heads {
                let start = head * c.head_dim;
                key_heads.push(self.rope_positions(
                    self.norm(
                        raw_k.clone().slice([0..n, start..start + c.head_dim]),
                        &(p.clone() + "self_attn.k_norm.weight"),
                    ),
                    &positions,
                ));
            }
            let keys = T::cat(key_heads, 1);
            let memory = if count > 0 {
                let mut ks = prefix
                    .iter()
                    .map(|m| m.keys[layer].clone())
                    .collect::<Vec<_>>();
                let mut vs = prefix
                    .iter()
                    .map(|m| m.values[layer].clone())
                    .collect::<Vec<_>>();
                if current_keys > 0 {
                    ks.push(keys.clone().slice([0..current_keys, 0..c.kv_width()?]));
                    vs.push(values.clone().slice([0..current_keys, 0..c.kv_width()?]));
                }
                Some((T::cat(ks, 0), T::cat(vs, 0)))
            } else {
                None
            };
            own_keys.push(keys);
            own_values.push(values);
            let mut responses = Vec::new();
            for head in 0..c.num_attention_heads {
                let start = head * c.head_dim;
                let query = self.rope_positions(
                    self.norm(
                        q.clone().slice([0..n, start..start + c.head_dim]),
                        &(p.clone() + "self_attn.q_norm.weight"),
                    ),
                    &positions,
                );
                let kh = head / (c.num_attention_heads / c.num_key_value_heads);
                let kstart = kh * c.head_dim;
                responses.push(if let (Some((k, v)), Some((mask, _))) = (&memory, &masks) {
                    self.attend(
                        query,
                        k.clone().slice([0..count, kstart..kstart + c.head_dim]),
                        v.clone().slice([0..count, kstart..kstart + c.head_dim]),
                        mask.clone(),
                    )
                } else {
                    query * 0.0
                });
            }
            x = x + self.linear(
                T::cat(responses, 1),
                &(p.clone() + "self_attn.o_proj.weight"),
            );
            let h = self.norm(x.clone(), &(p.clone() + "post_attention_layernorm.weight"));
            let gate = self.linear(h.clone(), &(p.clone() + "mlp.gate_proj.weight"));
            let up = self.linear(h, &(p.clone() + "mlp.up_proj.weight"));
            x = x + self.linear(up * silu(gate), &(p + "mlp.down_proj.weight"));
            let centered = x.clone() - x.clone().mean_dim(1);
            let h = centered.clone() / (centered.powf_scalar(2.0).mean_dim(1) + 1e-5).sqrt();
            let h = h * self.parameters["kidra.exchange.norm.weight"]
                .clone()
                .to_device(&device)
                + self.parameters["kidra.exchange.norm.bias"]
                    .clone()
                    .to_device(&device);
            let q = self.linear(h, "kidra.exchange.query.weight");
            let mut responses = Vec::new();
            for head in 0..c.num_attention_heads {
                let start = head * c.head_dim;
                let kh = head / (c.num_attention_heads / c.num_key_value_heads);
                let kstart = kh * c.head_dim;
                let query = q.clone().slice([0..n, start..start + c.head_dim]);
                responses.push(if let (Some((k, v)), Some((_, mask))) = (&memory, &masks) {
                    let identities = T::cat(
                        slots
                            .iter()
                            .map(|slot| {
                                self.parameters["kidra.exchange.identities.weight"]
                                    .clone()
                                    .to_device(&device)
                                    .slice([*slot..*slot + 1, kstart..kstart + c.head_dim])
                            })
                            .collect(),
                        0,
                    );
                    self.attend(
                        query,
                        k.clone().slice([0..count, kstart..kstart + c.head_dim]) + identities,
                        v.clone().slice([0..count, kstart..kstart + c.head_dim]),
                        mask.clone(),
                    )
                } else {
                    query * 0.0
                });
            }
            x = x + self.linear(T::cat(responses, 1), "kidra.exchange.out.weight");
        }
        let logits = self.linear(
            self.norm(
                x.slice([logit_rows, 0..d]).to_device(&self.device),
                "model.norm.weight",
            ),
            if c.tie_word_embeddings {
                "model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            },
        );
        Ok(Frame {
            logits,
            keys: own_keys,
            values: own_values,
        })
    }
}

#[cfg(test)]
mod embedding_gather_tests {
    use super::*;
    fn model() -> Model {
        Model::remix_demo(
            serde_json::from_slice(include_bytes!(
                "../../../../fixtures/tokenizer-identity.json"
            ))
            .unwrap(),
            24,
            42,
            true,
        )
        .unwrap()
    }
    fn old_slices(model: &Model, rows: &[StateRow]) -> T {
        let d = model.config.hidden_size;
        T::cat(
            rows.iter()
                .map(|r| {
                    let input = match r.token {
                        Some(token) => model.parameters["model.embed_tokens.weight"]
                            .clone()
                            .slice([token as usize..token as usize + 1, 0..d]),
                        None => model.parameters["kidra.query_embedding.weight"].clone(),
                    };
                    input
                        + model.parameters["kidra.window_embeddings.weight"]
                            .clone()
                            .slice([r.slot..r.slot + 1, 0..d])
                })
                .collect(),
            0,
        )
    }
    #[test]
    fn batched_gather_matches_slice_values_and_all_embedding_gradients() {
        verify(false);
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires one qualified CUDA device; run by the target-prefix workstation helper"]
    fn cuda_embedding_gather_gradients() {
        verify(true);
    }
    fn verify(cuda: bool) {
        for tokens in [
            vec![Some(0), Some(1), Some(1), None],
            vec![None, None, None, None],
            vec![Some(1), Some(1), Some(2), Some(0)],
        ] {
            let rows = tokens
                .into_iter()
                .enumerate()
                .map(|(i, token)| StateRow {
                    slot: [0, 1, 23, 1][i],
                    position: i,
                    token,
                })
                .collect::<Vec<_>>();
            let mut gathered_model = model();
            let mut reference_model = model();
            if cuda {
                use super::super::runtime::{Plan, Precision, Target};
                for model in [&mut gathered_model, &mut reference_model] {
                    let target = Target::Cuda { index: 0 };
                    model
                        .configure_runtime(Plan {
                            root: target.clone(),
                            layers: vec![target; model.config.num_hidden_layers],
                            precision: Precision::Tf32,
                            parameter_budget_bytes: 2 * 1024 * 1024,
                        })
                        .unwrap();
                }
            }
            let gathered = gathered_model.frame_embeddings(&rows);
            let reference = old_slices(&reference_model, &rows);
            assert_eq!(
                gathered.clone().into_data().try_to_vec::<f32>().unwrap(),
                reference.clone().into_data().try_to_vec::<f32>().unwrap()
            );
            let a = gathered.powf_scalar(2.0).sum().backward();
            let b = reference.powf_scalar(2.0).sum().backward();
            for name in [
                "model.embed_tokens.weight",
                "kidra.window_embeddings.weight",
                "kidra.query_embedding.weight",
            ] {
                let read = |model: &Model, gs: &_| {
                    model.parameters[name]
                        .grad(gs)
                        .map(|g| g.into_data().try_to_vec::<f32>().unwrap())
                        .unwrap_or_else(|| {
                            vec![0.0; model.parameters[name].dims().iter().product()]
                        })
                };
                let actual = read(&gathered_model, &a);
                let expected = read(&reference_model, &b);
                for (x, y) in actual.iter().zip(expected) {
                    assert!((*x - y).abs() < 1e-6, "{name}: {x} vs {y}");
                }
                if name == "model.embed_tokens.weight" && rows.iter().all(|r| r.token.is_none()) {
                    assert!(
                        gathered_model.parameters[name].grad(&a).is_none(),
                        "query-only frames must not attach a dummy token table"
                    );
                }
            }
        }
    }
}
