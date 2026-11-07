//! Shared Qwen local blocks/exchange for dense and attached projected-KV execution.
use super::*;
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
                            .slice([r.slot..r.slot + 1, 0..d])
                })
                .collect(),
            0,
        );
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
            x = x + self.linear(
                up * gate.clone() / (1.0 + (-gate).exp()),
                &(p + "mlp.down_proj.weight"),
            );
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
