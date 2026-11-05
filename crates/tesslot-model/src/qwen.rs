//! Native, allocation-free Qwen3 configuration inspection. Does not load weights.
use serde::{Deserialize, Serialize};
/// Explicit supported HF Qwen3 config; unknown numerical features fail closed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Hugging Face `architectures` metadata.
    pub architectures: Vec<String>,
    /// Hugging Face `attention_bias` metadata.
    pub attention_bias: bool,
    /// Hugging Face `attention_dropout` metadata.
    pub attention_dropout: f64,
    /// Hugging Face `bos_token_id` metadata.
    pub bos_token_id: usize,
    /// Hugging Face `eos_token_id` metadata.
    pub eos_token_id: usize,
    /// Hugging Face `head_dim` metadata.
    pub head_dim: usize,
    /// Hugging Face `hidden_act` metadata.
    pub hidden_act: String,
    /// Hugging Face `hidden_size` metadata.
    pub hidden_size: usize,
    /// Hugging Face `initializer_range` metadata.
    pub initializer_range: f64,
    /// Hugging Face `intermediate_size` metadata.
    pub intermediate_size: usize,
    /// Hugging Face `max_position_embeddings` metadata.
    pub max_position_embeddings: usize,
    /// Hugging Face `max_window_layers` metadata.
    pub max_window_layers: usize,
    /// Hugging Face `model_type` metadata.
    pub model_type: String,
    /// Hugging Face `num_attention_heads` metadata.
    pub num_attention_heads: usize,
    /// Hugging Face `num_hidden_layers` metadata.
    pub num_hidden_layers: usize,
    /// Hugging Face `num_key_value_heads` metadata.
    pub num_key_value_heads: usize,
    /// Hugging Face `rms_norm_eps` metadata.
    pub rms_norm_eps: f64,
    /// Hugging Face `rope_scaling` metadata.
    pub rope_scaling: Option<serde_json::Value>,
    /// Hugging Face `rope_theta` metadata.
    pub rope_theta: f64,
    /// Hugging Face `sliding_window` metadata.
    pub sliding_window: Option<usize>,
    /// Hugging Face `tie_word_embeddings` metadata.
    pub tie_word_embeddings: bool,
    /// Hugging Face `torch_dtype` metadata.
    pub torch_dtype: String,
    /// Hugging Face `transformers_version` metadata.
    pub transformers_version: String,
    /// Hugging Face `use_cache` metadata.
    pub use_cache: bool,
    /// Hugging Face `use_sliding_window` metadata.
    pub use_sliding_window: bool,
    /// Hugging Face `vocab_size` metadata.
    pub vocab_size: usize,
}
impl Config {
    /// Parse actual HF config and reject unsupported numerical semantics.
    pub fn from_json(data: &[u8]) -> Result<Self, String> {
        let c: Self =
            serde_json::from_slice(data).map_err(|e| format!("Qwen3 configuration: {e}"))?;
        c.validate()?;
        Ok(c)
    }
    /// Validate shapes without assuming hidden_size / heads equals head_dim.
    pub fn validate(&self) -> Result<(), String> {
        if self.model_type != "qwen3"
            || self.architectures != ["Qwen3ForCausalLM"]
            || self.hidden_act != "silu"
        {
            return Err("unsupported Qwen3 architecture or activation".into());
        }
        if self.attention_bias
            || self.attention_dropout != 0.0
            || self.use_sliding_window
            || self.sliding_window.is_some()
            || self.rope_scaling.is_some()
        {
            return Err(
                "unsupported attention bias/dropout, sliding window or RoPE scaling".into(),
            );
        }
        if [
            self.hidden_size,
            self.intermediate_size,
            self.head_dim,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.num_hidden_layers,
            self.vocab_size,
            self.max_position_embeddings,
        ]
        .contains(&0)
            || !self.head_dim.is_multiple_of(2)
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err("invalid dimensions, RoPE head width or grouped-query heads".into());
        }
        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 1.0
            || !self.initializer_range.is_finite()
            || self.initializer_range <= 0.0
        {
            return Err("invalid finite normalization/RoPE/initialization constants".into());
        }
        if self.bos_token_id >= self.vocab_size
            || self.eos_token_id >= self.vocab_size
            || !["float32", "float16", "bfloat16"].contains(&self.torch_dtype.as_str())
        {
            return Err("invalid base controls or unsupported weight dtype".into());
        }
        self.query_width()?;
        self.kv_width()?;
        self.vocab_size
            .checked_mul(self.hidden_size)
            .ok_or("embedding dimensions overflow")?;
        Ok(())
    }
    /// Query projection output width, which may differ from hidden_size.
    pub fn query_width(&self) -> Result<usize, String> {
        self.num_attention_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| "Q projection dimensions overflow".into())
    }
    /// Grouped-query key/value projection output width.
    pub fn kv_width(&self) -> Result<usize, String> {
        self.num_key_value_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| "KV projection dimensions overflow".into())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_qwen_configs_and_unsupported_features() {
        for source in [
            include_bytes!("../../../examples/qwen3-0.6b.config.json").as_slice(),
            include_bytes!("../../../examples/qwen3-1.7b.config.json").as_slice(),
        ] {
            let c = Config::from_json(source).unwrap();
            assert_eq!(c.query_width().unwrap(), 2048);
            assert_eq!(c.kv_width().unwrap(), 1024);
            assert_eq!(c.head_dim, 128);
            let mut v: serde_json::Value = serde_json::from_slice(source).unwrap();
            v["rope_scaling"] = serde_json::json!({"type":"dynamic"});
            assert!(Config::from_json(&serde_json::to_vec(&v).unwrap()).is_err());
        }
        let c =
            Config::from_json(include_bytes!("../../../examples/qwen3-0.6b.config.json")).unwrap();
        assert_ne!(c.hidden_size, c.query_width().unwrap());
        let mut bad = c.clone();
        bad.num_key_value_heads = 0;
        assert!(bad.validate().is_err());
        bad = c.clone();
        bad.head_dim = 3;
        assert!(bad.validate().is_err());
        bad = c;
        bad.num_attention_heads = usize::MAX - 1;
        assert!(bad.validate().is_err());
    }
}
