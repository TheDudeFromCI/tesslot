//! HF Qwen3 source schemas and tensor-only Kidra conversion; no weight downloads.
use crate::{Error, Mapping, Report, Tensors, import, inspect, invalid, plan};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Seek, SeekFrom},
};
use tesslot_model::qwen::Config;
/// Version of the tensor conversion contract, not a claim of executable Qwen inference.
pub const FORMAT: &str = "kidra-qwen-tensors/1";
/// HF names/shapes from Transformers v4.51.3 Qwen3; tied output is a logical alias.
pub fn source_shapes(config: &Config) -> Result<BTreeMap<String, Vec<usize>>, Error> {
    config.validate().map_err(invalid)?;
    let c = config;
    let d = c.hidden_size;
    let q = c.query_width().map_err(invalid)?;
    let kv = c.kv_width().map_err(invalid)?;
    // Keep malformed operator configs from expanding an unbounded metadata map.
    if c.num_hidden_layers > 4096 {
        return Err(invalid(
            "Qwen layer count exceeds supported metadata limit 4096",
        ));
    }
    let mut shapes = BTreeMap::from([
        ("model.embed_tokens.weight".into(), vec![c.vocab_size, d]),
        ("model.norm.weight".into(), vec![d]),
    ]);
    if !c.tie_word_embeddings {
        shapes.insert("lm_head.weight".into(), vec![c.vocab_size, d]);
    }
    for i in 0..c.num_hidden_layers {
        for (name, shape) in [
            ("self_attn.q_proj.weight", vec![q, d]),
            ("self_attn.k_proj.weight", vec![kv, d]),
            ("self_attn.v_proj.weight", vec![kv, d]),
            ("self_attn.o_proj.weight", vec![d, q]),
            ("self_attn.q_norm.weight", vec![c.head_dim]),
            ("self_attn.k_norm.weight", vec![c.head_dim]),
            ("input_layernorm.weight", vec![d]),
            ("post_attention_layernorm.weight", vec![d]),
            ("mlp.gate_proj.weight", vec![c.intermediate_size, d]),
            ("mlp.up_proj.weight", vec![c.intermediate_size, d]),
            ("mlp.down_proj.weight", vec![d, c.intermediate_size]),
        ] {
            shapes.insert(format!("model.layers.{i}.{name}"), shape);
        }
    }
    byte_count(&shapes)?;
    Ok(shapes)
}
fn byte_count(shapes: &BTreeMap<String, Vec<usize>>) -> Result<usize, Error> {
    shapes.values().try_fold(0usize, |total, shape| {
        let n = shape
            .iter()
            .try_fold(4usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| invalid("Qwen tensor byte size overflow"))?;
        total
            .checked_add(n)
            .ok_or_else(|| invalid("Qwen aggregate size overflow"))
    })
}
/// Seeded fixture initialization; projection weights uniform in [-range,range], norm scales one.
/// Reproducibility depends on names/shapes and seed, not RNG implementation versions.
pub fn random_source(config: &Config, seed: u64, max_bytes: usize) -> Result<Tensors, Error> {
    let shapes = source_shapes(config)?;
    if byte_count(&shapes)? > max_bytes {
        return Err(invalid("random source exceeds explicit byte budget"));
    }
    let mut rng = Random(seed);
    Ok(shapes
        .into_iter()
        .map(|(name, shape)| {
            let n = shape.iter().product();
            let norm = name.ends_with("norm.weight");
            let data = (0..n)
                .map(|_| {
                    if norm {
                        1.0
                    } else {
                        rng.next() * config.initializer_range as f32
                    }
                })
                .collect();
            (name, (shape, data))
        })
        .collect())
}
struct Random(u64);
impl Random {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}
/// Converted weight payload plus a complete source mapping, logical aliases and provenance.
pub struct Converted {
    /// Reused HF tensor names plus separately initialized Kidra components.
    pub tensors: Tensors,
    /// Serializable conversion manifest; store beside the output weights.
    pub manifest: Manifest,
}
/// Artifact provenance; native Kidra tokenizer extension remains a separate explicit step.
#[derive(Debug, Serialize)]
pub struct Manifest {
    /// Independent conversion contract version.
    pub format: String,
    /// Complete source-file hash, not merely its header.
    pub source_sha256: String,
    /// Immutable base architecture metadata.
    pub source_config: Config,
    /// Seed for new components only; source weights are never reinitialized.
    pub seed: u64,
    /// Slot capacity for added window identities.
    pub windows: usize,
    /// All source mappings and explicit new components.
    pub report: Report,
    /// Alias -> stored tensor; avoids copying or reinitializing a tied output head.
    pub aliases: BTreeMap<String, String>,
    /// Base tokenizer/controls must still be deliberately converted before execution/training.
    pub tokenizer_conversion_required: bool,
}
/// Validate and convert a supplied single-file Qwen3 source within an explicit FP32 budget.
/// The output retains Qwen local blocks/norms and adds full-rank shared global exchange.
/// No claim is made that the tiny single-head reference can execute these GQA tensors.
pub fn convert(
    reader: &mut (impl Read + Seek),
    config: &Config,
    windows: usize,
    seed: u64,
    max_bytes: usize,
) -> Result<Converted, Error> {
    if !(1..=4096).contains(&windows) {
        return Err(invalid("conversion window slots must be 1..4096"));
    }
    let base = source_shapes(config)?;
    let mut expected = base.clone();
    let d = config.hidden_size;
    let q = config.query_width().map_err(invalid)?;
    let kv = config.kv_width().map_err(invalid)?;
    // Separate input and KV identity tables allow hidden width != KV projection width.
    for (name, shape) in [
        ("kidra.window_embeddings.weight", vec![windows, d]),
        ("kidra.query_embedding.weight", vec![1, d]),
        ("kidra.exchange.identities.weight", vec![windows, kv]),
        ("kidra.exchange.query.weight", vec![q, d]),
        ("kidra.exchange.out.weight", vec![d, q]),
        ("kidra.exchange.norm.weight", vec![d]),
        ("kidra.exchange.norm.bias", vec![d]),
    ] {
        expected.insert(name.into(), shape);
    }
    if byte_count(&expected)? > max_bytes {
        return Err(invalid("converted FP32 parameters exceed byte budget"));
    }
    let initialize: BTreeSet<_> = expected
        .keys()
        .filter(|n| !base.contains_key(*n))
        .cloned()
        .collect();
    let mapping = Mapping {
        expected: expected.clone(),
        sources: base.keys().map(|n| (n.clone(), n.clone())).collect(),
        initialize: initialize.clone(),
    };
    let inventory = inspect(reader)?;
    let report = plan(&inventory, &mapping);
    if !report.is_valid() {
        return Err(invalid(serde_json::to_string(&report).map_err(invalid)?));
    }
    let mut rng = Random(seed);
    let initial = initialize
        .into_iter()
        .map(|name| {
            let shape = expected[&name].clone();
            let n = shape.iter().product();
            let values = (0..n)
                .map(|_| {
                    if name.ends_with("norm.weight") {
                        1.0
                    } else if name.ends_with("norm.bias") {
                        0.0
                    } else {
                        rng.next() * config.initializer_range as f32
                    }
                })
                .collect();
            (name, (shape, values))
        })
        .collect();
    let tensors = import(reader, &mapping, &initial, max_bytes)?;
    reader.seek(SeekFrom::Start(0))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    let aliases = if config.tie_word_embeddings {
        BTreeMap::from([("lm_head.weight".into(), "model.embed_tokens.weight".into())])
    } else {
        BTreeMap::new()
    };
    Ok(Converted {
        tensors,
        manifest: Manifest {
            format: FORMAT.into(),
            source_sha256: format!("{:x}", hash.finalize()),
            source_config: config.clone(),
            seed,
            windows,
            report,
            aliases,
            tokenizer_conversion_required: true,
        },
    })
}
