//! Local HF checkpoints and append-only tokenizer conversion. No network access.
use crate::{Error, NativeTokenizer, Tensors, inspect, invalid, read_f32};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Seek},
    path::{Component, Path},
};
use tesslot_model::qwen::Config;

/// Validated source files and their complete content digests.
#[derive(Debug, Serialize)]
pub struct Provenance {
    /// Hashes of the index (when present) and every referenced weight file.
    pub files: BTreeMap<String, String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Index {
    #[serde(default)]
    metadata: BTreeMap<String, serde_json::Value>,
    #[serde(deserialize_with = "unique_map")]
    weight_map: BTreeMap<String, String>,
}
fn unique_map<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<BTreeMap<String, String>, D::Error> {
    struct Visitor;
    impl<'de> serde::de::Visitor<'de> for Visitor {
        type Value = BTreeMap<String, String>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("unique tensor names")
        }
        fn visit_map<M: serde::de::MapAccess<'de>>(
            self,
            mut map: M,
        ) -> Result<Self::Value, M::Error> {
            let mut result = BTreeMap::new();
            while let Some((key, value)) = map.next_entry::<String, String>()? {
                if result.insert(key, value).is_some() {
                    return Err(serde::de::Error::custom("duplicate tensor in weight_map"));
                }
            }
            Ok(result)
        }
    }
    d.deserialize_map(Visitor)
}
fn bounded(path: &Path, limit: usize) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(invalid(format!(
            "{} exceeds {limit} byte limit",
            path.display()
        )));
    }
    Ok(bytes)
}
fn hash(file: &mut File) -> Result<String, Error> {
    file.rewind()?;
    let mut hash = Sha256::new();
    let mut block = [0; 65536];
    loop {
        let n = file.read(&mut block)?;
        if n == 0 {
            break;
        }
        hash.update(&block[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
/// Read exactly the HF schema, one tensor at a time, within an explicit aggregate FP32 budget.
/// Shard paths must be flat filenames whose resolved targets stay inside the checkpoint root.
pub fn load(
    root: &Path,
    config: &Config,
    max_bytes: usize,
) -> Result<(Tensors, Provenance), Error> {
    let expected = crate::qwen::source_shapes(config)?;
    let required = expected
        .values()
        .try_fold(0usize, |n, s| {
            s.iter()
                .try_fold(4usize, |v, d| v.checked_mul(*d))
                .and_then(|v| n.checked_add(v))
        })
        .ok_or_else(|| invalid("weight size overflow"))?;
    if required > max_bytes {
        return Err(invalid(format!(
            "FP32 weights require {required} bytes; budget is {max_bytes}"
        )));
    }
    let root = root.canonicalize()?;
    let index_path = root.join("model.safetensors.index.json");
    let mut files = BTreeMap::new();
    let mut expected_total = None;
    let mapping = if index_path.exists() {
        let bytes = bounded(&index_path, 16 * 1024 * 1024)?;
        let index: Index = serde_json::from_slice(&bytes).map_err(invalid)?;
        expected_total = index
            .metadata
            .get("total_size")
            .map(|v| {
                v.as_u64().ok_or_else(|| {
                    invalid("index metadata.total_size must be a nonnegative byte count")
                })
            })
            .transpose()?;
        files.insert(
            "model.safetensors.index.json".into(),
            format!("{:x}", Sha256::digest(bytes)),
        );
        index.weight_map
    } else {
        expected
            .keys()
            .map(|n| (n.clone(), "model.safetensors".into()))
            .collect()
    };
    if mapping.keys().ne(expected.keys()) {
        let missing = expected
            .keys()
            .filter(|n| !mapping.contains_key(*n))
            .take(32)
            .collect::<Vec<_>>();
        let unexpected = mapping
            .keys()
            .filter(|n| !expected.contains_key(*n))
            .take(32)
            .collect::<Vec<_>>();
        return Err(invalid(format!(
            "weight_map differs from Qwen schema; missing (first 32): {missing:?}; unexpected (first 32): {unexpected:?}"
        )));
    }
    let shards = mapping.values().cloned().collect::<BTreeSet<_>>();
    let mut tensors = Tensors::new();
    let mut actual_total = 0u64;
    for name in shards {
        if Path::new(&name).components().count() != 1
            || !matches!(
                Path::new(&name).components().next(),
                Some(Component::Normal(_))
            )
            || !name.ends_with(".safetensors")
        {
            return Err(invalid(format!("unsafe shard filename {name}")));
        }
        let path = root.join(&name).canonicalize()?;
        if path.parent() != Some(root.as_path()) {
            return Err(invalid("shard escapes checkpoint directory"));
        }
        let mut file = File::open(path)?;
        let before = hash(&mut file)?;
        let inventory = inspect(&mut file)?;
        actual_total = actual_total
            .checked_add(inventory.file_bytes - inventory.data_start)
            .ok_or_else(|| invalid("sharded payload byte count overflow"))?;
        let names = mapping
            .iter()
            .filter(|(_, s)| **s == name)
            .map(|(n, _)| n)
            .collect::<BTreeSet<_>>();
        if inventory.tensors.keys().collect::<BTreeSet<_>>() != names {
            return Err(invalid(format!(
                "shard {name} contents disagree with index"
            )));
        }
        for (tensor, info) in &inventory.tensors {
            if info.shape != expected[tensor] {
                return Err(invalid(format!(
                    "incompatible shape for {tensor}: {:?}, expected {:?}",
                    info.shape, expected[tensor]
                )));
            }
            let (_, data) = read_f32(&mut file, tensor, max_bytes)?;
            tensors.insert(tensor.clone(), (info.shape.clone(), data));
        }
        let after = hash(&mut file)?;
        if before != after {
            return Err(invalid(format!(
                "source shard {name} changed during import"
            )));
        }
        files.insert(name, after);
    }
    if expected_total.is_some_and(|n| n != actual_total) {
        return Err(invalid(format!(
            "index metadata.total_size differs from actual payload bytes {actual_total}"
        )));
    }
    Ok((tensors, Provenance { files }))
}
/// Literal context delimiters emitted by the pinned Trajectory Remix compiler.
pub const CONTEXT_MARKERS: [&str; 8] = [
    "<tool>",
    "</tool>",
    "<msg>",
    "</msg>",
    "<tool_result>",
    "</tool_result>",
    "<ctx>",
    "</ctx>",
];
/// Number of appended rows: three native lifecycle controls followed by eight Remix markers.
pub const ADDED_TOKEN_ROWS: usize = 3 + CONTEXT_MARKERS.len();

/// Append-only conversion metadata. Original token IDs and all padded embedding rows survive.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerConversion {
    /// Independent format version.
    pub format: String,
    /// Normalized source tokenizer digest.
    pub source_sha256: String,
    /// Original embedding row count (may exceed tokenizer vocabulary).
    pub original_rows: usize,
    /// Explicitly reserved padded rows; these are never reused as Kidra controls.
    pub reserved: BTreeMap<String, u32>,
    /// Dedicated literal context markers emitted by Trajectory Remix.
    pub context_markers: BTreeMap<String, u32>,
    /// Exact identity to use in Remix exports.
    pub target: tesslot_dataset::TokenizerIdentity,
}
impl TokenizerConversion {
    fn validate(&self) -> Result<(), Error> {
        if self.original_rows > 2_000_000 {
            return Err(invalid("tokenizer row limit exceeded"));
        }
        self.target.validate().map_err(invalid)?;
        let markers = CONTEXT_MARKERS
            .iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), (self.original_rows + 3 + i) as u32))
            .collect::<BTreeMap<_, _>>();
        if self.format != "kidra-tokenizer-append/2"
            || self.original_rows.checked_add(ADDED_TOKEN_ROWS) != Some(self.target.vocab_size)
            || self.target.special_tokens.bos as usize != self.original_rows
            || self.target.special_tokens.eos as usize != self.original_rows + 1
            || self.target.special_tokens.noop as usize != self.original_rows + 2
            || self.context_markers != markers
        {
            return Err(invalid(
                "incompatible append-only control/context-marker mapping; reconvert from the base model",
            ));
        }
        Ok(())
    }
}
/// Normalize a base tokenizer, preserve padded rows, then append lifecycle and context tokens.
/// Existing spellings cause an error rather than silently changing their semantics.
pub fn extend_tokenizer(
    bytes: &[u8],
    original_rows: usize,
) -> Result<(Vec<u8>, TokenizerConversion), Error> {
    if bytes.len() > 64 * 1024 * 1024 || original_rows > 2_000_000 {
        return Err(invalid("tokenizer byte/row limit exceeded"));
    }
    let mut t = tokenizers::Tokenizer::from_bytes(bytes).map_err(invalid)?;
    t.with_padding(None);
    t.with_truncation(None).map_err(invalid)?;
    let source_sha256 = format!(
        "{:x}",
        Sha256::digest(t.to_string(false).map_err(invalid)?.as_bytes())
    );
    let vocab = t.get_vocab(true);
    let ids = vocab.values().copied().collect::<BTreeSet<_>>();
    if ids.len() != vocab.len()
        || ids.iter().copied().ne(0..ids.len() as u32)
        || vocab.len() > original_rows
    {
        return Err(invalid(
            "base tokenizer must have contiguous unique IDs within embedding rows",
        ));
    }
    let controls = ["<|kidra_bos|>", "<|kidra_eos|>", "<|kidra_noop|>"];
    if controls
        .iter()
        .chain(CONTEXT_MARKERS.iter())
        .any(|s| t.token_to_id(s).is_some())
    {
        return Err(invalid(
            "dedicated control/context marker spelling already exists; explicit migration required",
        ));
    }
    let mut reserved = BTreeMap::new();
    for id in vocab.len()..original_rows {
        let text = format!("<|kidra_reserved_{id}|>");
        if t.token_to_id(&text).is_some() {
            return Err(invalid(
                "reserved padding spelling conflicts with base vocabulary",
            ));
        }
        t.add_special_tokens([tokenizers::AddedToken::from(text.clone(), true)])
            .map_err(invalid)?;
        if t.token_to_id(&text) != Some(id as u32) {
            return Err(invalid("tokenizer did not preserve padded row IDs"));
        }
        reserved.insert(text, id as u32);
    }
    for (offset, text) in controls.iter().enumerate() {
        t.add_special_tokens([tokenizers::AddedToken::from(*text, true)])
            .map_err(invalid)?;
        if t.token_to_id(text) != Some((original_rows + offset) as u32) {
            return Err(invalid("control ID assignment is not append-only"));
        }
    }
    let mut context_markers = BTreeMap::new();
    for (offset, text) in CONTEXT_MARKERS.iter().enumerate() {
        t.add_special_tokens([tokenizers::AddedToken::from(*text, true)])
            .map_err(invalid)?;
        let id = (original_rows + controls.len() + offset) as u32;
        if t.token_to_id(text) != Some(id)
            || t.encode(*text, false).map_err(invalid)?.get_ids() != [id]
        {
            return Err(invalid(
                "context marker assignment is not append-only and atomic",
            ));
        }
        context_markers.insert(text.to_string(), id);
    }
    if vocab
        .iter()
        .any(|(text, id)| t.token_to_id(text) != Some(*id))
    {
        return Err(invalid("conversion changed an existing token ID"));
    }
    let bytes = t.to_string(false).map_err(invalid)?.into_bytes();
    let native = NativeTokenizer::from_bytes(&bytes, controls[0], controls[1], controls[2])?;
    Ok((
        bytes,
        TokenizerConversion {
            format: "kidra-tokenizer-append/2".into(),
            source_sha256,
            original_rows,
            reserved,
            context_markers,
            target: native.identity().clone(),
        },
    ))
}
/// Extend embedding and untied output rows only. Source coordinates remain bitwise unchanged.
pub fn extend_embeddings(
    tensors: &mut Tensors,
    config: &mut Config,
    conversion: &TokenizerConversion,
    seed: u64,
    max_bytes: usize,
) -> Result<(), Error> {
    if config.vocab_size != conversion.original_rows {
        return Err(invalid(
            "tokenizer conversion source row count differs from model",
        ));
    }
    config.validate().map_err(invalid)?;
    conversion.validate()?;
    let target = conversion.target.vocab_size;
    let mut rng = crate::qwen::Random(seed);
    let extra = (target - config.vocab_size)
        .checked_mul(config.hidden_size)
        .ok_or_else(|| invalid("vocabulary byte size overflow"))?;
    let current = tensors
        .values()
        .try_fold(0usize, |n, (_, v)| n.checked_add(v.len()))
        .ok_or_else(|| invalid("parameter size overflow"))?;
    let total = extra
        .checked_mul(if config.tie_word_embeddings { 1 } else { 2 })
        .and_then(|v| v.checked_add(current))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| invalid("parameter size overflow"))?;
    if total > max_bytes {
        return Err(invalid("extended FP32 weights exceed byte budget"));
    }
    for name in ["model.embed_tokens.weight", "lm_head.weight"] {
        if name == "lm_head.weight" && config.tie_word_embeddings {
            continue;
        }
        let (shape, values) = tensors
            .get(name)
            .ok_or_else(|| invalid(format!("missing {name}")))?;
        if *shape != [config.vocab_size, config.hidden_size]
            || values.len() != config.vocab_size * config.hidden_size
        {
            return Err(invalid(format!("invalid {name} shape")));
        }
    }
    for name in ["model.embed_tokens.weight", "lm_head.weight"] {
        if name == "lm_head.weight" && config.tie_word_embeddings {
            continue;
        }
        let (shape, values) = tensors
            .get_mut(name)
            .ok_or_else(|| invalid(format!("missing {name}")))?;
        values.extend((0..extra).map(|_| rng.next() * config.initializer_range as f32));
        shape[0] = target;
    }
    config.vocab_size = target;
    config.bos_token_id = conversion.target.special_tokens.bos as usize;
    config.eos_token_id = conversion.target.special_tokens.eos as usize;
    config.validate().map_err(invalid)
}
/// Validated complete converted artifact, ready for backend-specific allocation.
pub struct Converted {
    /// Target HF-shaped config with extended vocabulary and distinct controls.
    pub config: Config,
    /// FP32 master parameters; tied head is a logical alias, never duplicated.
    pub weights: Tensors,
    /// Exact training/export tokenizer identity.
    pub tokenizer: tesslot_dataset::TokenizerIdentity,
    /// Configurable window slot capacity.
    pub windows: usize,
    /// Initialization provenance.
    pub seed: u64,
    /// Complete converted weight digest.
    pub sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConvertedManifest {
    format: String,
    architecture: String,
    source_config: Config,
    source_files: BTreeMap<String, String>,
    tokenizer_conversion: TokenizerConversion,
    seed: u64,
    windows: usize,
    initialized: Vec<String>,
    tied_output: bool,
    config_sha256: String,
    weights_sha256: String,
}
/// Load an atomically published `convert-pretrained` output; validate digests and all identities.
pub fn load_converted(root: &Path, max_bytes: usize) -> Result<Converted, Error> {
    let m: ConvertedManifest =
        serde_json::from_slice(&bounded(&root.join("conversion.json"), 16 * 1024 * 1024)?)
            .map_err(invalid)?;
    let config = Config::from_json(&bounded(&root.join("config.json"), 65536)?).map_err(invalid)?;
    let tc = &m.tokenizer_conversion;
    tc.validate()?;
    let mut expected_config = m.source_config.clone();
    expected_config.vocab_size = tc.target.vocab_size;
    expected_config.bos_token_id = tc.target.special_tokens.bos as usize;
    expected_config.eos_token_id = tc.target.special_tokens.eos as usize;
    if m.format != "kidra-pretrained/1"
        || m.architecture != "kidra-qwen-remix/1"
        || tc.original_rows != m.source_config.vocab_size
        || !(1..=64).contains(&m.windows)
        || m.tied_output != config.tie_word_embeddings
        || m.source_files.is_empty()
        || serde_json::to_vec(&expected_config).map_err(invalid)?
            != serde_json::to_vec(&config).map_err(invalid)?
        || format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&config).map_err(invalid)?)
        ) != m.config_sha256
    {
        return Err(invalid(
            "incompatible converted artifact manifest/config/tokenizer",
        ));
    }
    let native = NativeTokenizer::from_bytes(
        &bounded(&root.join("tokenizer.json"), 64 * 1024 * 1024)?,
        "<|kidra_bos|>",
        "<|kidra_eos|>",
        "<|kidra_noop|>",
    )?;
    if serde_json::to_vec(native.identity()).map_err(invalid)?
        != serde_json::to_vec(&tc.target).map_err(invalid)?
    {
        return Err(invalid("converted tokenizer identity mismatch"));
    }
    for (marker, id) in &tc.context_markers {
        if native.encode(marker)? != [*id] {
            return Err(invalid("converted context marker ID mismatch"));
        }
    }
    let mut file = File::open(root.join("model.safetensors"))?;
    if hash(&mut file)? != m.weights_sha256 {
        return Err(invalid("converted weights checksum mismatch"));
    }
    let mut expected = crate::qwen::source_shapes(&config)?;
    let extra = crate::qwen::exchange_shapes(&config, m.windows)?;
    if m.initialized.iter().collect::<BTreeSet<_>>() != extra.keys().collect()
        || m.initialized.len() != extra.len()
    {
        return Err(invalid(
            "converted initialization report differs from architecture",
        ));
    }
    expected.extend(extra);
    let sources = expected.keys().map(|n| (n.clone(), n.clone())).collect();
    let weights = crate::import(
        &mut file,
        &crate::Mapping {
            expected,
            sources,
            initialize: Default::default(),
        },
        &Tensors::new(),
        max_bytes,
    )?;
    Ok(Converted {
        config,
        weights,
        tokenizer: tc.target.clone(),
        windows: m.windows,
        seed: m.seed,
        sha256: m.weights_sha256,
    })
}
