//! Native tokenizer loading and bounded SafeTensors inspection/import. No downloads.
use safetensors::tensor::{Dtype, Metadata};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Seek, SeekFrom},
};
use tesslot_core::SpecialTokens;
use tesslot_dataset::TokenizerIdentity;
/// User-supplied sharded weights and append-only tokenizer conversion.
pub mod pretrained;
/// Qwen3 tensor schema and deterministic conversion.
pub mod qwen;

/// Artifact errors preserve file/contract context without executing model code.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Malformed or incompatible artifact.
    #[error("{0}")]
    Invalid(String),
    /// Stream failure.
    #[error("artifact I/O: {0}")]
    Io(#[from] std::io::Error),
}
fn invalid(message: impl ToString) -> Error {
    Error::Invalid(message.to_string())
}
/// Actual tokenizer plus its normalized upstream-compatible identity.
pub struct NativeTokenizer {
    tokenizer: tokenizers::Tokenizer,
    identity: TokenizerIdentity,
}
impl NativeTokenizer {
    /// Load local tokenizer bytes, disable truncation/padding, and validate dedicated controls.
    pub fn from_bytes(bytes: &[u8], bos: &str, eos: &str, noop: &str) -> Result<Self, Error> {
        if bytes.len() > 64 * 1024 * 1024 {
            return Err(invalid("tokenizer exceeds 64 MiB limit"));
        }
        let mut tokenizer = tokenizers::Tokenizer::from_bytes(bytes).map_err(invalid)?;
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(invalid)?;
        let mut ids = Vec::new();
        for text in [bos, eos, noop] {
            let id = tokenizer
                .token_to_id(text)
                .ok_or_else(|| invalid(format!("missing dedicated control {text}")))?;
            if tokenizer.encode(text, false).map_err(invalid)?.get_ids() != [id] {
                return Err(invalid(format!(
                    "control {text} does not encode as one token"
                )));
            }
            ids.push(id);
        }
        let serialized = tokenizer.to_string(false).map_err(invalid)?;
        let identity = TokenizerIdentity {
            sha256: format!("{:x}", Sha256::digest(serialized.as_bytes())),
            vocab_size: tokenizer.get_vocab_size(true),
            special_tokens: SpecialTokens {
                bos: ids[0],
                eos: ids[1],
                noop: ids[2],
            },
        };
        identity.validate().map_err(invalid)?;
        Ok(Self {
            tokenizer,
            identity,
        })
    }
    /// Exact normalized fingerprint, vocabulary and native controls.
    pub fn identity(&self) -> &TokenizerIdentity {
        &self.identity
    }
    /// Encode without automatic added tokens, matching Remix compilation.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(invalid)?
            .get_ids()
            .to_vec())
    }
}
/// A tensor's validated source metadata; dimensions are never guessed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Native SafeTensors dtype label.
    pub dtype: String,
    /// Source tensor dimensions.
    pub shape: Vec<usize>,
    /// Offsets relative to payload start.
    pub offsets: (usize, usize),
}
/// Header-only inspection, independent of tensor backends or total weight size.
#[derive(Debug, Serialize)]
pub struct Inventory {
    /// Exact hash of the encoded header, not of the weight payload.
    pub header_sha256: String,
    /// Original SafeTensors string metadata, retained without reinterpretation.
    pub metadata: BTreeMap<String, String>,
    /// Absolute start of data payload.
    pub data_start: u64,
    /// Entire file length validated against the header's declared ranges.
    pub file_bytes: u64,
    /// Actual names, shapes, dtypes and ranges from the artifact.
    pub tensors: BTreeMap<String, TensorInfo>,
}
/// Read at most 16 MiB of header; verify contiguous offsets and full file length.
pub fn inspect(reader: &mut (impl Read + Seek)) -> Result<Inventory, Error> {
    let size = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(0))?;
    let mut length = [0; 8];
    reader.read_exact(&mut length)?;
    let n = u64::from_le_bytes(length);
    if n > 16 * 1024 * 1024 || n > size.saturating_sub(8) {
        return Err(invalid(
            "SafeTensors header length exceeds file or 16 MiB limit",
        ));
    }
    let mut header = vec![0; n as usize];
    reader.read_exact(&mut header)?;
    // Reject duplicate top-level names before the library's HashMap deserialization.
    let mut de = serde_json::Deserializer::from_slice(&header);
    serde::de::Deserializer::deserialize_map(&mut de, UniqueNames).map_err(invalid)?;
    let metadata: Metadata = serde_json::from_slice(&header).map_err(invalid)?;
    let end = (metadata.data_len() as u64)
        .checked_add(n + 8)
        .ok_or_else(|| invalid("payload length overflow"))?;
    if end != size {
        return Err(invalid(
            "SafeTensors payload is truncated or contains trailing bytes",
        ));
    }
    Ok(Inventory {
        header_sha256: format!("{:x}", Sha256::digest(&header)),
        metadata: metadata
            .metadata()
            .clone()
            .unwrap_or_default()
            .into_iter()
            .collect(),
        data_start: n + 8,
        file_bytes: size,
        tensors: metadata
            .tensors()
            .into_iter()
            .map(|(n, t)| {
                (
                    n,
                    TensorInfo {
                        dtype: format!("{:?}", t.dtype),
                        shape: t.shape.clone(),
                        offsets: t.data_offsets,
                    },
                )
            })
            .collect(),
    })
}
struct UniqueNames;
impl<'de> serde::de::Visitor<'de> for UniqueNames {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "unique SafeTensors names")
    }
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        let mut names = BTreeSet::new();
        while let Some((name, _)) = map.next_entry::<String, serde::de::IgnoredAny>()? {
            if !names.insert(name) {
                return Err(serde::de::Error::custom("duplicate SafeTensors name"));
            }
        }
        Ok(())
    }
}
/// Read one tensor into f32 with an explicit byte limit; no unsafe mmap or whole-file allocation.
/// The inventory is re-read to prevent callers substituting unchecked offsets.
pub fn read_f32(
    reader: &mut (impl Read + Seek),
    name: &str,
    max_bytes: usize,
) -> Result<(Vec<usize>, Vec<f32>), Error> {
    let inventory = inspect(reader)?;
    let t = inventory
        .tensors
        .get(name)
        .ok_or_else(|| invalid(format!("missing source tensor {name}")))?;
    let bytes = t.offsets.1 - t.offsets.0;
    if bytes > max_bytes {
        return Err(invalid(format!(
            "tensor {name} exceeds byte budget {max_bytes}"
        )));
    }
    if !["F32", "F16", "BF16", "F64"].contains(&t.dtype.as_str()) {
        return Err(invalid(format!(
            "unsupported float dtype {} for {name}",
            t.dtype
        )));
    }
    reader.seek(SeekFrom::Start(inventory.data_start + t.offsets.0 as u64))?;
    let mut buffer = vec![0; bytes];
    reader.read_exact(&mut buffer)?;
    let width = match t.dtype.as_str() {
        "F64" => 8,
        "F32" => 4,
        _ => 2,
    };
    let values: Vec<f32> = buffer
        .chunks_exact(width)
        .map(|b| match t.dtype.as_str() {
            "F32" => f32::from_le_bytes(b.try_into().expect("validated width")),
            "F64" => f64::from_le_bytes(b.try_into().expect("validated width")) as f32,
            "F16" => {
                half::f16::from_bits(u16::from_le_bytes(b.try_into().expect("validated width")))
                    .to_f32()
            }
            _ => half::bf16::from_bits(u16::from_le_bytes(b.try_into().expect("validated width")))
                .to_f32(),
        })
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid(format!(
            "tensor {name} contains nonfinite or out-of-f32-range values"
        )));
    }
    Ok((t.shape.clone(), values))
}
/// Explicit destination-to-source mapping with expected destination shapes.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mapping {
    /// Destination names and shapes; tied tensors require explicit aliases in source names.
    pub expected: BTreeMap<String, Vec<usize>>,
    /// Destination to source name; no implicit tensor-name heuristics.
    pub sources: BTreeMap<String, String>,
    /// Destinations deliberately initialized separately by the model converter.
    pub initialize: BTreeSet<String>,
}
/// Complete non-lossy conversion preflight; a report is never permission to discard weights.
#[derive(Debug, Serialize)]
pub struct Report {
    /// Shape/dtype-compatible destination to source mapping.
    pub compatible: BTreeMap<String, String>,
    /// Destinations/sources missing from the plan or file.
    pub missing: Vec<String>,
    /// Unmapped source tensors or unknown destination plan entries.
    pub unexpected: Vec<String>,
    /// Shape, dtype or conflicting mapping/initialization errors.
    pub incompatible: Vec<String>,
    /// Explicitly separate-initialized destination names.
    pub initialize: BTreeSet<String>,
}
impl Report {
    /// Safe to proceed only when no unexplained tensors or errors remain.
    pub fn is_valid(&self) -> bool {
        self.missing.is_empty() && self.unexpected.is_empty() && self.incompatible.is_empty()
    }
}
/// Check an explicit mapping against actual artifact names and shapes without reading weights.
pub fn plan(inventory: &Inventory, mapping: &Mapping) -> Report {
    let mut r = Report {
        compatible: BTreeMap::new(),
        missing: Vec::new(),
        unexpected: Vec::new(),
        incompatible: Vec::new(),
        initialize: mapping.initialize.clone(),
    };
    let mut used = BTreeSet::new();
    for name in mapping.sources.keys().chain(mapping.initialize.iter()) {
        if !mapping.expected.contains_key(name) {
            r.unexpected.push(format!("destination {name}"));
        }
    }
    for (name, shape) in &mapping.expected {
        match (mapping.sources.get(name), mapping.initialize.contains(name)) {
            (Some(_), true) => r
                .incompatible
                .push(format!("{name}: both mapped and initialized")),
            (None, false) => r.missing.push(format!("destination {name}")),
            (None, true) => {}
            (Some(source), false) => {
                used.insert(source.clone());
                match inventory.tensors.get(source) {
                    None => r.missing.push(format!("source {source} for {name}")),
                    Some(t) => {
                        if &t.shape != shape
                            || !["F32", "F16", "BF16", "F64"].contains(&t.dtype.as_str())
                        {
                            r.incompatible.push(format!(
                                "{source} -> {name}: expected {shape:?} float tensor, got {:?} {}",
                                t.shape, t.dtype
                            ));
                        } else {
                            r.compatible.insert(name.clone(), source.clone());
                        }
                    }
                }
            }
        }
    }
    for source in inventory.tensors.keys() {
        if !used.contains(source) {
            r.unexpected.push(format!("source {source}"));
        }
    }
    r
}
/// Serialize tiny f32 tensor fixtures/artifacts using the upstream SafeTensors codec.
pub fn encode_f32(tensors: &BTreeMap<String, (Vec<usize>, Vec<f32>)>) -> Result<Vec<u8>, Error> {
    encode_f32_with_metadata(tensors, BTreeMap::new())
}
/// Serialize f32 tensors and explicit source/conversion metadata with the upstream codec.
pub fn encode_f32_with_metadata(
    tensors: &Tensors,
    metadata: BTreeMap<String, String>,
) -> Result<Vec<u8>, Error> {
    let bytes: BTreeMap<_, _> = tensors
        .iter()
        .map(|(name, (_, v))| {
            (
                name,
                v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
            )
        })
        .collect();
    let views = tensors
        .iter()
        .map(|(name, (shape, _))| {
            safetensors::tensor::TensorView::new(Dtype::F32, shape.clone(), &bytes[name])
                .map(|v| (name.as_str(), v))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(invalid)?;
    // The upstream metadata argument is a randomly ordered HashMap. Keep its tensor
    // codec/layout, but prepend sorted metadata so fixture/checkpoint bytes are stable.
    let encoded = safetensors::tensor::serialize(views, None).map_err(invalid)?;
    let header_len = u64::from_le_bytes(encoded[..8].try_into().map_err(invalid)?) as usize;
    let tensor_header = std::str::from_utf8(&encoded[8..8 + header_len])
        .map_err(invalid)?
        .trim_end();
    let mut header = format!(
        "{{\"__metadata__\":{}",
        serde_json::to_string(&metadata).map_err(invalid)?
    );
    if tensor_header != "{}" {
        header.push(',');
        header.push_str(&tensor_header[1..tensor_header.len() - 1]);
    }
    header.push('}');
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut output = (header.len() as u64).to_le_bytes().to_vec();
    output.extend_from_slice(header.as_bytes());
    output.extend_from_slice(&encoded[8 + header_len..]);
    Ok(output)
}

/// Imported output-major tensors in host f32. Seeded new tensors must be supplied explicitly.
pub type Tensors = BTreeMap<String, (Vec<usize>, Vec<f32>)>;
/// Apply a fully valid mapping, reading only named payload ranges within an aggregate budget.
/// Initialization values are validated; nothing is implicitly dropped or randomly initialized.
pub fn import(
    reader: &mut (impl Read + Seek),
    mapping: &Mapping,
    initialized: &Tensors,
    max_bytes: usize,
) -> Result<Tensors, Error> {
    let inventory = inspect(reader)?;
    let report = plan(&inventory, mapping);
    if !report.is_valid() {
        return Err(invalid(serde_json::to_string(&report).map_err(invalid)?));
    }
    if initialized.keys().cloned().collect::<BTreeSet<_>>() != mapping.initialize {
        return Err(invalid(
            "initialization tensors must match the explicit initialization set",
        ));
    }
    let mut remaining = max_bytes;
    let mut result = BTreeMap::new();
    for (name, shape) in &mapping.expected {
        let elements = shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| invalid("shape product overflow"))?;
        let cost = elements
            .checked_mul(4)
            .ok_or_else(|| invalid("f32 allocation size overflow"))?;
        remaining = remaining
            .checked_sub(cost)
            .ok_or_else(|| invalid("import exceeds aggregate f32 byte budget"))?;
        let tensor = if let Some(source) = mapping.sources.get(name) {
            // Source F64 can require twice the f32 output budget for this individual read.
            let source_bytes =
                inventory.tensors[source].offsets.1 - inventory.tensors[source].offsets.0;
            if source_bytes > max_bytes {
                return Err(invalid("source tensor exceeds byte budget"));
            }
            read_f32(reader, source, max_bytes)?
        } else {
            initialized[name].clone()
        };
        if &tensor.0 != shape
            || tensor.1.len() != elements
            || tensor.1.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(format!(
                "invalid imported/initialized tensor {name}"
            )));
        }
        result.insert(name.clone(), tensor);
    }
    Ok(result)
}
