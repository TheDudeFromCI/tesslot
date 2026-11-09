//! Bounded index and shard-header inspection; payloads are never loaded.
use super::{Error, Inventory, TensorInfo, UniqueNames, inspect, invalid};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Read,
    path::{Component, Path},
};

/// Source identity for a single shard, excluding payload checksums.
#[derive(Debug, Serialize)]
pub struct Shard {
    /// Exact SHA-256 of its SafeTensors header.
    pub header_sha256: String,
    /// Validated complete file length.
    pub file_bytes: u64,
}
/// Verified index-to-header inventory, not a verified tensor payload.
#[derive(Debug, Serialize)]
pub struct Directory {
    /// Independently versioned report format.
    pub format: String,
    /// Full index-file SHA-256, absent for an unsharded source.
    pub index_sha256: Option<String>,
    /// File containing each tensor, validated against actual headers.
    pub weight_map: BTreeMap<String, String>,
    /// Actual tensor names, dimensions, dtypes and shard-relative offsets.
    pub tensors: BTreeMap<String, TensorInfo>,
    /// Per-file header identities and lengths.
    pub shards: BTreeMap<String, Shard>,
    /// Number of source scalar values, counting physically duplicated tensors.
    pub stored_parameters: u64,
    /// Sum of payload bytes, excluding headers.
    pub payload_bytes: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Index {
    #[serde(default)]
    metadata: BTreeMap<String, serde_json::Value>,
    #[serde(deserialize_with = "unique_weight_map")]
    weight_map: BTreeMap<String, String>,
}
fn unique_weight_map<'de, D: serde::Deserializer<'de>>(
    de: D,
) -> Result<BTreeMap<String, String>, D::Error> {
    struct Visitor;
    impl<'de> serde::de::Visitor<'de> for Visitor {
        type Value = BTreeMap<String, String>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "unique tensor to shard mapping")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut input: A,
        ) -> Result<Self::Value, A::Error> {
            let mut map = BTreeMap::new();
            while let Some((key, value)) = input.next_entry::<String, String>()? {
                if map.insert(key, value).is_some() {
                    return Err(serde::de::Error::custom("duplicate indexed tensor"));
                }
            }
            Ok(map)
        }
    }
    de.deserialize_map(Visitor)
}
fn regular(path: &Path) -> Result<File, Error> {
    if !fs::symlink_metadata(path)?.file_type().is_file() {
        return Err(invalid("model sources must be regular files, not symlinks"));
    }
    Ok(File::open(path)?)
}
fn shard_name(name: &str) -> bool {
    let mut parts = Path::new(name).components();
    !name.contains(['/', '\\', ':'])
        && name.ends_with(".safetensors")
        && matches!(parts.next(), Some(Component::Normal(_)))
        && parts.next().is_none()
}
/// Inspect `model.safetensors` or `model.safetensors.index.json` and all its shards.
/// Reject ambiguous layouts, unsafe paths, missing/extra tensors and unindexed shards.
/// Limits: 16 MiB index, 512 shards, 200,000 indexed tensors, 16 MiB per header.
pub fn inspect_directory(root: &Path) -> Result<Directory, Error> {
    let index_path = root.join("model.safetensors.index.json");
    let single_path = root.join("model.safetensors");
    let (index, index_sha256) = if index_path.try_exists()? {
        if single_path.try_exists()? {
            return Err(invalid(
                "ambiguous model directory contains both single-file weights and a shard index",
            ));
        }
        let mut bytes = Vec::new();
        regular(&index_path)?
            .take(16 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 16 * 1024 * 1024 {
            return Err(invalid("model index exceeds 16 MiB limit"));
        }
        let mut de = serde_json::Deserializer::from_slice(&bytes);
        serde::de::Deserializer::deserialize_map(&mut de, UniqueNames).map_err(invalid)?;
        let index: Index = serde_json::from_slice(&bytes).map_err(invalid)?;
        if index.weight_map.is_empty() || index.weight_map.len() > 200_000 {
            return Err(invalid("invalid indexed tensor count"));
        }
        (Some(index), Some(format!("{:x}", Sha256::digest(&bytes))))
    } else {
        (None, None)
    };
    let files: BTreeSet<_> = match &index {
        Some(index) => index.weight_map.values().cloned().collect(),
        None => BTreeSet::from(["model.safetensors".into()]),
    };
    if files.len() > 512 || files.iter().any(|name| !shard_name(name)) {
        return Err(invalid("invalid shard path or shard count"));
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.path().extension().is_some_and(|s| s == "safetensors")
            && !files.contains(&entry.file_name().to_string_lossy().into_owned())
        {
            return Err(invalid(
                "unindexed SafeTensors file in model directory; use an unambiguous source directory",
            ));
        }
    }
    let mut result = Directory {
        format: "tesslot-model-directory/1".into(),
        index_sha256,
        weight_map: BTreeMap::new(),
        tensors: BTreeMap::new(),
        shards: BTreeMap::new(),
        stored_parameters: 0,
        payload_bytes: 0,
    };
    for name in files {
        let Inventory {
            header_sha256,
            file_bytes,
            tensors,
            ..
        } = inspect(&mut regular(&root.join(&name))?)?;
        result.shards.insert(
            name.clone(),
            Shard {
                header_sha256,
                file_bytes,
            },
        );
        for (tensor_name, tensor) in tensors {
            if index
                .as_ref()
                .is_some_and(|index| index.weight_map.get(&tensor_name) != Some(&name))
            {
                return Err(invalid(format!(
                    "index/header shard mismatch for {tensor_name}"
                )));
            }
            if result
                .weight_map
                .insert(tensor_name.clone(), name.clone())
                .is_some()
            {
                return Err(invalid("duplicate tensor across shards"));
            }
            let count = tensor.shape.iter().try_fold(1u64, |n, d| {
                n.checked_mul(*d as u64)
                    .ok_or_else(|| invalid("parameter count overflow"))
            })?;
            result.stored_parameters = result
                .stored_parameters
                .checked_add(count)
                .ok_or_else(|| invalid("parameter count overflow"))?;
            result.payload_bytes = result
                .payload_bytes
                .checked_add((tensor.offsets.1 - tensor.offsets.0) as u64)
                .ok_or_else(|| invalid("payload size overflow"))?;
            result.tensors.insert(tensor_name, tensor);
        }
    }
    if let Some(index) = index {
        if index.weight_map != result.weight_map {
            return Err(invalid("indexed tensor missing from shard headers"));
        }
        if let Some(size) = index.metadata.get("total_size")
            && size.as_u64() != Some(result.payload_bytes)
        {
            return Err(invalid(
                "index total_size differs from actual tensor payload bytes",
            ));
        }
    }
    Ok(result)
}
