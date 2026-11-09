//! Real index/shard metadata validation without model-sized payloads.
use std::{collections::BTreeMap, fs};
use tesslot_artifacts::{directory::inspect_directory, encode_f32};
fn shard(path: &std::path::Path, name: &str) {
    fs::write(
        path,
        encode_f32(&BTreeMap::from([(name.into(), (vec![2], vec![1.0, 2.0]))])).unwrap(),
    )
    .unwrap();
}
fn index(root: &std::path::Path, map: serde_json::Value, total: u64) {
    fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"metadata":{"total_size":total},"weight_map":map}))
            .unwrap(),
    )
    .unwrap();
}
#[test]
fn single_and_sharded_headers_preserve_source_identity() {
    let folder = tempfile::tempdir().unwrap();
    shard(&folder.path().join("model.safetensors"), "one");
    let r = inspect_directory(folder.path()).unwrap();
    assert_eq!(r.stored_parameters, 2);
    assert_eq!(r.payload_bytes, 8);
    assert!(r.index_sha256.is_none());
    fs::rename(
        folder.path().join("model.safetensors"),
        folder.path().join("a.safetensors"),
    )
    .unwrap();
    shard(&folder.path().join("b.safetensors"), "two");
    index(
        folder.path(),
        serde_json::json!({"one":"a.safetensors","two":"b.safetensors"}),
        16,
    );
    let r = inspect_directory(folder.path()).unwrap();
    assert_eq!(r.stored_parameters, 4);
    assert_eq!(r.payload_bytes, 16);
    assert_eq!(r.shards.len(), 2);
    assert_eq!(r.index_sha256.unwrap().len(), 64);
    assert_eq!(r.weight_map["two"], "b.safetensors");
    assert!(r.shards.values().all(|s| s.header_sha256.len() == 64));
}
#[test]
fn invalid_index_coverage_paths_duplicates_and_ambiguous_files_fail() {
    let folder = tempfile::tempdir().unwrap();
    shard(&folder.path().join("a.safetensors"), "one");
    for (map, total) in [
        (serde_json::json!({"missing":"a.safetensors"}), 8),
        (serde_json::json!({"one":"../a.safetensors"}), 8),
        (serde_json::json!({"one":"C:\\a.safetensors"}), 8),
        (serde_json::json!({"one":"missing.safetensors"}), 8),
        (serde_json::json!({"one":"a.safetensors"}), 99),
    ] {
        index(folder.path(), map, total);
        assert!(inspect_directory(folder.path()).is_err());
    }
    fs::write(
        folder.path().join("model.safetensors.index.json"),
        br#"{"weight_map":{"one":"a.safetensors","one":"a.safetensors"}}"#,
    )
    .unwrap();
    assert!(
        inspect_directory(folder.path())
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    index(
        folder.path(),
        serde_json::json!({"one":"a.safetensors","missing":"a.safetensors"}),
        8,
    );
    assert!(
        inspect_directory(folder.path())
            .unwrap_err()
            .to_string()
            .contains("missing")
    );
    index(folder.path(), serde_json::json!({"one":"a.safetensors"}), 8);
    shard(&folder.path().join("extra.safetensors"), "extra");
    assert!(
        inspect_directory(folder.path())
            .unwrap_err()
            .to_string()
            .contains("unindexed")
    );
    fs::rename(
        folder.path().join("extra.safetensors"),
        folder.path().join("model.safetensors"),
    )
    .unwrap();
    assert!(
        inspect_directory(folder.path())
            .unwrap_err()
            .to_string()
            .contains("ambiguous")
    );
}
#[cfg(unix)]
#[test]
fn symlink_shards_are_rejected() {
    let folder = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    shard(&outside.path().join("source.safetensors"), "one");
    std::os::unix::fs::symlink(
        outside.path().join("source.safetensors"),
        folder.path().join("model.safetensors"),
    )
    .unwrap();
    assert!(
        inspect_directory(folder.path())
            .unwrap_err()
            .to_string()
            .contains("symlink")
    );
}
