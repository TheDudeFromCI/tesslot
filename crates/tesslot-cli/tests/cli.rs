//! Test the actual CLI argument boundary, including global config/name collisions.
#[test]
fn qwen_inspection_does_not_read_model_metadata_as_server_config() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/qwen3-0.6b.config.json");
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_tesslot"))
        .arg("inspect-qwen-config")
        .arg(path)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(v["hidden_size"], 1024);
    assert_eq!(v["query_width"], 2048);
    assert_eq!(v["weight_loading"], "unimplemented");
}
