//! Regenerate only the tiny random fixture; never downloads a model.
use std::{collections::BTreeMap, path::PathBuf};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tiny-qwen");
    let config = tesslot_model::qwen::Config::from_json(&std::fs::read(root.join("config.json"))?)
        .map_err(std::io::Error::other)?;
    let tensors = tesslot_artifacts::qwen::random_source(&config, 20261106, 64 * 1024)?;
    let count: usize = tensors.values().map(|(_, v)| v.len()).sum();
    assert_eq!(count, 5184);
    let bytes = tesslot_artifacts::encode_f32_with_metadata(
        &tensors,
        BTreeMap::from([
            ("format".into(), "pt".into()),
            (
                "description".into(),
                "Tesslot random unit-test fixture; not pretrained Qwen weights".into(),
            ),
        ]),
    )?;
    std::fs::write(root.join("model.safetensors"), bytes)?;
    println!("generated {count} synthetic parameters");
    Ok(())
}
