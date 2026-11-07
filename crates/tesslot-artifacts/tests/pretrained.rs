//! Tiny local artifacts cover the same single/sharded contracts as separately supplied base models.
use std::collections::BTreeMap;
use tesslot_artifacts::{encode_f32, pretrained::*, qwen};
use tesslot_model::qwen::Config;
fn config() -> Config {
    Config::from_json(include_bytes!("../../../fixtures/tiny-qwen/config.json")).unwrap()
}
#[test]
fn controls_are_append_only_and_padded_rows_and_tied_heads_survive() {
    let bytes = include_bytes!("../../../fixtures/demo-tokenizer.json");
    let old =
        tesslot_artifacts::NativeTokenizer::from_bytes(bytes, "<|bos|>", "<|eos|>", "<|noop|>")
            .unwrap();
    for tied in [true, false] {
        let mut c = config();
        c.tie_word_embeddings = tied;
        let mut weights = qwen::random_source(&c, 42, 65536).unwrap();
        let before = weights.clone();
        let (target, conversion) = extend_tokenizer(bytes, c.vocab_size).unwrap();
        assert_eq!(conversion.target.vocab_size, 131);
        assert_eq!(conversion.reserved.len(), 128 - old.identity().vocab_size);
        let target = tesslot_artifacts::NativeTokenizer::from_bytes(
            &target,
            "<|kidra_bos|>",
            "<|kidra_eos|>",
            "<|kidra_noop|>",
        )
        .unwrap();
        for text in ["a b c", "<|bos|>", "<|noop|>", "hello world"] {
            assert_eq!(old.encode(text).unwrap(), target.encode(text).unwrap());
        }
        extend_embeddings(&mut weights, &mut c, &conversion, 99, 65536).unwrap();
        for (name, (_, v)) in &before {
            assert_eq!(&weights[name].1[..v.len()], v);
        }
        assert_eq!(weights.contains_key("lm_head.weight"), !tied);
        assert_eq!(c.bos_token_id, 128);
        assert_eq!(c.eos_token_id, 129);
        let mut again = before.clone();
        let mut c2 = config();
        c2.tie_word_embeddings = tied;
        extend_embeddings(&mut again, &mut c2, &conversion, 99, 65536).unwrap();
        assert_eq!(weights, again);
        let mut c2 = config();
        c2.tie_word_embeddings = tied;
        assert!(extend_embeddings(&mut before.clone(), &mut c2, &conversion, 1, 1).is_err());
    }
}
#[test]
fn sharded_and_single_files_have_exact_schema_and_budget_checks() {
    let dir = tempfile::tempdir().unwrap();
    let c = config();
    let weights = qwen::random_source(&c, 42, 65536).unwrap();
    std::fs::write(
        dir.path().join("model.safetensors"),
        encode_f32(&weights).unwrap(),
    )
    .unwrap();
    assert_eq!(load(dir.path(), &c, 65536).unwrap().0, weights);
    assert!(load(dir.path(), &c, 1).is_err());
    let mut wrong = c.clone();
    wrong.hidden_size = 17;
    assert!(
        load(dir.path(), &wrong, 65536)
            .unwrap_err()
            .to_string()
            .contains("shape")
    );
    let mut map = BTreeMap::new();
    for (i, (name, value)) in weights.iter().enumerate() {
        let shard = format!("part-{i}.safetensors");
        std::fs::write(
            dir.path().join(&shard),
            encode_f32(&BTreeMap::from([(name.clone(), value.clone())])).unwrap(),
        )
        .unwrap();
        map.insert(name.clone(), shard);
    }
    let index = dir.path().join("model.safetensors.index.json");
    let write = |map: &BTreeMap<String, String>| {
        std::fs::write(
            &index,
            serde_json::to_vec(
                &serde_json::json!({"metadata":{"total_size":20736},"weight_map":map}),
            )
            .unwrap(),
        )
        .unwrap()
    };
    write(&map);
    let (actual, p) = load(dir.path(), &c, 65536).unwrap();
    assert_eq!(actual, weights);
    assert_eq!(p.files.len(), 14);
    std::fs::write(
        &index,
        serde_json::to_vec(&serde_json::json!({"metadata":{"total_size":1},"weight_map":map}))
            .unwrap(),
    )
    .unwrap();
    assert!(
        load(dir.path(), &c, 65536)
            .unwrap_err()
            .to_string()
            .contains("total_size")
    );
    write(&map);
    let first = map.keys().next().unwrap().clone();
    let mut bad = map.clone();
    bad.insert(first.clone(), "../model.safetensors".into());
    write(&bad);
    assert!(load(dir.path(), &c, 65536).is_err());
    let mut bad = map.clone();
    bad.remove(&first);
    write(&bad);
    assert!(load(dir.path(), &c, 65536).is_err());
    let mut bad = map.clone();
    bad.insert(first, "part-1.safetensors".into());
    write(&bad);
    assert!(load(dir.path(), &c, 65536).is_err());
    std::fs::write(
        &index,
        r#"{"weight_map":{"x":"a.safetensors","x":"b.safetensors"}}"#,
    )
    .unwrap();
    assert!(
        load(dir.path(), &c, 65536)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    write(&map);
    std::fs::remove_file(dir.path().join(map.values().next().unwrap())).unwrap();
    assert!(load(dir.path(), &c, 65536).is_err());
}
#[cfg(unix)]
#[test]
fn shards_cannot_escape_via_symlink() {
    let dir = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    std::fs::write(
        other.path().join("outside.safetensors"),
        include_bytes!("../../../fixtures/tiny-qwen/model.safetensors"),
    )
    .unwrap();
    std::os::unix::fs::symlink(
        other.path().join("outside.safetensors"),
        dir.path().join("model.safetensors"),
    )
    .unwrap();
    assert!(
        load(dir.path(), &config(), 65536)
            .unwrap_err()
            .to_string()
            .contains("escapes")
    );
}
