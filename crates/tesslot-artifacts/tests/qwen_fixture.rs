//! A real checked-in synthetic Qwen-format file exercises load, conversion and serialization.
use std::{collections::BTreeMap, io::Cursor};
use tesslot_artifacts::{qwen::*, *};
use tesslot_model::qwen::Config;
const CONFIG: &[u8] = include_bytes!("../../../fixtures/tiny-qwen/config.json");
const WEIGHTS: &[u8] = include_bytes!("../../../fixtures/tiny-qwen/model.safetensors");
fn config() -> Config {
    Config::from_json(CONFIG).unwrap()
}
#[test]
fn fixture_matches_qwen_schema_and_is_reproducible() {
    assert!(WEIGHTS.len() < 32 * 1024, "demo fixture must remain small");
    let c = config();
    assert_eq!(c.query_width().unwrap(), 32);
    assert_eq!(c.hidden_size, 16);
    assert_eq!(c.kv_width().unwrap(), 16);
    let inventory = inspect(&mut Cursor::new(WEIGHTS)).unwrap();
    let shapes = source_shapes(&c).unwrap();
    assert_eq!(shapes.len(), 13);
    assert_eq!(inventory.tensors.len(), 13);
    let count: usize = shapes
        .iter()
        .map(|(name, shape)| {
            assert_eq!(&inventory.tensors[name].shape, shape);
            shape.iter().product::<usize>()
        })
        .sum();
    assert_eq!(count, 5184);
    assert!(!inventory.tensors.contains_key("lm_head.weight"));
    assert_eq!(inventory.metadata["format"], "pt");
    let generated = random_source(&c, 20261106, 64 * 1024).unwrap();
    let expected = encode_f32_with_metadata(
        &generated,
        BTreeMap::from([
            ("format".into(), "pt".into()),
            (
                "description".into(),
                "Tesslot random unit-test fixture; not pretrained Qwen weights".into(),
            ),
        ]),
    )
    .unwrap();
    assert_eq!(expected, WEIGHTS);
}
#[test]
fn conversion_reuses_every_weight_and_initializes_only_explicit_additions() {
    let c = config();
    let converted = convert(&mut Cursor::new(WEIGHTS), &c, 4, 42, 128 * 1024).unwrap();
    assert!(converted.manifest.report.is_valid());
    assert_eq!(converted.manifest.report.compatible.len(), 13);
    assert_eq!(converted.manifest.report.initialize.len(), 7);
    assert_eq!(
        converted.manifest.aliases["lm_head.weight"],
        "model.embed_tokens.weight"
    );
    assert!(converted.manifest.tokenizer_conversion_required);
    for name in source_shapes(&c).unwrap().keys() {
        assert_eq!(
            converted.tensors[name],
            read_f32(&mut Cursor::new(WEIGHTS), name, 128 * 1024).unwrap()
        );
    }
    assert_eq!(converted.tensors["kidra.exchange.query.weight"].0, [32, 16]);
    assert_eq!(
        converted.tensors["kidra.exchange.identities.weight"].0,
        [4, 16]
    );
    let again = convert(&mut Cursor::new(WEIGHTS), &c, 4, 42, 128 * 1024).unwrap();
    assert_eq!(converted.tensors, again.tensors);
    let changed = convert(&mut Cursor::new(WEIGHTS), &c, 4, 43, 128 * 1024).unwrap();
    assert_ne!(
        converted.tensors["kidra.window_embeddings.weight"],
        changed.tensors["kidra.window_embeddings.weight"]
    );
    assert_eq!(
        converted.tensors["model.embed_tokens.weight"],
        changed.tensors["model.embed_tokens.weight"]
    );
    let payload = encode_f32_with_metadata(
        &converted.tensors,
        BTreeMap::from([("format".into(), FORMAT.into())]),
    )
    .unwrap();
    let inventory = inspect(&mut Cursor::new(&payload)).unwrap();
    assert_eq!(inventory.tensors.len(), 20);
    for (name, tensor) in &converted.tensors {
        assert_eq!(
            read_f32(&mut Cursor::new(&payload), name, 128 * 1024).unwrap(),
            *tensor
        );
    }
    assert!(convert(&mut Cursor::new(WEIGHTS), &c, 4, 42, 16).is_err());
    assert!(convert(&mut Cursor::new(WEIGHTS), &c, 0, 42, 128 * 1024).is_err());
}
#[test]
fn bad_source_shapes_missing_extra_and_nonfinite_weights_fail() {
    let c = config();
    let original = random_source(&c, 20261106, 64 * 1024).unwrap();
    let mut bad = original.clone();
    bad.remove("model.norm.weight");
    assert!(
        convert(
            &mut Cursor::new(encode_f32(&bad).unwrap()),
            &c,
            4,
            42,
            128 * 1024
        )
        .is_err()
    );
    let mut bad = original.clone();
    bad.insert("unexpected.weight".into(), (vec![1], vec![1.0]));
    assert!(
        convert(
            &mut Cursor::new(encode_f32(&bad).unwrap()),
            &c,
            4,
            42,
            128 * 1024
        )
        .is_err()
    );
    let mut bad = original.clone();
    let t = bad
        .get_mut("model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    t.0 = vec![16, 32];
    assert!(
        convert(
            &mut Cursor::new(encode_f32(&bad).unwrap()),
            &c,
            4,
            42,
            128 * 1024
        )
        .is_err()
    );
    let mut bad = original;
    bad.get_mut("model.norm.weight").unwrap().1[0] = f32::NAN;
    assert!(
        convert(
            &mut Cursor::new(encode_f32(&bad).unwrap()),
            &c,
            4,
            42,
            128 * 1024
        )
        .is_err()
    );
}
#[test]
fn untied_output_head_is_explicit_and_preserved() {
    let mut c = config();
    c.tie_word_embeddings = false;
    let tensors = random_source(&c, 1, 64 * 1024).unwrap();
    assert!(tensors.contains_key("lm_head.weight"));
    let bytes = encode_f32(&tensors).unwrap();
    let converted = convert(&mut Cursor::new(bytes), &c, 4, 42, 128 * 1024).unwrap();
    assert!(converted.manifest.aliases.is_empty());
    assert_eq!(
        converted.tensors["lm_head.weight"],
        tensors["lm_head.weight"]
    );
}
