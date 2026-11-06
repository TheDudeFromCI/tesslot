//! Actual native tokenizer and upstream-codec artifact round trips and rejection cases.
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Cursor,
};
use tesslot_artifacts::*;
fn data() -> Vec<u8> {
    encode_f32(&BTreeMap::from([(
        "source.weight".into(),
        (vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
    )]))
    .unwrap()
}
#[test]
fn tokenizer_matches_actual_exporter_identity() {
    let tokenizer = NativeTokenizer::from_bytes(
        include_bytes!("../../../fixtures/demo-tokenizer.json"),
        "<|bos|>",
        "<|eos|>",
        "<|noop|>",
    )
    .unwrap();
    let expected: tesslot_dataset::TokenizerIdentity =
        serde_json::from_slice(include_bytes!("../../../fixtures/tokenizer-identity.json"))
            .unwrap();
    assert_eq!(tokenizer.identity().sha256, expected.sha256);
    assert_eq!(tokenizer.identity().special_tokens, expected.special_tokens);
    assert_eq!(tokenizer.identity().vocab_size, expected.vocab_size);
    assert!(!tokenizer.encode("hello").unwrap().is_empty());
    assert!(
        NativeTokenizer::from_bytes(
            include_bytes!("../../../fixtures/demo-tokenizer.json"),
            "missing",
            "<|eos|>",
            "<|noop|>"
        )
        .is_err()
    );
    assert!(
        NativeTokenizer::from_bytes(
            include_bytes!("../../../fixtures/demo-tokenizer.json"),
            "<|eos|>",
            "<|eos|>",
            "<|noop|>"
        )
        .is_err()
    );
}
#[test]
fn header_only_inspection_and_bounded_f32_payload() {
    let bytes = data();
    let inventory = inspect(&mut Cursor::new(&bytes)).unwrap();
    assert_eq!(inventory.tensors["source.weight"].shape, [2, 2]);
    assert_eq!(inventory.tensors["source.weight"].dtype, "F32");
    assert_eq!(
        read_f32(&mut Cursor::new(&bytes), "source.weight", 16).unwrap(),
        (vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
    );
    assert!(read_f32(&mut Cursor::new(&bytes), "source.weight", 15).is_err());
    for bad in [
        bytes[..bytes.len() - 1].to_vec(),
        [bytes.clone(), vec![0]].concat(),
        u64::MAX.to_le_bytes().to_vec(),
    ] {
        assert!(inspect(&mut Cursor::new(bad)).is_err());
    }
    // A header claims a gigabyte payload: inspection must not read/allocate it.
    struct HeaderOnly {
        header: Cursor<Vec<u8>>,
        size: u64,
    }
    impl std::io::Read for HeaderOnly {
        fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
            assert!(self.header.position() < self.header.get_ref().len() as u64);
            std::io::Read::read(&mut self.header, b)
        }
    }
    impl std::io::Seek for HeaderOnly {
        fn seek(&mut self, p: std::io::SeekFrom) -> std::io::Result<u64> {
            if p == std::io::SeekFrom::End(0) {
                Ok(self.size)
            } else {
                std::io::Seek::seek(&mut self.header, p)
            }
        }
    }
    let raw = br#"{"huge":{"dtype":"F32","shape":[268435456],"data_offsets":[0,1073741824]}}"#;
    let mut h = HeaderOnly {
        header: Cursor::new([&(raw.len() as u64).to_le_bytes()[..], raw].concat()),
        size: 8 + raw.len() as u64 + 1073741824,
    };
    assert_eq!(inspect(&mut h).unwrap().tensors["huge"].shape, [268435456]);
}
#[test]
fn explicit_mapping_never_silently_discards_tensors() {
    let inventory = inspect(&mut Cursor::new(data())).unwrap();
    let mut mapping = Mapping {
        expected: BTreeMap::from([
            ("model.weight".into(), vec![2, 2]),
            ("window.embedding".into(), vec![4, 2]),
        ]),
        sources: BTreeMap::from([("model.weight".into(), "source.weight".into())]),
        initialize: BTreeSet::from(["window.embedding".into()]),
    };
    assert!(plan(&inventory, &mapping).is_valid());
    let initial = BTreeMap::from([("window.embedding".into(), (vec![4, 2], vec![0.125; 8]))]);
    let converted = import(&mut Cursor::new(data()), &mapping, &initial, 48).unwrap();
    assert_eq!(converted["model.weight"].1, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(converted["window.embedding"].1, [0.125; 8]);
    assert!(import(&mut Cursor::new(data()), &mapping, &initial, 47).is_err());
    assert!(import(&mut Cursor::new(data()), &mapping, &BTreeMap::new(), 48).is_err());
    mapping.expected.get_mut("model.weight").unwrap()[0] = 3;
    assert!(!plan(&inventory, &mapping).incompatible.is_empty());
    mapping.sources.clear();
    let r = plan(&inventory, &mapping);
    assert!(!r.missing.is_empty());
    assert!(!r.unexpected.is_empty());
}
#[test]
fn half_precision_decode_and_invalid_header_rejection() {
    use safetensors::tensor::{Dtype, TensorView, serialize};
    let payload = 1.0f32.to_le_bytes();
    let file = serialize(
        [(
            "meta",
            TensorView::new(Dtype::F32, vec![1], &payload).unwrap(),
        )],
        Some(std::collections::HashMap::from([(
            "architecture".into(),
            "example/1".into(),
        )])),
    )
    .unwrap();
    assert_eq!(
        inspect(&mut Cursor::new(file)).unwrap().metadata["architecture"],
        "example/1"
    );
    for (dtype, bytes) in [
        (
            Dtype::BF16,
            half::bf16::from_f32(1.5).to_bits().to_le_bytes(),
        ),
        (Dtype::F16, half::f16::from_f32(1.5).to_bits().to_le_bytes()),
    ] {
        let view = TensorView::new(dtype, vec![1], &bytes).unwrap();
        let file = serialize([("x", view)], None).unwrap();
        assert_eq!(read_f32(&mut Cursor::new(file), "x", 2).unwrap().1, [1.5]);
    }
    let bytes = encode_f32(&BTreeMap::from([("x".into(), (vec![1], vec![f32::NAN]))])).unwrap();
    assert!(read_f32(&mut Cursor::new(bytes), "x", 4).is_err());
    for header in [
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
    ] {
        let bytes = [
            &(header.len() as u64).to_le_bytes()[..],
            header.as_bytes(),
            &[0; 4],
        ]
        .concat();
        assert!(inspect(&mut Cursor::new(bytes)).is_err());
    }
}
