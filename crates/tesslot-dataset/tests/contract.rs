//! Tests against canonical upstream exports and adversarial mutations.
use std::io::Cursor;
use tesslot_core::visible_to;
use tesslot_dataset::{Limits, Reader, Sample, TokenizerIdentity};

const DATA: &str = include_str!("../../../fixtures/canonical.jsonl");
fn identity() -> TokenizerIdentity {
    serde_json::from_str(include_str!("../../../fixtures/tokenizer-identity.json")).unwrap()
}
fn read(data: &str) -> Result<Vec<Sample>, tesslot_dataset::Error> {
    Reader::new(Cursor::new(data), identity(), Limits::default())?.collect()
}
fn records() -> Vec<serde_json::Value> {
    DATA.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}
fn encode(records: &[serde_json::Value]) -> String {
    records.iter().map(|v| format!("{v}\n")).collect()
}

#[test]
fn canonical_reconstruction_and_all_upstream_visibility_pairs() {
    let sample = read(DATA).unwrap().remove(0);
    let single = read(include_str!("../../../fixtures/canonical-single.jsonl"))
        .unwrap()
        .remove(0);
    assert_eq!(sample.rows, single.rows);
    assert_eq!(sample.chunk_ends.len(), 25);
    assert_eq!(sample.rows.len(), 429);
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("../../../fixtures/visibility.json")).unwrap();
    assert_eq!(oracle["row_count"], sample.rows.len());
    for (i, q) in sample.rows.iter().enumerate() {
        for (j, k) in sample.rows.iter().enumerate() {
            assert_eq!(
                visible_to(q, k, &sample.metadata.windows),
                {
                    let hex = oracle["allowed_lsb0_hex"][i].as_str().unwrap();
                    let byte = u8::from_str_radix(&hex[(j / 8) * 2..(j / 8) * 2 + 2], 16).unwrap();
                    byte & (1 << (j % 8)) != 0
                },
                "query {i}, key {j}"
            );
        }
    }
    for group in sample.ticks() {
        assert!(group.iter().all(|r| r.tick == group[0].tick));
    }
    assert!(sample.rows.iter().any(|r| r.loss_weight == 0.25));
    assert!(
        sample
            .rows
            .iter()
            .any(|r| r.phase == tesslot_core::Phase::B && r.loss_weight == 0.5)
    );
    assert!(
        sample
            .rows
            .iter()
            .any(|r| !r.durable && r.loss_weight > 0.0)
    );
    assert_eq!(sample.order_key(42), single.order_key(42));
    assert_eq!(
        sample.split(42, 8000, 1000).unwrap(),
        single.split(42, 8000, 1000).unwrap()
    );
    assert_ne!(sample.order_key(42), sample.order_key(43));
}

#[test]
fn missing_duplicate_reordered_and_noncontiguous_chunks_fail() {
    let original = records();
    for index in [0, 1, original.len() - 1] {
        let mut changed = original.clone();
        changed.remove(index);
        assert!(read(&encode(&changed)).is_err(), "missing chunk {index}");
    }
    let mut c = original.clone();
    c.insert(1, c[0].clone());
    assert!(read(&encode(&c)).is_err());
    let mut c = original.clone();
    c.swap(0, 1);
    assert!(read(&encode(&c)).is_err());
    let mut c = original.clone();
    c.extend(original);
    assert!(read(&encode(&c)).is_err());
}

#[test]
fn malformed_contracts_fail_with_errors() {
    for (field, value) in [
        ("format", serde_json::json!("kidra-token-chunk/2")),
        ("attention_rule", serde_json::json!("ordinary-causal")),
        ("cache_policy", serde_json::json!("detach")),
        ("tokenizer_sha256", serde_json::json!("0".repeat(64))),
        ("slot_policy", serde_json::json!("unknown")),
        ("chunk_count", serde_json::json!(999999)),
        ("sample_row_count", serde_json::json!(1)),
        ("trajectory_uuid", serde_json::json!("invalid")),
        ("requires_prefix_through_row", serde_json::json!(0)),
    ] {
        let mut c = records();
        c[0][field] = value;
        assert!(read(&encode(&c)).is_err(), "accepted bad {field}");
    }
    let mut c = records();
    c[0]["special_tokens"]["noop"] = serde_json::json!(2);
    assert!(read(&encode(&c)).is_err());
    let mut c = records();
    c[0]["window_slots"]["a"] = serde_json::json!(999);
    assert!(read(&encode(&c)).is_err());
    let mut c = records();
    c[0]["rows"][0]["token_id"] = serde_json::json!(99999);
    assert!(read(&encode(&c)).is_err());
    let mut c = records();
    c[0]["rows"][0]["position"] = serde_json::json!(9);
    assert!(read(&encode(&c)).is_err());
    let mut c = records();
    c[0]["labels"][0] = serde_json::json!(1);
    assert!(read(&encode(&c)).is_err());
    let mut c = records();
    c[0]["loss_weights"][0] = serde_json::json!(-1.0);
    assert!(read(&encode(&c)).is_err());
    let mut c = records();
    c[1]["chunk_count"] = serde_json::json!(24);
    assert!(read(&encode(&c)).is_err());
    assert!(read("{broken json}").is_err());
}

#[test]
fn malformed_lifecycle_and_noop_rejected_after_arrays_match() {
    let single = include_str!("../../../fixtures/canonical-single.jsonl");
    for field in ["durable", "phase", "window", "index", "removed_at"] {
        let mut c: serde_json::Value = serde_json::from_str(single).unwrap();
        let index = c["rows"]
            .as_array()
            .unwrap()
            .iter()
            .position(|r| r["token_id"] == 3)
            .unwrap();
        c["rows"][index][field] = match field {
            "durable" => serde_json::json!(true),
            "phase" => serde_json::json!("A"),
            "window" => serde_json::json!("nonexistent"),
            "index" => serde_json::json!(100000),
            _ => serde_json::json!({"tick":0,"input_rank":0}),
        };
        assert!(read(&c.to_string()).is_err(), "accepted bad {field}");
    }
}

#[test]
fn bounded_lines_and_tokenizer_contract() {
    let limits = Limits {
        line_bytes: 100,
        ..Limits::default()
    };
    let mut r = Reader::new(Cursor::new(DATA), identity(), limits).unwrap();
    assert!(r.next().unwrap().is_err());
    assert!(r.next().is_none());
    let mut id = identity();
    id.special_tokens.noop = id.special_tokens.eos;
    assert!(Reader::new(Cursor::new(DATA), id, Limits::default()).is_err());
    let mut id = identity();
    id.sha256 = "bad".into();
    assert!(Reader::new(Cursor::new(DATA), id, Limits::default()).is_err());
}

#[test]
fn fixture_manifest_integrity() {
    use sha2::{Digest, Sha256};
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    for (file, hash) in manifest["files"].as_object().unwrap() {
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(std::fs::read(root.join(file)).unwrap())
            ),
            hash.as_str().unwrap(),
            "{file}"
        );
    }
}

#[test]
fn long_trajectory_and_multiple_samples_stream_without_losing_ticks() {
    let data = format!("{DATA}{}", include_str!("../../../fixtures/long.jsonl"));
    let samples = read(&data).unwrap();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[1].ticks().count(), 2050);
    assert!(samples[1].rows.len() > 4000);
    assert_eq!(samples[1].metadata.sample_id, "long-idle");
    assert_eq!(samples[1].rows.last().unwrap().tick, 2049);
    assert_ne!(
        samples[0].metadata.trajectory_uuid,
        samples[1].metadata.trajectory_uuid
    );
}

#[test]
fn bounded_shuffle_is_seeded_and_keeps_complete_samples() {
    let sample = read(DATA).unwrap().remove(0);
    let run = |seed| {
        let source = (0..12).map(|i| {
            let mut s = sample.clone();
            s.metadata.sample_id = format!("sample-{i}");
            Ok(s)
        });
        tesslot_dataset::shuffle_samples(source, seed, 4)
            .unwrap()
            .map(|s| {
                let s = s.unwrap();
                assert_eq!(s.rows.len(), 429);
                assert_eq!(s.chunk_ends.len(), 25);
                s.metadata.sample_id
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(run(42), run(42));
    assert_ne!(run(42), run(43));
    let ids: std::collections::BTreeSet<_> = run(42).into_iter().collect();
    assert_eq!(ids.len(), 12);
    let source = std::iter::once(Err(tesslot_dataset::Error::Contract(
        "invalid input".into(),
    )));
    let mut shuffled = tesslot_dataset::shuffle_samples(source, 42, 4).unwrap();
    assert!(shuffled.next().unwrap().is_err());
    assert!(shuffled.next().is_none());
}

#[test]
fn materialized_sample_revalidation_rejects_changed_rows_and_chunk_splits() {
    let s = read(DATA).unwrap().remove(0);
    s.validate(&identity(), &Limits::default()).unwrap();
    let mut bad = s.clone();
    bad.rows.last_mut().unwrap().loss_weight = f64::INFINITY;
    assert!(bad.validate(&identity(), &Limits::default()).is_err());
    let mut bad = s.clone();
    bad.rows[0].token_id = 0;
    assert!(bad.validate(&identity(), &Limits::default()).is_err());
    let mut bad = s.clone();
    bad.chunk_ends[0] = 1;
    assert!(bad.validate(&identity(), &Limits::default()).is_err());
    let mut bad = s;
    bad.chunk_ends.pop();
    assert!(bad.validate(&identity(), &Limits::default()).is_err());
}

#[test]
fn uploaded_tokenizer_smoke_fixture_preserves_capacity_masks_and_cross_window_causality() {
    use sha2::{Digest, Sha256};
    let data = include_str!("../../../fixtures/pretrained-backward-24/trajectory.jsonl");
    let id: TokenizerIdentity = serde_json::from_str(include_str!(
        "../../../fixtures/pretrained-backward-24/tokenizer-identity.json"
    ))
    .unwrap();
    let manifest: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/pretrained-backward-24/manifest.json"
    ))
    .unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(data)),
        manifest["export_sha256"]
    );
    let samples: Vec<_> = Reader::new(Cursor::new(data), id, Limits::default())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(samples.len(), 1);
    let sample = &samples[0];
    assert_eq!(sample.rows.len(), 8);
    assert_eq!(sample.chunk_ends.len(), 3);
    assert_eq!(sample.metadata.max_window_slots, Some(24));
    assert_eq!(sample.metadata.windows.len(), 2);
    let targets = sample
        .rows
        .iter()
        .filter(|r| r.loss_weight > 0.0)
        .collect::<Vec<_>>();
    assert_eq!(targets.len(), 1);
    assert_eq!(targets[0].window, "answer");
    assert!(!visible_to(
        targets[0],
        targets[0],
        &sample.metadata.windows
    ));
    let context = sample
        .rows
        .iter()
        .find(|r| r.window == "context" && r.phase == tesslot_core::Phase::B)
        .unwrap();
    assert_eq!(context.loss_weight, 0.0);
    assert!(visible_to(targets[0], context, &sample.metadata.windows));
    assert_eq!(context.token_id, targets[0].token_id);
    assert!(sample.metadata.windows.iter().all(|w| w.close == Some(2)));
}
