//! Opt-in CPU process-memory comparison. Run each mode in a fresh process.
use std::{io::Cursor, time::Instant};
use tesslot_dataset::{Limits, Reader, TokenizerIdentity};
use tesslot_tensor::qwen::Model;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::args()
        .nth(1)
        .ok_or("expected ordinary or recompute")?;
    let identity: TokenizerIdentity =
        serde_json::from_slice(include_bytes!("../../../fixtures/tokenizer-identity.json"))?;
    let sample = Reader::new(
        Cursor::new(include_bytes!("../../../fixtures/autograd.jsonl")),
        identity.clone(),
        Limits::default(),
    )?
    .next()
    .ok_or("missing sample")??;
    let mut model = Model::remix_demo(identity, 7, 42, true).map_err(std::io::Error::other)?;
    let start = Instant::now();
    let metrics = match mode.as_str() {
        "ordinary" => model.train_remix_step(&sample),
        "recompute" => model.train_remix_recomputed(&sample),
        _ => return Err("expected ordinary or recompute".into()),
    }
    .map_err(std::io::Error::other)?;
    let status = std::fs::read_to_string("/proc/self/status")?;
    let peak = status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .ok_or("Linux peak RSS unavailable")?;
    println!(
        "{}",
        serde_json::json!({"mode":mode,"seconds":start.elapsed().as_secs_f64(),"loss":metrics.loss,"process_peak_rss":peak,"measurement":"Linux process high-water resident set, includes runtime and model; not GPU allocator memory"})
    );
    Ok(())
}
