//! Explicit local checkpoint operations, separate from the bounded dashboard experiments.
use clap::Args;
use std::{collections::BTreeMap, path::PathBuf};
use tesslot_artifacts::{pretrained, qwen};
use tesslot_model::qwen::Config;
/// Convert a supplied HF checkpoint and tokenizer without changing existing IDs or rows.
#[derive(Args)]
pub struct Convert {
    /// Directory containing config.json, tokenizer.json and single-file or indexed weights.
    #[arg(long)]
    input: PathBuf,
    /// New directory; never overwrite existing artifacts.
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 4)]
    windows: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Aggregate FP32 parameter budget. Larger checkpoints require explicitly raising this.
    #[arg(long, default_value_t = 134217728)]
    max_bytes: usize,
}
pub fn convert(o: Convert) -> Result<(), Box<dyn std::error::Error>> {
    if !(1..=64).contains(&o.windows) {
        return Err("executable conversion window slots must be 1..64".into());
    }
    if o.output.exists() {
        return Err("conversion destination already exists".into());
    }
    let source = Config::from_json(&std::fs::read(o.input.join("config.json"))?)
        .map_err(std::io::Error::other)?;
    let (mut weights, provenance) = pretrained::load(&o.input, &source, o.max_bytes)?;
    let tokenizer_file = std::fs::File::open(o.input.join("tokenizer.json"))?;
    let mut tokenizer_bytes = Vec::new();
    std::io::Read::read_to_end(
        &mut std::io::Read::take(tokenizer_file, 64 * 1024 * 1024 + 1),
        &mut tokenizer_bytes,
    )?;
    let (tokenizer, conversion) =
        pretrained::extend_tokenizer(&tokenizer_bytes, source.vocab_size)?;
    let mut config = source.clone();
    pretrained::extend_embeddings(&mut weights, &mut config, &conversion, o.seed, o.max_bytes)?;
    let weights = qwen::augment(weights, &config, o.windows, o.seed, o.max_bytes)?;
    let initialized = weights
        .keys()
        .filter(|n| n.starts_with("kidra."))
        .cloned()
        .collect::<Vec<_>>();
    let bytes = tesslot_artifacts::encode_f32_with_metadata(
        &weights,
        BTreeMap::from([("format".into(), "pt".into())]),
    )?;
    use sha2::{Digest, Sha256};
    let manifest = serde_json::json!({"format":"kidra-pretrained/1","architecture":"kidra-qwen-remix/1","source_config":source,"source_files":provenance.files,"tokenizer_conversion":conversion,"seed":o.seed,"windows":o.windows,"initialized":initialized,"tied_output":config.tie_word_embeddings,"config_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&config)?)),"weights_sha256":format!("{:x}",Sha256::digest(&bytes))});
    let parent = o
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    let tmp = tempfile::tempdir_in(parent)?;
    for (name, data) in [
        ("model.safetensors", bytes),
        ("config.json", serde_json::to_vec_pretty(&config)?),
        ("tokenizer.json", tokenizer),
        ("conversion.json", serde_json::to_vec_pretty(&manifest)?),
    ] {
        let path = tmp.path().join(name);
        std::fs::write(&path, data)?;
        std::fs::File::open(path)?.sync_all()?;
    }
    std::fs::File::open(tmp.path())?.sync_all()?;
    // Reserve destination before rename so an existing empty directory is never replaced.
    std::fs::create_dir(&o.output)?;
    if let Err(e) = std::fs::rename(tmp.path(), &o.output) {
        let _ = std::fs::remove_dir(&o.output);
        return Err(e.into());
    }
    std::fs::File::open(parent)?.sync_all()?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}
/// Local base-Qwen forward smoke command. No tokenizer reinterpretation or HTTP exposure.
#[derive(Args)]
pub struct Forward {
    #[arg(long)]
    input: PathBuf,
    /// Comma-separated original token IDs.
    #[arg(long, value_delimiter = ',')]
    tokens: Vec<u32>,
    /// Explicit runtime::Plan JSON; default is FP32 CPU.
    #[arg(long)]
    runtime: Option<PathBuf>,
    #[arg(long, default_value_t = 134217728)]
    max_bytes: usize,
}
pub fn forward(o: Forward) -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_json(&std::fs::read(o.input.join("config.json"))?)
        .map_err(std::io::Error::other)?;
    let plan = match o.runtime {
        Some(p) => serde_json::from_slice(&std::fs::read(p)?)?,
        None => {
            tesslot_tensor::qwen::runtime::Plan::cpu(config.num_hidden_layers, o.max_bytes as u64)
        }
    };
    let (weights, _) = pretrained::load(&o.input, &config, o.max_bytes)?;
    let model = tesslot_tensor::qwen::Model::from_tensors(config, weights, None, false, plan)
        .map_err(std::io::Error::other)?;
    let logits = model.forward(&o.tokens).map_err(std::io::Error::other)?;
    let shape = logits.dims();
    let last = logits
        .slice([shape[0] - 1..shape[0], 0..shape[1]])
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    if last.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite output".into());
    }
    let next = last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .ok_or("empty output")?
        .0;
    println!(
        "{}",
        serde_json::json!({"next_token_id":next,"logits_shape":shape,"persistent_state_estimate":model.estimate()})
    );
    Ok(())
}

/// One attached trajectory backward; no AdamW allocation, step or checkpoint writes.
#[derive(Args)]
pub struct Backward {
    /// Complete convert-pretrained output directory.
    #[arg(long)]
    input: PathBuf,
    /// Exactly one small native trajectory compiled with the converted tokenizer.
    #[arg(long)]
    export: PathBuf,
    /// Explicit runtime plan. Default is FP32 CPU; CUDA requests never fall back.
    #[arg(long)]
    runtime: Option<PathBuf>,
    /// Construct only the exact causal prefix for one effective target; same full objective.
    #[arg(long)]
    target_prefix: bool,
    /// FP32 model load budget; does not cap activations/gradients.
    #[arg(long, default_value_t = 134217728)]
    max_bytes: usize,
}
pub fn backward(o: Backward) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{Cursor, Read};
    let mut tokenizer_bytes = Vec::new();
    std::fs::File::open(o.input.join("tokenizer.json"))?
        .take(64 * 1024 * 1024 + 1)
        .read_to_end(&mut tokenizer_bytes)?;
    let tokenizer = tesslot_artifacts::NativeTokenizer::from_bytes(
        &tokenizer_bytes,
        "<|kidra_bos|>",
        "<|kidra_eos|>",
        "<|kidra_noop|>",
    )?;
    let mut bytes = Vec::new();
    std::fs::File::open(o.export)?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 1024 * 1024 {
        return Err("backward smoke export exceeds 1 MiB".into());
    }
    let mut reader = tesslot_dataset::Reader::new(
        Cursor::new(&bytes),
        tokenizer.identity().clone(),
        tesslot_dataset::Limits {
            line_bytes: 1024 * 1024,
            sample_rows: 32,
            window_slots: 64,
        },
    )?;
    let sample = reader.next().ok_or("backward smoke export is empty")??;
    if reader.next().transpose()?.is_some() {
        return Err("backward smoke requires exactly one complete trajectory".into());
    }
    let weight = sample.rows.iter().map(|r| r.loss_weight).sum::<f64>();
    if weight <= 0.0 || !weight.is_finite() {
        return Err("backward probe requires positive finite target weight".into());
    }
    if o.target_prefix && sample.rows.iter().filter(|r| r.loss_weight > 0.0).count() != 1 {
        return Err("target-prefix probe requires exactly one effective target".into());
    }
    let config = Config::from_json(&std::fs::read(o.input.join("config.json"))?)
        .map_err(std::io::Error::other)?;
    let plan = match o.runtime {
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
        None => {
            tesslot_tensor::qwen::runtime::Plan::cpu(config.num_hidden_layers, o.max_bytes as u64)
        }
    };
    eprintln!("{}", serde_json::json!({"phase":"loading_model"}));
    let model = tesslot_tensor::qwen::Model::from_pretrained(&o.input, true, plan, o.max_bytes)
        .map_err(std::io::Error::other)?;
    eprintln!(
        "{}",
        serde_json::json!({"phase":"building_forward","target_prefix":o.target_prefix})
    );
    let loss = if o.target_prefix {
        model
            .remix_single_target_loss(&sample)
            .map_err(std::io::Error::other)?
    } else {
        let outputs = model
            .remix_forward(&sample, tesslot_tensor::qwen::remix::Execution::Chunked)
            .map_err(std::io::Error::other)?;
        model
            .remix_loss(&outputs, &sample)
            .map_err(std::io::Error::other)?
    };
    let value = loss
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|e| std::io::Error::other(e.to_string()))?[0];
    if !value.is_finite() {
        return Err("nonfinite trajectory loss".into());
    }
    eprintln!(
        "{}",
        serde_json::json!({"phase":"forward_complete","loss":value})
    );
    let summary = model
        .remix_backward_summary(&loss)
        .map_err(std::io::Error::other)?;
    if summary.gradient_norm == 0.0 {
        return Err("all parameter gradients are zero".into());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"format":"tesslot-backward/1","target_prefix":o.target_prefix,"rows":sample.rows.len(),"chunks":sample.chunk_ends.len(),"window_capacity":sample.metadata.max_window_slots,"active_windows":sample.metadata.windows.len(),"effective_weight":weight,"summary":summary,"optimizer_allocated":false,"updates":model.step,"persistent_state_estimate":{"weights":model.estimate().weights,"gradients":model.estimate().gradients,"optimizer":0,"update_temporaries":0},"sample_cursor":model.remix_state().map(|s|s.completed_samples)})
        )?
    );
    Ok(())
}
