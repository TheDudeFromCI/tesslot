//! Explicit CPU reference trainer. Reads one complete trajectory at a time.
use clap::Args;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};
use tesslot_artifacts::NativeTokenizer;
use tesslot_dataset::{Limits, Reader, TokenizerIdentity};
use tesslot_tensor::qwen::{Model, remix::MAX_ROWS};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const FILE_LIMIT: u64 = 64 * 1024 * 1024;
#[derive(Args)]
pub(crate) struct Options {
    /// Native kidra-token-chunk/1 JSONL, never flattened into chat examples.
    export: PathBuf,
    /// Real tokenizer.json; normalized fingerprint must match every exported sample.
    #[arg(long)]
    tokenizer_file: PathBuf,
    /// New atomic checkpoint directory; existing directories are never overwritten.
    #[arg(long)]
    output: PathBuf,
    /// Additional whole-trajectory optimizer updates (CPU-only experiment).
    #[arg(long, default_value_t = 20)]
    steps: usize,
    /// Slot capacity for fresh synthetic initialization (default 7).
    #[arg(long, conflicts_with = "resume")]
    windows: Option<usize>,
    /// Seed for fresh synthetic initialization (default 42).
    #[arg(long, conflicts_with = "resume")]
    seed: Option<u64>,
    /// Restore weights, AdamW and dataset cursor instead of starting a new model.
    #[arg(long)]
    resume: Option<PathBuf>,
    /// Exact target-prefix replay; extra compute, no chunk-gradient truncation.
    #[arg(long)]
    recompute: bool,
}
#[derive(Args)]
pub(crate) struct LocalOptions {
    #[command(flatten)]
    training: Options,
    /// Complete convert-pretrained output; required for a fresh local-model experiment.
    #[arg(long,required_unless_present="resume",conflicts_with_all=["resume","windows","seed"])]
    pretrained: Option<PathBuf>,
    /// Explicit device placement/compute precision JSON. Defaults to CPU FP32.
    #[arg(long)]
    runtime: Option<PathBuf>,
    /// Explicit checkpoint and FP32 import budget; larger models require raising this.
    #[arg(long, default_value_t = 134217728)]
    max_bytes: usize,
}
pub(crate) fn run_local(options: LocalOptions) -> Result<()> {
    run_with(
        options.training,
        Some((options.pretrained, options.runtime, options.max_bytes)),
    )
}
fn fingerprint(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 8192];
    let mut bytes = 0u64;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        bytes += n as u64;
        if bytes > FILE_LIMIT {
            return Err("CPU reference export exceeds 64 MiB limit".into());
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn reader(path: &Path, identity: TokenizerIdentity) -> Result<Reader<BufReader<File>>> {
    Ok(Reader::new(
        BufReader::new(File::open(path)?),
        identity,
        Limits {
            line_bytes: 8 * 1024 * 1024,
            sample_rows: MAX_ROWS,
            window_slots: 64,
        },
    )?)
}
pub(crate) fn run(options: Options) -> Result<()> {
    run_with(options, None)
}
type LocalInput = (Option<PathBuf>, Option<PathBuf>, usize);
fn run_with(options: Options, local: Option<LocalInput>) -> Result<()> {
    if !(1..=4000).contains(&options.steps) {
        return Err("use 1..4000 complete-trajectory updates".into());
    }
    if options.output.exists() {
        return Err("checkpoint destination exists; choose a new directory".into());
    }
    let parent = options
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if !parent.is_dir() {
        return Err("checkpoint parent directory does not exist; create it before training".into());
    }
    let mut bytes = Vec::new();
    File::open(&options.tokenizer_file)?
        .take(FILE_LIMIT + 1)
        .read_to_end(&mut bytes)?;
    let controls = if local.is_some() {
        ["<|kidra_bos|>", "<|kidra_eos|>", "<|kidra_noop|>"]
    } else {
        ["<|bos|>", "<|eos|>", "<|noop|>"]
    };
    let tokenizer = NativeTokenizer::from_bytes(&bytes, controls[0], controls[1], controls[2])?;
    let identity = tokenizer.identity().clone();
    let mut model = if let Some(path) = &options.resume {
        let model = Model::restore_with_budget(
            path,
            true,
            local
                .as_ref()
                .map(|(_, _, b)| *b)
                .unwrap_or(2 * 1024 * 1024),
        )?;
        let state = model
            .remix_state()
            .ok_or("resume requires a Qwen/Remix checkpoint")?;
        if state.tokenizer.sha256 != identity.sha256
            || state.tokenizer.vocab_size != identity.vocab_size
            || state.tokenizer.special_tokens != identity.special_tokens
            || model.optimizer_id() != "adamw-f32/1"
        {
            return Err("checkpoint tokenizer or optimizer does not match this trainer".into());
        }
        model
    } else if let Some((Some(path), runtime, budget)) = &local {
        let config =
            tesslot_model::qwen::Config::from_json(&std::fs::read(path.join("config.json"))?)
                .map_err(std::io::Error::other)?;
        let plan = match runtime {
            Some(p) => serde_json::from_slice(&std::fs::read(p)?)?,
            None => {
                tesslot_tensor::qwen::runtime::Plan::cpu(config.num_hidden_layers, *budget as u64)
            }
        };
        let mut model = Model::from_pretrained(path, true, plan, *budget)?;
        model.enable_adamw(Default::default(), 0.003)?;
        model
    } else {
        let mut model = Model::remix_demo(
            identity.clone(),
            options.windows.unwrap_or(7),
            options.seed.unwrap_or(42),
            true,
        )?;
        model.enable_adamw(Default::default(), 0.003)?;
        model
    };
    let state = model.remix_state().ok_or("missing Remix state")?;
    if state.tokenizer.sha256 != identity.sha256
        || state.tokenizer.vocab_size != identity.vocab_size
        || state.tokenizer.special_tokens != identity.special_tokens
    {
        return Err("supplied tokenizer differs from converted model".into());
    }
    if options.resume.is_some()
        && let Some((_, Some(path), _)) = &local
    {
        model.configure_runtime(serde_json::from_slice(&std::fs::read(path)?)?)?;
    }
    if local.is_some() {
        println!(
            "{}",
            serde_json::json!({"event":"persistent_state_estimate","bytes":model.estimate(),"excludes":"activations, attention temporaries, allocator overhead and checkpoint serialization"})
        );
    }
    let hash = fingerprint(&options.export)?;
    model.bind_remix_dataset(hash.clone())?;
    // Validate the entire file, including suffixes not reached in this invocation,
    // before allowing even the first optimizer update. Memory stays sample-bounded.
    let mut count = 0u64;
    for sample in reader(&options.export, identity.clone())? {
        let sample = sample?;
        model.validate_remix_sample(&sample)?;
        if !sample.rows.iter().any(|r| r.loss_weight > 0.0) {
            return Err("sample has no effective training targets".into());
        }
        count += 1;
    }
    if count == 0 {
        return Err("export contains no complete samples".into());
    }
    if fingerprint(&options.export)? != hash {
        return Err("export changed during preflight".into());
    }
    let mut skip = model
        .remix_state()
        .ok_or("missing Remix state")?
        .completed_samples
        % count;
    let mut remaining = options.steps;
    while remaining > 0 {
        let mut progressed = false;
        for (index, sample) in reader(&options.export, identity.clone())?.enumerate() {
            let sample = sample?;
            if (index as u64) < skip {
                continue;
            }
            let metric = if options.recompute {
                model.train_remix_recomputed(&sample)?
            } else {
                model.train_remix_step(&sample)?
            };
            progressed = true;
            println!(
                "{}",
                serde_json::json!({"event":"optimizer_updated","sample_id":sample.metadata.sample_id,"trajectory_uuid":sample.metadata.trajectory_uuid,"chunks":sample.chunk_ends.len(),"effective_target_weight":sample.rows.iter().map(|r|r.loss_weight).sum::<f64>(),"metrics":metric})
            );
            remaining -= 1;
            if remaining == 0 {
                break;
            }
        }
        if !progressed {
            return Err("export changed or resumed sample disappeared during training".into());
        }
        skip = 0;
    }
    if fingerprint(&options.export)? != hash {
        return Err("export changed during training; checkpoint was not published".into());
    }
    model.save(&options.output)?;
    println!(
        "{}",
        serde_json::json!({"event":"checkpoint_saved","path":options.output.to_string_lossy(),"step":model.step,"dataset_samples":count,"mode":"full_parameter_full_trajectory","state":model.remix_state(),"optimizer":model.optimizer_id(),"recomputation":options.recompute,"runtime":if local.is_some(){"explicit_plan_or_cpu_default"}else{"cpu_fp32"}})
    );
    Ok(())
}
