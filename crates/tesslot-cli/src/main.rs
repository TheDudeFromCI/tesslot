//! Headless entry point for the Tesslot CPU reference and local status server.

mod pretrained;
mod remix;

use clap::{Parser, Subcommand};
use std::{
    fs::File,
    io::{BufRead, BufReader, Cursor},
    path::PathBuf,
};
use tesslot_core::Config;
use tesslot_dataset::{Limits, Reader, TokenizerIdentity};
use tesslot_model::{ReferenceModel, WeightedLoss};
use tesslot_tensor::Model;

#[derive(Parser)]
#[command(version, about = "Tesslot: Kidra execution platform (CPU foundation)")]
struct Cli {
    #[arg(
        long,
        global = true,
        help = "Validated JSON configuration; defaults to loopback CPU"
    )]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Start the local HTTP server and embedded dashboard.
    Serve {
        #[arg(long)]
        tiny_experiment: bool,
    },
    /// Validate a complete export against a trusted tokenizer identity JSON.
    Validate {
        export: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
    },
    /// Replay a tiny trajectory using deterministic full and cached execution.
    Demo,
    /// Numerically replay a validated export (tiny vocabularies only).
    Replay {
        export: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
    },
    /// Inspect supported HF Qwen3 config without loading or downloading weights.
    InspectQwenConfig { model_config: PathBuf },
    /// Inspect local SafeTensors metadata without reading its weight payload.
    InspectWeights {
        weights: PathBuf,
        #[arg(long)]
        mapping: Option<PathBuf>,
    },
    /// Derive normalized native tokenizer identity and validate distinct controls.
    InspectTokenizer {
        tokenizer_file: PathBuf,
        #[arg(long, default_value = "<|bos|>")]
        bos: String,
        #[arg(long, default_value = "<|eos|>")]
        eos: String,
        #[arg(long, default_value = "<|noop|>")]
        noop: String,
    },
    /// Convert user-provided single-file Qwen tensors; never downloads models.
    ConvertQwen {
        #[arg(long)]
        model_config: PathBuf,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 4)]
        windows: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 134217728)]
        max_bytes: usize,
    },
    /// Opt-in synthetic CPU Qwen full-parameter training on native Remix exports.
    TrainRemixDemo(remix::Options),
    /// Opt-in full-parameter smoke trainer for a separately supplied complete Kidra conversion.
    TrainRemixLocal(remix::LocalOptions),
    /// Convert local HF shards, tokenizer and dedicated controls into a complete Kidra artifact.
    ConvertPretrained(pretrained::Convert),
    /// Execute a user-supplied base checkpoint with explicit parameter budgets and placement.
    ForwardQwen(pretrained::Forward),
    /// Print the default configuration for editing.
    Config,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    let config = match cli.config {
        Some(path) => Config::from_json(&std::fs::read(path)?)?,
        None => Config::default(),
    };
    config.validate()?;
    match cli.command {
        Command::ConvertPretrained(o) => pretrained::convert(o)?,
        Command::ForwardQwen(o) => pretrained::forward(o)?,
        Command::TrainRemixDemo(options) => remix::run(options)?,
        Command::TrainRemixLocal(options) => remix::run_local(options)?,
        Command::InspectQwenConfig { model_config: path } => {
            let c = tesslot_model::qwen::Config::from_json(&std::fs::read(path)?)
                .map_err(std::io::Error::other)?;
            println!(
                "{}",
                serde_json::json!({"model_type":c.model_type,"layers":c.num_hidden_layers,"hidden_size":c.hidden_size,"query_width":c.query_width().map_err(std::io::Error::other)?,"kv_width":c.kv_width().map_err(std::io::Error::other)?,"dtype":c.torch_dtype,"weight_loading":"local_single_file_or_sharded"})
            );
        }
        Command::InspectWeights { weights, mapping } => {
            let inventory = tesslot_artifacts::inspect(&mut File::open(weights)?)?;
            if let Some(path) = mapping {
                let mapping: tesslot_artifacts::Mapping =
                    serde_json::from_slice(&std::fs::read(path)?)?;
                let report = tesslot_artifacts::plan(&inventory, &mapping);
                println!("{}", serde_json::to_string_pretty(&report)?);
                if !report.is_valid() {
                    return Err("incompatible tensor mapping; inspect the report".into());
                }
            } else {
                println!("{}", serde_json::to_string_pretty(&inventory)?);
            }
        }
        Command::InspectTokenizer {
            tokenizer_file,
            bos,
            eos,
            noop,
        } => {
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(
                &mut std::io::Read::take(File::open(tokenizer_file)?, 64 * 1024 * 1024 + 1),
                &mut bytes,
            )?;
            let tokenizer =
                tesslot_artifacts::NativeTokenizer::from_bytes(&bytes, &bos, &eos, &noop)?;
            println!("{}", serde_json::to_string_pretty(tokenizer.identity())?);
        }
        Command::ConvertQwen {
            model_config,
            weights,
            output,
            windows,
            seed,
            max_bytes,
        } => {
            if output.exists() {
                return Err("conversion destination already exists; choose a new directory".into());
            }
            let c = tesslot_model::qwen::Config::from_json(&std::fs::read(model_config)?)
                .map_err(std::io::Error::other)?;
            let converted = tesslot_artifacts::qwen::convert(
                &mut File::open(weights)?,
                &c,
                windows,
                seed,
                max_bytes,
            )?;
            let parent = output
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| std::path::Path::new("."));
            let temporary = tempfile::Builder::new()
                .prefix(".tesslot-conversion-")
                .tempdir_in(parent)?;
            let metadata = std::collections::BTreeMap::from([
                ("format".into(), tesslot_artifacts::qwen::FORMAT.into()),
                (
                    "source_sha256".into(),
                    converted.manifest.source_sha256.clone(),
                ),
            ]);
            std::fs::write(
                temporary.path().join("model.safetensors"),
                tesslot_artifacts::encode_f32_with_metadata(&converted.tensors, metadata)?,
            )?;
            std::fs::write(
                temporary.path().join("config.json"),
                serde_json::to_vec_pretty(&c)?,
            )?;
            std::fs::write(
                temporary.path().join("conversion.json"),
                serde_json::to_vec_pretty(&converted.manifest)?,
            )?;
            // Reserve the destination exclusively, including against another converter.
            std::fs::create_dir(&output)?;
            if let Err(error) = std::fs::rename(temporary.path(), &output) {
                let _ = std::fs::remove_dir(&output);
                return Err(error.into());
            }
            println!(
                "{}",
                serde_json::json!({"output":output,"format":tesslot_artifacts::qwen::FORMAT,"source_sha256":converted.manifest.source_sha256,"reused_tensors":converted.manifest.report.compatible.len(),"initialized_tensors":converted.manifest.report.initialize.len(),"tokenizer_conversion_required":true})
            );
        }
        Command::Config => println!("{}", serde_json::to_string_pretty(&config)?),
        Command::Serve { tiny_experiment } => {
            tracing::info!(bind=%config.bind, tiny_experiment, "starting local server");
            if tiny_experiment {
                tesslot_api::serve_experimental(&config).await?;
            } else {
                tesslot_api::serve(&config).await?;
            }
        }
        Command::Validate { export, tokenizer } => {
            let expected: TokenizerIdentity = serde_json::from_slice(&std::fs::read(tokenizer)?)?;
            let mut samples = 0;
            let mut rows = 0;
            let mut chunks = 0;
            let mut weight = 0.0;
            for sample in Reader::new(
                BufReader::new(File::open(export)?),
                expected,
                Limits::default(),
            )? {
                let sample = sample?;
                samples += 1;
                rows += sample.rows.len();
                chunks += sample.chunk_ends.len();
                weight += sample.rows.iter().map(|r| r.loss_weight).sum::<f64>();
            }
            println!(
                "{}",
                serde_json::json!({"valid":true,"samples":samples,"rows":rows,"chunks":chunks,"effective_target_weight":weight})
            );
        }
        Command::Demo => {
            let expected =
                serde_json::from_str(include_str!("../../../fixtures/tokenizer-identity.json"))?;
            replay(
                Cursor::new(include_str!("../../../fixtures/canonical.jsonl")),
                expected,
                &config,
            )?;
        }
        Command::Replay { export, tokenizer } => {
            let expected = serde_json::from_slice(&std::fs::read(tokenizer)?)?;
            replay(BufReader::new(File::open(export)?), expected, &config)?;
        }
    }
    Ok(())
}

fn replay(
    input: impl BufRead,
    expected: TokenizerIdentity,
    config: &Config,
) -> Result<(), Box<dyn std::error::Error>> {
    if expected.vocab_size > config.model.vocab_size {
        return Err("tokenizer vocabulary exceeds configured reference vocabulary".into());
    }
    let values = ReferenceModel::new(config.model.clone())?.parameters();
    let model = Model::inference(config.model.clone(), &values).map_err(std::io::Error::other)?;
    drop(values);
    for sample in Reader::new(input, expected, Limits::default())? {
        let sample = sample?;
        if sample.rows.len() > 4096 {
            return Err("CPU dense oracle is limited to 4096 replay rows; dataset validation has larger independent limits".into());
        }
        let tensor_outputs = model
            .forward(&sample.rows, &sample.metadata.windows)
            .map_err(std::io::Error::other)?;
        let outputs: Vec<_> = tensor_outputs
            .into_iter()
            .map(|p| tesslot_model::Prediction {
                row_index: p.row_index,
                logits: p
                    .logits
                    .into_data_as::<f64>()
                    .try_to_vec::<f64>()
                    .expect("FP32 readback conversion"),
            })
            .collect();
        let ticks = sample.ticks().count();
        let mut loss = WeightedLoss::default();
        let mut windows = std::collections::BTreeMap::<String, usize>::new();
        for p in &outputs {
            let r = &sample.rows[p.row_index];
            loss.add(&p.logits, r.token_id as usize, r.loss_weight)?;
            *windows.entry(r.window.clone()).or_default() += 1;
        }
        let preview: Vec<_>=outputs.iter().filter(|p| sample.rows[p.row_index].phase==tesslot_core::Phase::D).take(8).map(|p| {
            let r=&sample.rows[p.row_index]; let predicted=p.logits.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).map(|(i,_)| i);
            serde_json::json!({"tick":r.tick,"window":r.window,"target":r.token_id,"predicted":predicted,"target_durable":r.durable})
        }).collect();
        println!(
            "{}",
            serde_json::to_string_pretty(
                &serde_json::json!({"sample_id":sample.metadata.sample_id,"mode":"teacher_forced_cpu_reference","dtype":"float32",
            "ticks":ticks,"window_count":sample.metadata.windows.len(),"queries_per_window":windows,
            "reference_loss":loss.mean(),"effective_target_weight":loss.effective_weight,"decision_preview":preview})
            )?
        );
    }
    Ok(())
}
