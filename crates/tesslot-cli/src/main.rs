//! Headless entry point for the Tesslot CPU reference and local status server.

use clap::{Parser, Subcommand};
use std::{
    fs::File,
    io::{BufRead, BufReader, Cursor},
    path::PathBuf,
};
use tesslot_core::Config;
use tesslot_dataset::{Limits, Reader, TokenizerIdentity};
use tesslot_model::{Cache, ReferenceModel, WeightedLoss};

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
    Serve,
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
        Command::Config => println!("{}", serde_json::to_string_pretty(&config)?),
        Command::Serve => {
            tracing::info!(bind=%config.bind,"starting local status server; execution APIs unavailable");
            tesslot_api::serve(&config).await?;
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
    let model = ReferenceModel::new(config.model.clone())?;
    for sample in Reader::new(input, expected, Limits::default())? {
        let sample = sample?;
        if sample.rows.len() > 4096 {
            return Err("CPU dense oracle is limited to 4096 replay rows; dataset validation has larger independent limits".into());
        }
        let full = model.forward(&sample.rows, &sample.metadata.windows)?;
        let mut cache = Cache::default();
        let mut outputs = Vec::new();
        let mut ticks = 0;
        for group in sample.ticks() {
            ticks += 1;
            outputs.extend(model.tick(group, &sample.metadata.windows, &mut cache)?);
        }
        let delta = full
            .iter()
            .zip(&outputs)
            .flat_map(|(a, b)| a.logits.iter().zip(&b.logits).map(|(x, y)| (x - y).abs()))
            .fold(0.0_f64, f64::max);
        if full.len() != outputs.len() || delta > 1e-12 {
            return Err(
                format!("full/cached execution mismatch: max absolute error {delta}").into(),
            );
        }
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
                &serde_json::json!({"sample_id":sample.metadata.sample_id,"mode":"teacher_forced_cpu_reference",
            "ticks":ticks,"window_count":sample.metadata.windows.len(),"queries_per_window":windows,"live_durable_cache_tokens":cache.durable_tokens(),
            "reference_loss":loss.mean(),"effective_target_weight":loss.effective_weight,"max_full_cached_error":delta,"decision_preview":preview})
            )?
        );
    }
    Ok(())
}
