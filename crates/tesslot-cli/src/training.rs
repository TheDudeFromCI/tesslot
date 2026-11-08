//! Headless tiny CPU training, bounded by complete-trajectory reader/shuffle limits.
use std::{
    fs::File,
    io::{BufReader, Write},
    path::PathBuf,
    time::Instant,
};
use tesslot_dataset::{Limits, Reader, Sample, Split, TokenizerIdentity, shuffle_samples};
use tesslot_train::{Config, Metrics, Trainer, checkpoint};

#[derive(clap::Args)]
pub struct Checkpoints {
    /// Save a verified checkpoint after each complete epoch. Single writer per directory.
    #[arg(long)]
    checkpoint_dir: Option<PathBuf>,
    /// Resume the latest complete checkpoint; requires matching input and configs.
    #[arg(long, requires = "checkpoint_dir")]
    resume: bool,
    /// Stop safely after this many additional epochs, without writing final weights.
    #[arg(long, requires = "checkpoint_dir")]
    stop_after_epochs: Option<usize>,
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn run(
    export: PathBuf,
    tokenizer: PathBuf,
    controls: Option<PathBuf>,
    output: PathBuf,
    config: &tesslot_core::Config,
    checkpoints: Checkpoints,
) -> Result<()> {
    let training: Config = match controls {
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
        None => Config::default(),
    };
    training.validate().map_err(std::io::Error::other)?;
    if checkpoints.stop_after_epochs == Some(0) {
        return Err("stop-after-epochs must be positive".into());
    }
    if !checkpoints.resume
        && checkpoints
            .checkpoint_dir
            .as_ref()
            .is_some_and(|p| p.join("latest").exists())
    {
        return Err(
            "checkpoint root already has a latest checkpoint; use --resume or a new directory"
                .into(),
        );
    }
    if output.try_exists()? {
        return Err("output already exists; choose a new SafeTensors path".into());
    }
    let expected: TokenizerIdentity = serde_json::from_slice(&std::fs::read(tokenizer)?)?;
    if expected.vocab_size > config.model.vocab_size {
        return Err("tokenizer vocabulary exceeds configured reference vocabulary".into());
    }
    let reader = || -> Result<_> {
        Ok(Reader::new(
            BufReader::new(File::open(&export)?),
            expected.clone(),
            Limits {
                sample_rows: training.max_sample_rows,
                window_slots: config.model.max_windows,
                ..Limits::default()
            },
        )?)
    };
    // Validate the entire export (including held-out data) before the first update.
    let mut counts = [0usize; 3];
    for sample in reader()? {
        let sample = sample?;
        let index = match sample.split(
            training.seed,
            training.train_basis_points,
            training.validation_basis_points,
        )? {
            Split::Train => 0,
            Split::Development => 1,
            Split::Test => 2,
        };
        counts[index] += 1;
    }
    if counts[0] == 0 {
        return Err("deterministic split contains no training trajectories; adjust seed or train_basis_points".into());
    }
    let dataset_sha256 = checkpoint::fingerprint(&export)?;
    let start = Instant::now();
    let (mut trainer, start_epoch) = if checkpoints.resume {
        let (trainer, progress) = checkpoint::resume(
            checkpoints.checkpoint_dir.as_ref().unwrap(),
            &config.model,
            &training,
            &dataset_sha256,
            &expected,
        )?;
        let steps_per_epoch = counts[0].div_ceil(training.accumulation_samples);
        if trainer.steps()
            != (progress.completed_epochs as u64)
                .checked_mul(steps_per_epoch as u64)
                .ok_or("step count overflow")?
        {
            return Err("checkpoint step and epoch counts disagree with dataset".into());
        }
        (trainer, progress.completed_epochs)
    } else {
        (
            Trainer::new(config.model.clone(), training.clone()).map_err(std::io::Error::other)?,
            0,
        )
    };
    let evaluate = |trainer: &Trainer| -> Result<_> {
        let mut train = Metrics::default();
        let mut validation = Metrics::default();
        for sample in reader()? {
            let sample = sample?;
            let target = match sample.split(
                training.seed,
                training.train_basis_points,
                training.validation_basis_points,
            )? {
                Split::Train => &mut train,
                Split::Development => &mut validation,
                Split::Test => continue,
            };
            target
                .merge(&trainer.evaluate(&sample).map_err(std::io::Error::other)?)
                .map_err(std::io::Error::other)?;
        }
        Ok(
            serde_json::json!({"train":metric_json(&train),"validation":if validation.samples == 0 {serde_json::Value::Null} else {metric_json(&validation)}}),
        )
    };
    println!(
        "{}",
        serde_json::json!({"event":"evaluation","epoch":start_epoch,"step":trainer.steps(),"metrics":evaluate(&trainer)?})
    );
    let mut final_evaluation = evaluate(&trainer)?;
    let end_epoch = checkpoints.stop_after_epochs.map_or(training.epochs, |n| {
        start_epoch.saturating_add(n).min(training.epochs)
    });
    for epoch in start_epoch..end_epoch {
        let input = reader()?.filter(|item| match item {
            Err(_) => true,
            Ok(sample) => matches!(
                sample.split(
                    training.seed,
                    training.train_basis_points,
                    training.validation_basis_points
                ),
                Ok(Split::Train)
            ),
        });
        let shuffled = shuffle_samples(
            input,
            training.seed.wrapping_add(epoch as u64),
            training.shuffle_buffer,
        )?;
        let mut batch: Vec<Sample> = Vec::new();
        for sample in shuffled {
            batch.push(sample?);
            if batch.len() == training.accumulation_samples {
                update(&mut trainer, &batch, epoch + 1)?;
                batch.clear();
            }
        }
        if !batch.is_empty() {
            update(&mut trainer, &batch, epoch + 1)?;
        }
        if checkpoint::fingerprint(&export)? != dataset_sha256 {
            return Err("dataset changed during training".into());
        }
        final_evaluation = evaluate(&trainer)?;
        if let Some(root) = &checkpoints.checkpoint_dir {
            let path = checkpoint::save(
                root,
                &trainer,
                checkpoint::Progress {
                    completed_epochs: epoch + 1,
                    dataset_sha256: dataset_sha256.clone(),
                    tokenizer: expected.clone(),
                },
            )?;
            println!(
                "{}",
                serde_json::json!({"event":"checkpoint_saved","epoch":epoch + 1,"step":trainer.steps(),"path":path})
            );
        }
        println!(
            "{}",
            serde_json::json!({"event":"evaluation","epoch":epoch + 1,"step":trainer.steps(),"metrics":final_evaluation})
        );
    }
    if end_epoch < training.epochs {
        println!(
            "{}",
            serde_json::json!({"event":"stopped","completed_epochs":end_epoch,"step":trainer.steps(),"resumable":true})
        );
        return Ok(());
    }
    let tensors = trainer
        .parameters()
        .iter()
        .map(|(name, matrix)| {
            (
                name.clone(),
                (
                    vec![matrix.len(), matrix[0].len()],
                    matrix.iter().flatten().map(|v| *v as f32).collect(),
                ),
            )
        })
        .collect();
    let bytes = tesslot_artifacts::encode_f32(&tensors)?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    println!(
        "{}",
        serde_json::json!({"event":"completed","mode":"tiny_cpu_full_parameter_sft","dtype":"float32","steps":trainer.steps(),"elapsed_seconds":start.elapsed().as_secs_f64(),"split_samples":{"train":counts[0],"validation":counts[1],"test":counts[2]},"model_config":config.model,"training_config":training,"tokenizer":expected,"dataset_format":tesslot_dataset::FORMAT,"final_metrics":final_evaluation,"weights":output,"resumable":false})
    );
    Ok(())
}

fn metric_json(m: &Metrics) -> serde_json::Value {
    serde_json::json!({"samples":m.samples,"loss_numerator":m.loss_numerator,"effective_weight":m.effective_weight,"weighted_loss":m.loss(),"weighted_token_accuracy":m.accuracy()})
}
fn update(trainer: &mut Trainer, samples: &[Sample], epoch: usize) -> Result<()> {
    let references: Vec<_> = samples.iter().collect();
    let step = trainer
        .train_batch(&references)
        .map_err(std::io::Error::other)?;
    println!(
        "{}",
        serde_json::json!({"event":"optimizer_step","epoch":epoch,"step":step.step,"learning_rate":step.learning_rate,"gradient_norm":step.gradient_norm,"clip_scale":step.clip_scale,"metrics":metric_json(&step.metrics)})
    );
    Ok(())
}
