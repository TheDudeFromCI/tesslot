//! Same-filesystem checkpoint publication for the tiny CPU trainer.
use super::{Config, State, Trainer};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};
use tesslot_core::ModelConfig;
use tesslot_dataset::TokenizerIdentity;

const VERSION: &str = "tesslot-checkpoint/1";
const LIMIT: u64 = 64 * 1024 * 1024;
/// Errors include validation and filesystem failures without modifying the trainer.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Epoch-boundary provenance. No sample is partly accumulated at this boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Progress {
    /// Number of fully completed epochs; next epoch has this zero-based index.
    pub completed_epochs: usize,
    /// SHA-256 of the exact input export bytes, including storage chunking.
    pub dataset_sha256: String,
    /// Trusted tokenizer and native controls.
    pub tokenizer: TokenizerIdentity,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    version: String,
    architecture: String,
    dataset_format: String,
    model: ModelConfig,
    training: Config,
    step: u64,
    first: State,
    second: State,
    progress: Progress,
    // There is no stochastic forward/optimizer RNG in this CPU architecture.
    // Shuffle RNG is reconstructed from the base seed and next epoch index.
    rng_scheme: String,
    placement: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: String,
    files: BTreeMap<String, String>,
}

/// Fingerprint a local export without materializing it in memory.
pub fn fingerprint(path: &Path) -> Result<String> {
    let mut reader = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read(path: &Path) -> Result<Vec<u8>> {
    if !fs::symlink_metadata(path)?.file_type().is_file() {
        return Err("checkpoint payload must be a regular file".into());
    }
    let mut bytes = Vec::new();
    File::open(path)?.take(LIMIT + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > LIMIT {
        return Err("tiny checkpoint file exceeds 64 MiB limit".into());
    }
    Ok(bytes)
}
fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    if bytes.len() as u64 > LIMIT {
        return Err("tiny checkpoint file exceeds 64 MiB limit".into());
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
fn validate_progress(progress: &Progress, config: &Config, step: u64) -> Result<()> {
    progress.tokenizer.validate()?;
    if progress.completed_epochs == 0
        || progress.completed_epochs > config.epochs
        || step < progress.completed_epochs as u64
        || progress.dataset_sha256.len() != 64
        || !progress
            .dataset_sha256
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("invalid epoch progress or dataset identity".into());
    }
    Ok(())
}

/// Publish a verified immutable checkpoint directory, then atomically replace `latest`.
/// Use one writer per root. Prior checkpoints are retained; unpublished staging
/// directories are ignored. Publication targets Linux local filesystems.
pub fn save(root: &Path, trainer: &Trainer, progress: Progress) -> Result<PathBuf> {
    validate_progress(&progress, &trainer.config, trainer.step)?;
    fs::create_dir_all(root)?;
    if let Some(parent) = root.canonicalize()?.parent() {
        File::open(parent)?.sync_all()?;
    }
    let stage = tempfile::Builder::new()
        .prefix(".partial-")
        .tempdir_in(root)?;
    let tensors = trainer
        .values
        .iter()
        .map(|(name, m)| {
            (
                name.clone(),
                (
                    vec![m.len(), m[0].len()],
                    m.iter().flatten().map(|v| *v as f32).collect(),
                ),
            )
        })
        .collect();
    let weights = tesslot_artifacts::encode_f32(&tensors)?;
    let state = serde_json::to_vec(&Snapshot {
        version: VERSION.into(),
        architecture: tesslot_core::ARCHITECTURE_VERSION.into(),
        dataset_format: tesslot_dataset::FORMAT.into(),
        model: trainer.model.clone(),
        training: trainer.config.clone(),
        step: trainer.step,
        first: trainer.first.clone(),
        second: trainer.second.clone(),
        progress,
        rng_scheme: "splitmix64-buffer/1; seed+epoch; no stochastic model RNG".into(),
        placement: "cpu/flex/f32".into(),
    })?;
    write(&stage.path().join("model.safetensors"), &weights)?;
    write(&stage.path().join("state.json"), &state)?;
    let manifest = Manifest {
        version: VERSION.into(),
        files: BTreeMap::from([
            ("model.safetensors".into(), digest(&weights)),
            ("state.json".into(), digest(&state)),
        ]),
    };
    write(
        &stage.path().join("manifest.json"),
        &serde_json::to_vec(&manifest)?,
    )?;
    File::open(stage.path())?.sync_all()?;
    load(stage.path())?; // Verify the complete candidate before it becomes visible.
    let name = format!(
        "checkpoint-{}-{}",
        trainer.step,
        stage
            .path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .trim_start_matches(".partial-")
    );
    let destination = root.join(&name);
    fs::rename(stage.path(), &destination)?;
    File::open(root)?.sync_all()?;
    let mut pointer = tempfile::NamedTempFile::new_in(root)?;
    pointer.write_all(name.as_bytes())?;
    pointer.as_file().sync_all()?;
    pointer.persist(root.join("latest"))?;
    File::open(root)?.sync_all()?;
    Ok(destination)
}

/// Load only the atomically published pointer; never guess from staging directories.
pub fn load_latest(root: &Path) -> Result<(Trainer, Progress)> {
    let pointer = String::from_utf8(read(&root.join("latest"))?)?;
    let mut parts = Path::new(&pointer).components();
    if !pointer.starts_with("checkpoint-")
        || !matches!(parts.next(), Some(Component::Normal(_)))
        || parts.next().is_some()
    {
        return Err("invalid latest checkpoint pointer".into());
    }
    load(&root.join(pointer))
}
/// Validate version, hashes, names, shapes and finite values before restoring state.
pub fn load(directory: &Path) -> Result<(Trainer, Progress)> {
    if !fs::symlink_metadata(directory)?.file_type().is_dir() {
        return Err("checkpoint must be a directory".into());
    }
    let manifest: Manifest = serde_json::from_slice(&read(&directory.join("manifest.json"))?)?;
    if manifest.version != VERSION
        || manifest
            .files
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>()
            != ["model.safetensors", "state.json"]
    {
        return Err("unsupported checkpoint version or payload names".into());
    }
    let mut payloads = BTreeMap::new();
    for (name, expected) in &manifest.files {
        let bytes = read(&directory.join(name))?;
        if digest(&bytes) != *expected {
            return Err(format!("checkpoint checksum mismatch: {name}").into());
        }
        payloads.insert(name.clone(), bytes);
    }
    let state: Snapshot = serde_json::from_slice(&payloads["state.json"])?;
    if state.version != VERSION
        || state.architecture != tesslot_core::ARCHITECTURE_VERSION
        || state.dataset_format != tesslot_dataset::FORMAT
        || state.placement != "cpu/flex/f32"
        || state.rng_scheme != "splitmix64-buffer/1; seed+epoch; no stochastic model RNG"
    {
        return Err("unsupported checkpoint architecture, placement or RNG scheme".into());
    }
    state.training.validate().map_err(std::io::Error::other)?;
    state.model.validate()?;
    validate_progress(&state.progress, &state.training, state.step)?;
    if state.progress.tokenizer.vocab_size > state.model.vocab_size {
        return Err("checkpoint tokenizer exceeds vocabulary".into());
    }
    let mut cursor = std::io::Cursor::new(&payloads["model.safetensors"]);
    let inventory = tesslot_artifacts::inspect(&mut cursor)?;
    let mut values = BTreeMap::new();
    for (name, tensor) in inventory.tensors {
        if tensor.dtype != "F32" || tensor.shape.len() != 2 || tensor.shape.contains(&0) {
            return Err("checkpoint requires nonempty FP32 matrices".into());
        }
        let (shape, data) = tesslot_artifacts::read_f32(&mut cursor, &name, LIMIT as usize)?;
        values.insert(
            name,
            data.chunks(shape[1])
                .map(|row| row.iter().map(|v| f64::from(*v)).collect())
                .collect(),
        );
    }
    let mut trainer = Trainer::from_parameters(state.model, state.training, values)
        .map_err(std::io::Error::other)?;
    for (actual, expected, nonnegative) in [
        (&state.first, &trainer.first, false),
        (&state.second, &trainer.second, true),
    ] {
        if actual.keys().ne(expected.keys())
            || expected.iter().any(|(n, m)| {
                actual[n].len() != m.len()
                    || actual[n].iter().zip(m).any(|(a, b)| {
                        a.len() != b.len()
                            || a.iter()
                                .any(|v| !v.is_finite() || (nonnegative && *v < 0.0))
                    })
            })
        {
            return Err("incompatible optimizer moment names, shapes or values".into());
        }
    }
    trainer.first = state.first;
    trainer.second = state.second;
    trainer.step = state.step;
    Ok((trainer, state.progress))
}

/// Resume only an identical CPU run; device migration is not supported yet.
pub fn resume(
    root: &Path,
    model: &ModelConfig,
    training: &Config,
    dataset_sha256: &str,
    tokenizer: &TokenizerIdentity,
) -> Result<(Trainer, Progress)> {
    let (trainer, progress) = load_latest(root)?;
    if &trainer.model != model
        || serde_json::to_value(&trainer.config)? != serde_json::to_value(training)?
        || progress.dataset_sha256 != dataset_sha256
        || serde_json::to_value(&progress.tokenizer)? != serde_json::to_value(tokenizer)?
    {
        return Err(
            "checkpoint model, training config, dataset or tokenizer identity mismatch".into(),
        );
    }
    Ok((trainer, progress))
}
