//! Explicitly enabled, bounded CPU experiments. No model downloads or client filesystem paths.
use super::{AppState, error};
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::json;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tesslot_tensor::qwen::{Metrics, Model};
pub(super) struct Experiment {
    model: Mutex<Model>,
    busy: AtomicBool,
    stop: AtomicBool,
    metrics: Mutex<Vec<Metrics>>,
    failure: Mutex<Option<String>>,
    directory: std::path::PathBuf,
    latest: Mutex<Option<std::path::PathBuf>>,
    sequence: AtomicUsize,
}
// Release ownership even if the HTTP request disconnects while its worker runs.
struct Operation(Arc<Experiment>);
impl Drop for Operation {
    fn drop(&mut self) {
        self.0.busy.store(false, Ordering::Release);
    }
}
impl Experiment {
    pub fn new(directory: std::path::PathBuf) -> Result<Self, String> {
        std::fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
        let pointer = directory.join("latest.txt");
        let latest = if pointer.exists() {
            let name = std::fs::read_to_string(&pointer).map_err(|e| e.to_string())?;
            if name.len() > 128
                || !name
                    .strip_prefix("checkpoint-")
                    .is_some_and(|n| !n.is_empty() && n.bytes().all(|c| c.is_ascii_digit()))
            {
                return Err("invalid experiment checkpoint pointer".into());
            }
            Some(directory.join(name))
        } else {
            None
        };
        Ok(Self {
            model: Mutex::new(Model::demo(true)?),
            busy: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            metrics: Mutex::new(Vec::new()),
            failure: Mutex::new(None),
            directory,
            latest: Mutex::new(latest),
            sequence: AtomicUsize::new(0),
        })
    }
}
fn get(state: &AppState) -> Result<Arc<Experiment>, &'static str> {
    state
        .tiny
        .clone()
        .ok_or("Start serve --tiny-experiment to enable the bounded random-Qwen experiment.")
}
fn response(result: Result<serde_json::Value, String>) -> Response {
    match result {
        Ok(v) => Json(v).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, "experiment_failed", &e),
    }
}
pub(super) async fn status(State(state): State<Arc<AppState>>) -> Response {
    let Some(e) = &state.tiny else {
        return Json(json!({"enabled":false})).into_response();
    };
    Json(json!({"enabled":true,"busy":e.busy.load(Ordering::Acquire),"metrics":*e.metrics.lock().unwrap_or_else(|p|p.into_inner()),"error":*e.failure.lock().unwrap_or_else(|p|p.into_inner()),"checkpoint_available":e.latest.lock().unwrap_or_else(|p|p.into_inner()).is_some(),"model":"random-qwen-5184","dtype":"float32","tokenizer":"ascii/1","windows":1})).into_response()
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Generate {
    prompt: String,
    max_tokens: usize,
}
pub(super) async fn generate(
    State(s): State<Arc<AppState>>,
    r: Result<Json<Generate>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let r = match r {
        Ok(Json(r)) => r,
        Err(e) => {
            return error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_request",
                &e.to_string(),
            );
        }
    };
    let e = match get(&s) {
        Ok(e) => e,
        Err(message) => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "experiment_disabled",
                message,
            );
        }
    };
    if r.prompt.is_empty() || !r.prompt.is_ascii() || r.prompt.len() > 64 || r.max_tokens > 32 {
        return error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request",
            "Use 1..64 ASCII prompt bytes and at most 32 new tokens.",
        );
    }
    if e.busy
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return error(
            StatusCode::CONFLICT,
            "experiment_busy",
            "The tiny experiment is busy.",
        );
    }
    let operation = Operation(e.clone());
    let result = tokio::task::spawn_blocking(move || {
        let _operation = operation;
        let model = e.model.lock().map_err(|_| "experiment lock poisoned")?;
        let ids = r.prompt.bytes().map(u32::from).collect::<Vec<_>>();
        let tokens = model.generate(&ids, r.max_tokens)?;
        let text = tokens
            .iter()
            .map(|t| {
                if (32..127).contains(t) {
                    char::from_u32(*t).unwrap().to_string()
                } else {
                    format!("\\x{t:02x}")
                }
            })
            .collect::<String>();
        Ok(json!({"token_ids":tokens,"text":text,"step":model.step}))
    })
    .await;
    response(result.unwrap_or_else(|e| Err(e.to_string())))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Train {
    steps: usize,
}
pub(super) async fn train(
    State(s): State<Arc<AppState>>,
    r: Result<Json<Train>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let r = match r {
        Ok(Json(r)) => r,
        Err(e) => {
            return error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_request",
                &e.to_string(),
            );
        }
    };
    let e = match get(&s) {
        Ok(e) => e,
        Err(message) => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "experiment_disabled",
                message,
            );
        }
    };
    if !(1..=200).contains(&r.steps) {
        return error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request",
            "Training steps must be 1..200.",
        );
    }
    if e.busy
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return error(
            StatusCode::CONFLICT,
            "experiment_busy",
            "The tiny experiment is busy.",
        );
    }
    e.stop.store(false, Ordering::Release);
    *e.failure.lock().unwrap_or_else(|p| p.into_inner()) = None;
    let worker = e.clone();
    tokio::spawn(async move {
        let result = tokio::task::spawn_blocking(move || -> Result<(), String> {
            let mut model = worker
                .model
                .lock()
                .map_err(|_| "experiment lock poisoned")?;
            let text = b"a b c a b c a b c a b c ";
            let ids = text.iter().map(|v| u32::from(*v)).collect::<Vec<_>>();
            for _ in 0..r.steps {
                if worker.stop.load(Ordering::Acquire) {
                    break;
                }
                let metric = model.train_step(&ids[..ids.len() - 1], &ids[1..])?;
                let mut metrics = worker.metrics.lock().map_err(|_| "metrics lock poisoned")?;
                metrics.push(metric);
                if metrics.len() > 1000 {
                    metrics.remove(0);
                }
            }
            Ok(())
        })
        .await;
        if let Err(message) = result.unwrap_or_else(|e| Err(e.to_string())) {
            *e.failure.lock().unwrap_or_else(|p| p.into_inner()) = Some(message);
        }
        e.busy.store(false, Ordering::Release);
    });
    (StatusCode::ACCEPTED, Json(json!({"accepted":true}))).into_response()
}
pub(super) async fn stop(State(s): State<Arc<AppState>>) -> Response {
    let e = match get(&s) {
        Ok(e) => e,
        Err(message) => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "experiment_disabled",
                message,
            );
        }
    };
    e.stop.store(true, Ordering::Release);
    Json(json!({"stop_requested":true,"boundary":"after_completed_sgd_update"})).into_response()
}
async fn checkpoint(s: Arc<AppState>, load: bool) -> Response {
    let e = match get(&s) {
        Ok(e) => e,
        Err(message) => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "experiment_disabled",
                message,
            );
        }
    };
    if e.busy
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return error(
            StatusCode::CONFLICT,
            "experiment_busy",
            "The tiny experiment is busy.",
        );
    }
    let operation = Operation(e.clone());
    let result = tokio::task::spawn_blocking(move || -> Result<serde_json::Value, String> {
        let _operation = operation;
        let mut m = e.model.lock().map_err(|_| "experiment lock poisoned")?;
        if load {
            let path = e
                .latest
                .lock()
                .map_err(|_| "checkpoint lock poisoned")?
                .clone()
                .ok_or("No experiment checkpoint is available")?;
            *m = Model::restore(&path, true)?;
            e.metrics
                .lock()
                .map_err(|_| "metrics lock poisoned")?
                .clear();
        } else {
            let path = loop {
                let p = e.directory.join(format!(
                    "checkpoint-{}",
                    e.sequence.fetch_add(1, Ordering::Relaxed)
                ));
                if !p.exists() {
                    break p;
                }
            };
            m.save(&path)?;
            use std::io::Write;
            let mut pointer =
                tempfile::NamedTempFile::new_in(&e.directory).map_err(|e| e.to_string())?;
            pointer
                .write_all(path.file_name().unwrap().as_encoded_bytes())
                .map_err(|e| e.to_string())?;
            pointer.as_file().sync_all().map_err(|e| e.to_string())?;
            pointer
                .persist(e.directory.join("latest.txt"))
                .map_err(|e| e.to_string())?;
            std::fs::File::open(&e.directory)
                .and_then(|f| f.sync_all())
                .map_err(|e| e.to_string())?;
            *e.latest.lock().map_err(|_| "checkpoint lock poisoned")? = Some(path);
        }
        Ok(json!({"step":m.step,"restored":load}))
    })
    .await;
    response(result.unwrap_or_else(|e| Err(e.to_string())))
}
pub(super) async fn save(State(s): State<Arc<AppState>>) -> Response {
    checkpoint(s, false).await
}
pub(super) async fn restore(State(s): State<Arc<AppState>>) -> Response {
    checkpoint(s, true).await
}
