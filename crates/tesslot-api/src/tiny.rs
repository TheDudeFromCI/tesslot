//! Explicitly enabled, bounded CPU experiments. No model downloads or client filesystem paths.
use super::{AppState, error};
use axum::{
    Extension, Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get as route_get, post},
};
use serde::Deserialize;
use serde_json::json;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tesslot_tensor::qwen::{
    Metrics, Model,
    corpus::{Dataset, Split},
    echo::pattern,
};
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Base,
    Echo,
    Corpus,
}
pub(super) fn routes(kind: Kind) -> Router<Arc<AppState>> {
    let name = match kind {
        Kind::Base => "tiny-qwen",
        Kind::Echo => "tiny-qwen-echo",
        Kind::Corpus => "tiny-qwen-echo-corpus",
    };
    let router = Router::new()
        .route(
            &format!("/inference/experiments/{name}/status"),
            route_get(status),
        )
        .route(
            &format!("/inference/experiments/{name}/generate"),
            if kind != Kind::Base {
                post(generate_echo)
            } else {
                post(generate)
            },
        )
        .route(&format!("/training/experiments/{name}/jobs"), post(train))
        .route(&format!("/training/experiments/{name}/stop"), post(stop))
        .route(
            &format!("/training/experiments/{name}/checkpoint"),
            post(save),
        )
        .route(
            &format!("/training/experiments/{name}/restore"),
            post(restore),
        );
    let router = if kind == Kind::Corpus {
        router.route(
            &format!("/training/experiments/{name}/evaluate"),
            post(evaluate_corpus),
        )
    } else {
        router
    };
    router.layer(Extension(kind))
}
pub(super) struct Experiment {
    model: Mutex<Model>,
    busy: AtomicBool,
    stop: AtomicBool,
    metrics: Mutex<Vec<Metrics>>,
    failure: Mutex<Option<String>>,
    directory: std::path::PathBuf,
    latest: Mutex<Option<std::path::PathBuf>>,
    sequence: AtomicUsize,
    echo: bool,
    corpus: bool,
    progress: Mutex<Option<tesslot_tensor::qwen::corpus::TrainingState>>,
    echo_result: Mutex<Option<serde_json::Value>>,
}
// Release ownership even if the HTTP request disconnects while its worker runs.
struct Operation(Arc<Experiment>);
impl Drop for Operation {
    fn drop(&mut self) {
        self.0.busy.store(false, Ordering::Release);
    }
}
impl Experiment {
    pub fn new(directory: std::path::PathBuf, kind: Kind) -> Result<Self, String> {
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
        let echo = kind != Kind::Base;
        let corpus = kind == Kind::Corpus;
        let mut model = if echo {
            Model::echo_demo(true)?
        } else {
            Model::demo(true)?
        };
        if corpus {
            model.start_echo_corpus(Dataset::default())?;
            model.enable_adamw(Default::default(), 0.003)?;
        }
        let progress = Mutex::new(model.echo_corpus().cloned());
        Ok(Self {
            model: Mutex::new(model),
            corpus,
            progress,
            echo,
            echo_result: Mutex::new(None),
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
fn get(state: &AppState, kind: Kind) -> Result<Arc<Experiment>, &'static str> {
    (match kind {
        Kind::Base => &state.tiny,
        Kind::Echo => &state.tiny_echo,
        Kind::Corpus => &state.tiny_corpus,
    })
    .clone()
    .ok_or("Start serve --tiny-experiment to enable the bounded random-Qwen experiment.")
}
fn response(result: Result<serde_json::Value, String>) -> Response {
    match result {
        Ok(v) => Json(v).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, "experiment_failed", &e),
    }
}
pub(super) async fn status(
    State(state): State<Arc<AppState>>,
    Extension(kind): Extension<Kind>,
) -> Response {
    let Ok(e) = get(&state, kind) else {
        return Json(json!({"enabled":false})).into_response();
    };
    Json(json!({
        "enabled":true,
        "busy":e.busy.load(Ordering::Acquire),
        "metrics":*e.metrics.lock().unwrap_or_else(|p|p.into_inner()),
        "error":*e.failure.lock().unwrap_or_else(|p|p.into_inner()),
        "checkpoint_available":e.latest.lock().unwrap_or_else(|p|p.into_inner()).is_some(),
        "model":if e.corpus {"random-qwen-kidra-echo-corpus"} else if e.echo {"random-qwen-kidra-echo"} else {"random-qwen-5184"},
        "dtype":"float32",
        "tokenizer":"ascii/1",
        "windows":if e.echo {2} else {1},
        "corpus":*e.progress.lock().unwrap_or_else(|p|p.into_inner()),
        "optimizer":if e.corpus {"adamw-f32/1"} else {"sgd-no-momentum/1"},
        "echo_result":*e.echo_result.lock().unwrap_or_else(|p|p.into_inner())
    })).into_response()
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Generate {
    prompt: String,
    max_tokens: usize,
}
pub(super) async fn generate(
    State(s): State<Arc<AppState>>,
    Extension(kind): Extension<Kind>,
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
    let e = match get(&s, kind) {
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
    seed: Option<u64>,
}
pub(super) async fn train(
    State(s): State<Arc<AppState>>,
    Extension(kind): Extension<Kind>,
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
    let e = match get(&s, kind) {
        Ok(e) => e,
        Err(message) => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "experiment_disabled",
                message,
            );
        }
    };
    if ((!e.echo || e.corpus) && r.seed.is_some())
        || !(1..=if e.corpus {
            4000
        } else if e.echo {
            1000
        } else {
            200
        })
            .contains(&r.steps)
    {
        return error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request",
            "Use 1..200 steps for ABC, 1..1000 for a single echo pattern, or 1..4000 for the corpus. Only single-pattern echo accepts a seed.",
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
                let metric = if worker.corpus {
                    model.train_echo_corpus_step()?
                } else if worker.echo {
                    model.train_echo_step(&pattern(r.seed.unwrap_or(42)))?
                } else {
                    model.train_step(&ids[..ids.len() - 1], &ids[1..])?
                };
                *worker
                    .progress
                    .lock()
                    .map_err(|_| "progress lock poisoned")? = model.echo_corpus().cloned();
                let mut metrics = worker.metrics.lock().map_err(|_| "metrics lock poisoned")?;
                metrics.push(metric);
                if metrics.len() > 1000 {
                    metrics.remove(0);
                }
            }
            if worker.echo {
                *worker
                    .echo_result
                    .lock()
                    .map_err(|_| "echo result lock poisoned")? = Some(if worker.corpus {
                    serde_json::to_value(
                        model.evaluate_echo_corpus(
                            Split::Development,
                            model
                                .echo_corpus()
                                .ok_or("missing corpus state")?
                                .dataset
                                .development_size
                                .min(8),
                        )?,
                    )
                    .map_err(|e| e.to_string())?
                } else {
                    echo_report(&model, r.seed.unwrap_or(42))?
                });
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
pub(super) async fn stop(
    State(s): State<Arc<AppState>>,
    Extension(kind): Extension<Kind>,
) -> Response {
    let e = match get(&s, kind) {
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
async fn checkpoint(s: Arc<AppState>, load: bool, kind: Kind) -> Response {
    let e = match get(&s, kind) {
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
            let restored = Model::restore(&path, true)?;
            if restored.is_echo() != e.echo
                || restored.echo_corpus().is_some() != e.corpus
                || (e.corpus && restored.optimizer_id() != "adamw-f32/1")
            {
                return Err("checkpoint experiment architecture mismatch".into());
            }
            *m = restored;
            *e.progress.lock().map_err(|_| "progress lock poisoned")? = m.echo_corpus().cloned();
            *e.echo_result
                .lock()
                .map_err(|_| "echo result lock poisoned")? = None;
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
pub(super) async fn save(
    State(s): State<Arc<AppState>>,
    Extension(kind): Extension<Kind>,
) -> Response {
    checkpoint(s, false, kind).await
}
pub(super) async fn restore(
    State(s): State<Arc<AppState>>,
    Extension(kind): Extension<Kind>,
) -> Response {
    checkpoint(s, true, kind).await
}

fn echo_report(model: &Model, seed: u64) -> Result<serde_json::Value, String> {
    let source = pattern(seed);
    Ok(
        json!({"seed":seed,"step":model.step,"replay":model.replay_echo(&source,true)?,"loss":model.evaluate_echo(&source,true)?,"source_masked_loss":model.evaluate_echo(&source,false)?}),
    )
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GenerateEcho {
    seed: u64,
}
async fn generate_echo(
    State(s): State<Arc<AppState>>,
    Extension(kind): Extension<Kind>,
    r: Result<Json<GenerateEcho>, axum::extract::rejection::JsonRejection>,
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
    let e = match get(&s, kind) {
        Ok(e) => e,
        Err(m) => return error(StatusCode::SERVICE_UNAVAILABLE, "experiment_disabled", m),
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
    let result = tokio::task::spawn_blocking(move || {
        let _operation = operation;
        let model = e.model.lock().map_err(|_| "experiment lock poisoned")?;
        let report = echo_report(&model, r.seed)?;
        *e.echo_result
            .lock()
            .map_err(|_| "echo result lock poisoned")? = Some(report.clone());
        Ok(report)
    })
    .await;
    response(result.unwrap_or_else(|e| Err(e.to_string())))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluateCorpus {
    split: Split,
    samples: usize,
}
async fn evaluate_corpus(
    State(s): State<Arc<AppState>>,
    Extension(kind): Extension<Kind>,
    r: Result<Json<EvaluateCorpus>, axum::extract::rejection::JsonRejection>,
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
    let e = match get(&s, kind) {
        Ok(e) => e,
        Err(m) => return error(StatusCode::SERVICE_UNAVAILABLE, "experiment_disabled", m),
    };
    if r.split == Split::Train || !(1..=16).contains(&r.samples) {
        return error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request",
            "Use a development/test split and 1..16 evaluation patterns.",
        );
    }
    if e.busy
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return error(
            StatusCode::CONFLICT,
            "experiment_busy",
            "The corpus experiment is busy.",
        );
    }
    let operation = Operation(e.clone());
    let result = tokio::task::spawn_blocking(move || {
        let _operation = operation;
        let model = e.model.lock().map_err(|_| "experiment lock poisoned")?;
        let report = serde_json::to_value(model.evaluate_echo_corpus(r.split, r.samples)?)
            .map_err(|e| e.to_string())?;
        *e.echo_result
            .lock()
            .map_err(|_| "echo result lock poisoned")? = Some(report.clone());
        Ok(report)
    })
    .await;
    response(result.unwrap_or_else(|e| Err(e.to_string())))
}
