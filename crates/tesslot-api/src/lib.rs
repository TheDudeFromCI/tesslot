//! Local API foundation with opt-in CPU experiments. Unavailable routes fail explicitly.

mod tiny;

use axum::{
    Json, Router,
    extract::{Request, State},
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc, time::Instant};
use tesslot_core::{API_VERSION, ARCHITECTURE_VERSION, Config};

/// Machine-readable failure envelope shared by all native HTTP routes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    /// Stable programmatic identifier.
    pub code: String,
    /// Human-readable actionable message.
    pub message: String,
    /// Structured context; never includes secrets or physical device placement.
    pub details: BTreeMap<String, String>,
}

/// Live system status. Unsupported measurements are never synthesized.
#[derive(Debug, Serialize, Deserialize)]
pub struct SystemStatus {
    /// Tesslot application version.
    pub application_version: String,
    /// Independent native protocol version.
    pub api_version: String,
    /// Reference architecture identifier.
    pub architecture_version: String,
    /// Monotonic elapsed server time.
    pub uptime_ms: u64,
    /// Current operational state.
    pub status: String,
    /// Actual availability of each subsystem.
    pub capabilities: BTreeMap<String, String>,
}

/// Proposed inference session request. Endpoint returns 501 in milestone one.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateInferenceSession {
    /// Registered model, not a remote arbitrary download URL.
    pub model_id: String,
    /// Explicit concurrency budget.
    pub max_windows: usize,
    /// Reproducible sampling seed.
    pub seed: u64,
}

/// Proposed training job request. Registration is intentionally unavailable.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateTrainingJob {
    /// Registered dataset identity.
    pub dataset_id: String,
    /// Registered architecture-converted model.
    pub model_id: String,
    /// Full trajectory is one optimization unit; never detached chunks.
    pub mode: TrainingMode,
}
/// Deliberately excludes implicit truncated backpropagation.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    /// All eligible parameters, attached full-trajectory gradients.
    FullParameterSft,
}

/// Proposed persistent task request; does not grant tools or parent privileges.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateAgentTask {
    /// Persistent project identity.
    pub project_id: String,
    /// Specialist instructions.
    pub instructions: String,
    /// Explicit authorized tool scopes; an empty list grants none.
    pub permissions: Vec<String>,
}

/// Versioned logical event envelope. Streaming transport is deferred.
#[derive(Debug, Serialize, Deserialize)]
pub struct Event {
    /// `window-events/1`, independent from the application version.
    pub schema_version: String,
    /// Persistent logical session, without GPU details.
    pub session_id: String,
    /// Strictly increasing session-local ordering and reconnect cursor.
    pub sequence: u64,
    /// Tick when semantically applicable; tool wall-time events may have no tick.
    pub tick: Option<u64>,
    /// Logical window identity if relevant.
    pub window_id: Option<String>,
    /// Tool, lifecycle or subagent operation correlation.
    pub operation_id: Option<String>,
    /// Typed category and payload.
    #[serde(flatten)]
    pub payload: EventPayload,
}

/// Initial event vocabulary; new protocol categories require version review.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum EventPayload {
    /// Window identity is in the envelope.
    WindowOpened {
        /// Embedding slot, unrelated to devices.
        slot: usize,
    },
    /// Closure after phase D.
    WindowClosed {
        /// EOS, cancellation or operator action.
        reason: String,
    },
    /// Includes non-durable NOOP decisions.
    TokenGenerated {
        /// Native tokenizer ID.
        token_id: u32,
        /// Whether committed to memory.
        durable: bool,
    },
    /// Atomic parallel decision group committed.
    TickCompleted,
    /// Observation is inserted in the named tick's B phase.
    ExternalObservationInserted {
        /// Inserted durable IDs.
        token_ids: Vec<u32>,
    },
    /// Terminal event; no further events follow for the session.
    SessionTerminated {
        /// Completion, failure or cancellation reason.
        reason: String,
    },
}

#[derive(Clone)]
struct AppState {
    started: Instant,
    tiny: Option<Arc<tiny::Experiment>>,
    tiny_echo: Option<Arc<tiny::Experiment>>,
}

/// Build the local API and embedded static dashboard with one execution path.
pub fn router() -> Router {
    build_router(None, None)
}
/// Enable the bounded random-Qwen CPU experiment explicitly.
pub fn experimental_router() -> Result<Router, String> {
    experimental_router_at(std::path::PathBuf::from(".tesslot/tiny-qwen"))
}
/// Operator-selected experiment checkpoint root; HTTP clients cannot choose paths.
pub fn experimental_router_at(directory: std::path::PathBuf) -> Result<Router, String> {
    let base = Arc::new(tiny::Experiment::new(directory.clone(), false)?);
    let echo = Arc::new(tiny::Experiment::new(directory.join("echo"), true)?);
    Ok(build_router(Some(base), Some(echo)))
}
fn build_router(
    tiny: Option<Arc<tiny::Experiment>>,
    tiny_echo: Option<Arc<tiny::Experiment>>,
) -> Router {
    let mut api = Router::new()
        .route("/system/status", get(status))
        .merge(tiny::routes(tiny::Kind::Base))
        .merge(tiny::routes(tiny::Kind::Echo))
        .route("/system/devices", get(devices))
        .route("/system/resources", get(resources))
        .route("/inference/models", get(models))
        .route("/schemas/openapi.json", get(openapi))
        .route("/schemas/events.json", get(events_schema));
    for path in [
        "/inference/sessions",
        "/inference/sessions/{id}/input",
        "/inference/sessions/{id}/cancel",
        "/training/datasets",
        "/training/jobs",
        "/training/jobs/{id}/pause",
        "/training/jobs/{id}/resume",
        "/training/jobs/{id}/stop",
        "/agents/sessions",
        "/agents/tasks",
    ] {
        api = api.route(path, post(unimplemented));
    }
    for path in [
        "/inference/sessions/{id}",
        "/inference/sessions/{id}/events",
        "/inference/sessions/{id}/socket",
        "/training/datasets",
        "/training/jobs",
        "/training/jobs/{id}",
        "/agents/tasks/{id}",
        "/agents/tasks/{id}/events",
    ] {
        api = api.route(path, get(unimplemented));
    }
    Router::new()
        .nest("/api/window-v1", api)
        .route(
            "/",
            get(|| async { Html(include_str!("../../../dashboard/index.html")) }),
        )
        .route(
            "/app.js",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
                    include_str!("../../../dashboard/app.js"),
                )
            }),
        )
        .route(
            "/style.css",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
                    include_str!("../../../dashboard/style.css"),
                )
            }),
        )
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(axum::middleware::from_fn(security_headers))
        .with_state(Arc::new(AppState {
            started: Instant::now(),
            tiny,
            tiny_echo,
        }))
}

/// Serve validated loopback configuration until Ctrl-C; no production tool executor.
pub async fn serve(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    serve_selected(config, false).await
}
/// Start the same public server with the explicitly opt-in tiny experiment.
pub async fn serve_experimental(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    serve_selected(config, true).await
}
async fn serve_selected(config: &Config, tiny: bool) -> Result<(), Box<dyn std::error::Error>> {
    config.validate()?;
    let router = if tiny {
        experimental_router().map_err(std::io::Error::other)?
    } else {
        router()
    };
    let listener = tokio::net::TcpListener::bind(config.bind).await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

async fn status(State(state): State<Arc<AppState>>) -> Json<SystemStatus> {
    Json(SystemStatus {
        application_version: env!("CARGO_PKG_VERSION").into(),
        api_version: API_VERSION.into(),
        architecture_version: ARCHITECTURE_VERSION.into(),
        uptime_ms: state.started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        status: "ready".into(),
        capabilities: tesslot_core::capabilities()
            .into_iter()
            .chain(std::iter::once((
                "tiny_qwen_experiment",
                if state.tiny.is_some() {
                    "available"
                } else {
                    "unavailable"
                },
            )))
            .map(|(k, v)| (k.into(), v.into()))
            .collect(),
    })
}
async fn devices() -> Json<serde_json::Value> {
    Json(
        serde_json::json!({"devices":[{"id":"cpu","kind":"cpu","architecture":std::env::consts::ARCH,"os":std::env::consts::OS,
        "available_parallelism":std::thread::available_parallelism().ok().map(|n| n.get())}],"cuda_discovery":"unavailable"}),
    )
}
async fn resources() -> Json<serde_json::Value> {
    Json(
        serde_json::json!({"scheduler":"unimplemented","reservations":[],"memory_measurements":null}),
    )
}
async fn models() -> Json<serde_json::Value> {
    Json(serde_json::json!({"models":[],"model_loading":"unimplemented"}))
}
async fn openapi() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/json")],
        include_str!("../../../docs/openapi.json"),
    )
}
async fn events_schema() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/schema+json")],
        include_str!("../../../docs/events.schema.json"),
    )
}

fn error(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(ApiError {
            code: code.into(),
            message: message.into(),
            details: BTreeMap::new(),
        }),
    )
        .into_response()
}
async fn unimplemented() -> Response {
    error(
        StatusCode::NOT_IMPLEMENTED,
        "capability_unimplemented",
        "This execution backend is not implemented in milestone 1. Inspect system/status capabilities; use the CLI for CPU reference replay.",
    )
}
async fn not_found() -> Response {
    error(
        StatusCode::NOT_FOUND,
        "route_not_found",
        "Unknown route or API protocol version. Native routes start with /api/window-v1/.",
    )
}
async fn method_not_allowed() -> Response {
    error(
        StatusCode::METHOD_NOT_ALLOWED,
        "method_not_allowed",
        "HTTP method is not supported by this route.",
    )
}

async fn security_headers(request: Request, next: axum::middleware::Next) -> Response {
    // Restrict Host to loopback identities to prevent browser DNS rebinding.
    if let Some(host) = request.headers().get(header::HOST) {
        let allowed = host
            .to_str()
            .ok()
            .filter(|s| !s.contains('@'))
            .and_then(|s| s.parse::<axum::http::uri::Authority>().ok())
            .is_some_and(|a| matches!(a.host(), "localhost" | "127.0.0.1" | "[::1]"));
        if !allowed {
            return error(
                StatusCode::FORBIDDEN,
                "host_forbidden",
                "Only loopback Host headers are accepted.",
            );
        }
    }
    if !matches!(request.method().as_str(), "GET" | "HEAD" | "OPTIONS")
        && let Some(origin) = request.headers().get(header::ORIGIN)
    {
        let same = origin
            .to_str()
            .ok()
            .and_then(|o| o.parse::<axum::http::Uri>().ok())
            .is_some_and(|o| {
                o.scheme_str() == Some(request.uri().scheme_str().unwrap_or("http"))
                    && o.authority().map(|a| a.as_str())
                        == request
                            .headers()
                            .get(header::HOST)
                            .and_then(|h| h.to_str().ok())
            });
        if !same {
            return error(
                StatusCode::FORBIDDEN,
                "origin_forbidden",
                "Cross-origin mutations are forbidden.",
            );
        }
    }
    let mut response = next.run(request).await;
    response.headers_mut().insert(header::CONTENT_SECURITY_POLICY,header::HeaderValue::from_static("default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"));
    response.headers_mut().insert(
        header::X_CONTENT_TYPE_OPTIONS,
        header::HeaderValue::from_static("nosniff"),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-store"),
    );
    response
}
