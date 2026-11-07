//! Native HTTP contract and security regression tests.
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use tesslot_api::{Event, EventPayload, router};
use tower::ServiceExt;

async fn request(method: &str, path: &str) -> (StatusCode, serde_json::Value) {
    let response = router()
        .oneshot(
            Request::builder()
                .method(method)
                .uri(path)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn status_discovery_and_unknown_metrics_are_truthful() {
    let (code, status) = request("GET", "/api/window-v1/system/status").await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(status["api_version"], "window-v1");
    assert_eq!(status["capabilities"]["training"], "unimplemented");
    assert_eq!(status["capabilities"]["cpu_reference"], "available");
    let (code, devices) = request("GET", "/api/window-v1/system/devices").await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(devices["devices"][0]["kind"], "cpu");
    assert_eq!(devices["cuda_discovery"], "unavailable");
    let (_, resources) = request("GET", "/api/window-v1/system/resources").await;
    assert!(resources["memory_measurements"].is_null());
    let (_, models) = request("GET", "/api/window-v1/inference/models").await;
    assert_eq!(models["models"], serde_json::json!([]));
}

#[tokio::test]
async fn reserved_routes_do_not_create_execution_state() {
    for (method, path) in [
        ("POST", "/inference/sessions"),
        ("POST", "/training/jobs"),
        ("POST", "/agents/tasks"),
        ("GET", "/inference/sessions/abc/events"),
        ("GET", "/inference/sessions/abc/socket"),
        ("POST", "/inference/sessions/abc/cancel"),
    ] {
        let (code, body) = request(method, &format!("/api/window-v1{path}")).await;
        assert_eq!(code, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(body["code"], "capability_unimplemented");
    }
    for path in [
        "/v1/chat/completions",
        "/api/window-v2/system/status",
        "/api/window-v1/no-such-route",
    ] {
        let (code, body) = request("GET", path).await;
        assert_eq!(code, StatusCode::NOT_FOUND);
        assert_eq!(body["code"], "route_not_found");
    }
    let (code, body) = request("POST", "/api/window-v1/system/status").await;
    assert_eq!(code, StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(body["code"], "method_not_allowed");
}

#[tokio::test]
async fn dashboard_and_schemas_served_with_security_headers() {
    for (path, contains) in [
        ("/", "Execution starts here."),
        ("/app.js", "/api/window-v1/system/status"),
        ("/style.css", "color-scheme"),
    ] {
        let response = router()
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().contains_key("content-security-policy"));
        assert!(
            !response
                .headers()
                .contains_key("access-control-allow-origin")
        );
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert!(std::str::from_utf8(&body).unwrap().contains(contains));
    }
    let (_, oas) = request("GET", "/api/window-v1/schemas/openapi.json").await;
    assert_eq!(oas["openapi"], "3.1.0");
    assert!(oas["paths"]["/system/status"]["get"]["responses"]["200"].is_object());
    let (_, events) = request("GET", "/api/window-v1/schemas/events.json").await;
    assert_eq!(events["oneOf"].as_array().unwrap().len(), 6);
}

#[tokio::test]
async fn dns_rebinding_host_is_rejected_and_loopback_works() {
    for host in [
        "evil.example",
        "localhost.evil.example",
        "127.0.0.1.evil.example",
        "user@localhost",
    ] {
        let response = router()
            .oneshot(
                Request::builder()
                    .uri("/api/window-v1/system/status")
                    .header("Host", host)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN, "{host}");
    }
    for host in ["127.0.0.1:8080", "localhost:8080", "[::1]:8080"] {
        let response = router()
            .oneshot(
                Request::builder()
                    .uri("/api/window-v1/system/status")
                    .header("Host", host)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{host}");
    }
}

#[test]
fn event_envelope_preserves_order_and_durability_fields() {
    let event = Event {
        schema_version: "window-events/1".into(),
        session_id: "session-a".into(),
        sequence: 7,
        tick: Some(4),
        window_id: Some("reasoning".into()),
        operation_id: None,
        payload: EventPayload::TokenGenerated {
            token_id: 3,
            durable: false,
        },
    };
    let value = serde_json::to_value(event).unwrap();
    assert_eq!(value["sequence"], 7);
    assert_eq!(value["type"], "token_generated");
    assert_eq!(value["data"]["durable"], false);
    let _: Event = serde_json::from_value(value).unwrap();
}

#[tokio::test]
async fn experiment_disabled_and_cross_origin_mutations_are_rejected() {
    let (code, _) = request(
        "POST",
        "/api/window-v1/training/experiments/tiny-qwen-echo-corpus/checkpoint",
    )
    .await;
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    let (code, _) = request(
        "POST",
        "/api/window-v1/training/experiments/tiny-qwen-echo/checkpoint",
    )
    .await;
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    let (code, _) = request(
        "POST",
        "/api/window-v1/training/experiments/tiny-qwen/checkpoint",
    )
    .await;
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    let response = router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/window-v1/training/experiments/tiny-qwen/stop")
                .header("Host", "127.0.0.1:8080")
                .header("Origin", "https://evil.example")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
#[ignore = "opt-in tiny Qwen HTTP generation/training/checkpoint workflow"]
async fn tiny_experiment_http_workflow() {
    let root = tempfile::tempdir().unwrap();
    let app = tesslot_api::experimental_router_at(root.path().to_path_buf()).unwrap();
    async fn call(
        app: axum::Router,
        method: &str,
        path: &str,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        let response = app
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(format!("/api/window-v1/{path}"))
                    .header("Content-Type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap())
    }
    let (status, v) = call(
        app.clone(),
        "POST",
        "inference/experiments/tiny-qwen/generate",
        serde_json::json!({"prompt":"a ","max_tokens":8}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(!v["token_ids"].as_array().unwrap().is_empty());
    let (status, _) = call(
        app.clone(),
        "POST",
        "training/experiments/tiny-qwen/jobs",
        serde_json::json!({"steps":2}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    for _ in 0..1000 {
        let (_, s) = call(
            app.clone(),
            "GET",
            "inference/experiments/tiny-qwen/status",
            serde_json::Value::Null,
        )
        .await;
        if s["busy"] == false {
            assert!(s["error"].is_null(), "{s}");
            assert_eq!(s["metrics"].as_array().unwrap().len(), 2);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let (status, saved) = call(
        app.clone(),
        "POST",
        "training/experiments/tiny-qwen/checkpoint",
        serde_json::Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(saved["step"], 2);
    let (status, loaded) = call(
        app,
        "POST",
        "training/experiments/tiny-qwen/restore",
        serde_json::Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(loaded["step"], 2);
    let restarted = tesslot_api::experimental_router_at(root.path().to_path_buf()).unwrap();
    let (status, value) = call(
        restarted,
        "POST",
        "training/experiments/tiny-qwen/restore",
        serde_json::Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(value["step"], 2);
}

#[tokio::test]
#[ignore = "opt-in two-window echo HTTP workflow"]
async fn echo_experiment_http_workflow() {
    let root = tempfile::tempdir().unwrap();
    let app = tesslot_api::experimental_router_at(root.path().to_path_buf()).unwrap();
    async fn call(
        app: axum::Router,
        method: &str,
        path: &str,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        let r = app
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(format!("/api/window-v1/{path}"))
                    .header("Host", "127.0.0.1:8080")
                    .header("Origin", "http://127.0.0.1:8080")
                    .header("Content-Type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = r.status();
        let bytes = r.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap())
    }
    let generate = "inference/experiments/tiny-qwen-echo/generate";
    let status_path = "inference/experiments/tiny-qwen-echo/status";
    let (status, before) = call(
        app.clone(),
        "POST",
        generate,
        serde_json::json!({"seed":42}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(before["replay"]["source"], "jbkquuxmrgvc");
    assert_eq!(before["replay"]["ticks"].as_array().unwrap().len(), 13);
    assert_eq!(
        before["replay"]["echo_token_ids"].as_array().unwrap().len(),
        12
    );
    let (_, s) = call(app.clone(), "GET", status_path, serde_json::Value::Null).await;
    assert_eq!(s["windows"], 2);
    let (status, _) = call(
        app.clone(),
        "POST",
        "training/experiments/tiny-qwen-echo/jobs",
        serde_json::json!({"steps":2,"seed":42}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let mut finished = false;
    for _ in 0..1000 {
        let (_, s) = call(app.clone(), "GET", status_path, serde_json::Value::Null).await;
        if s["busy"] == false {
            assert!(s["error"].is_null(), "{s}");
            assert_eq!(s["metrics"].as_array().unwrap().len(), 2);
            assert_eq!(s["echo_result"]["step"], 2);
            finished = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(finished, "echo job timed out");
    let (_, trained) = call(
        app.clone(),
        "POST",
        generate,
        serde_json::json!({"seed":42}),
    )
    .await;
    let (status, _) = call(
        app.clone(),
        "POST",
        "training/experiments/tiny-qwen-echo/checkpoint",
        serde_json::Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let restarted = tesslot_api::experimental_router_at(root.path().to_path_buf()).unwrap();
    let (status, value) = call(
        restarted.clone(),
        "POST",
        "training/experiments/tiny-qwen-echo/restore",
        serde_json::Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(value["step"], 2);
    let (_, restored) = call(
        restarted.clone(),
        "POST",
        generate,
        serde_json::json!({"seed":42}),
    )
    .await;
    assert_eq!(trained, restored);
    let (_, base) = call(
        restarted,
        "GET",
        "inference/experiments/tiny-qwen/status",
        serde_json::Value::Null,
    )
    .await;
    assert_eq!(base["windows"], 1);
    assert_eq!(base["checkpoint_available"], false);
    for body in [
        serde_json::json!({"seed":-1}),
        serde_json::json!({"seed":42,"pattern":"abc"}),
    ] {
        let (status, error) = call(app.clone(), "POST", generate, body).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(error["code"], "invalid_request");
    }
}

#[tokio::test]
#[ignore = "opt-in corpus AdamW sampling/evaluation/restart/stop HTTP workflow"]
async fn corpus_experiment_http_workflow() {
    let root = tempfile::tempdir().unwrap();
    let app = tesslot_api::experimental_router_at(root.path().to_path_buf()).unwrap();
    async fn call(
        app: axum::Router,
        method: &str,
        path: &str,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        let response = app
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(format!("/api/window-v1/{path}"))
                    .header("Host", "127.0.0.1:8080")
                    .header("Origin", "http://127.0.0.1:8080")
                    .header("Content-Type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let code = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (code, serde_json::from_slice(&bytes).unwrap())
    }
    const STATUS: &str = "inference/experiments/tiny-qwen-echo-corpus/status";
    const TRAIN: &str = "training/experiments/tiny-qwen-echo-corpus/jobs";
    const EVAL: &str = "training/experiments/tiny-qwen-echo-corpus/evaluate";
    const SAVE: &str = "training/experiments/tiny-qwen-echo-corpus/checkpoint";
    async fn idle(app: axum::Router) -> serde_json::Value {
        for _ in 0..1500 {
            let (_, v) = call(app.clone(), "GET", STATUS, serde_json::Value::Null).await;
            if v["busy"] == false {
                assert!(v["error"].is_null(), "{v}");
                return v;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("corpus job did not reach a safe boundary in time")
    }
    let (_, initial) = call(app.clone(), "GET", STATUS, serde_json::Value::Null).await;
    assert_eq!(initial["optimizer"], "adamw-f32/1");
    assert_eq!(initial["corpus"]["completed_samples"], 0);
    for body in [
        serde_json::json!({"steps":4001}),
        serde_json::json!({"steps":1,"seed":42}),
    ] {
        let (code, _) = call(app.clone(), "POST", TRAIN, body).await;
        assert_eq!(code, StatusCode::UNPROCESSABLE_ENTITY);
    }
    for body in [
        serde_json::json!({"split":"train","samples":1}),
        serde_json::json!({"split":"test","samples":17}),
    ] {
        let (code, _) = call(app.clone(), "POST", EVAL, body).await;
        assert_eq!(code, StatusCode::UNPROCESSABLE_ENTITY);
    }
    let (code, _) = call(app.clone(), "POST", TRAIN, serde_json::json!({"steps":2})).await;
    assert_eq!(code, StatusCode::ACCEPTED);
    let trained = idle(app.clone()).await;
    assert_eq!(trained["corpus"]["completed_samples"], 2);
    assert_eq!(trained["echo_result"]["split"], "development");
    assert_eq!(trained["echo_result"]["samples"], 8);
    let (code, test) = call(
        app.clone(),
        "POST",
        EVAL,
        serde_json::json!({"split":"test","samples":2}),
    )
    .await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(test["target_weight"], 24);
    assert_eq!(test["cases"][0]["seed"], "9223372036854775808");
    assert_eq!(test["model_step"], 2);
    let (code, _) = call(app.clone(), "POST", SAVE, serde_json::Value::Null).await;
    assert_eq!(code, StatusCode::OK);
    let restarted = tesslot_api::experimental_router_at(root.path().to_path_buf()).unwrap();
    let (code, saved) = call(
        restarted.clone(),
        "POST",
        "training/experiments/tiny-qwen-echo-corpus/restore",
        serde_json::Value::Null,
    )
    .await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(saved["step"], 2);
    let (_, resumed) = call(restarted.clone(), "GET", STATUS, serde_json::Value::Null).await;
    assert_eq!(resumed["corpus"], trained["corpus"]);
    let (_, retest) = call(
        restarted.clone(),
        "POST",
        EVAL,
        serde_json::json!({"split":"test","samples":2}),
    )
    .await;
    assert_eq!(
        retest, test,
        "evaluations must survive AdamW checkpoint restoration"
    );
    let (code, _) = call(
        restarted.clone(),
        "POST",
        TRAIN,
        serde_json::json!({"steps":256}),
    )
    .await;
    assert_eq!(code, StatusCode::ACCEPTED);
    let (_, stop) = call(
        restarted.clone(),
        "POST",
        "training/experiments/tiny-qwen-echo-corpus/stop",
        serde_json::Value::Null,
    )
    .await;
    assert_eq!(stop["stop_requested"], true);
    let stopped = idle(restarted.clone()).await;
    let cursor = stopped["corpus"]["completed_samples"].as_u64().unwrap();
    assert!(cursor < 258);
    let (code, saved) = call(restarted, "POST", SAVE, serde_json::Value::Null).await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(saved["step"], cursor);
}
