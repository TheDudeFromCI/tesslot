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
