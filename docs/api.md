# Native API — window-v1

The protocol root is `/api/window-v1/`, independent of Tesslot's application
version. [OpenAPI 3.1](openapi.json) specifies HTTP routes;
[window-events/1](events.schema.json) describes proposed streaming envelopes.
The schemas are checked in and served directly under `schemas/`; their Rust
counterparts live in `tesslot-api`. Execution schemas are design contracts,
not a claim that their backends accept/execute jobs yet.

## Operational endpoints

| Method and relative path | Current behavior |
|---|---|
| GET `system/status` | Live monotonic uptime, app/API/architecture versions, subsystem availability |
| GET `system/devices` | Actual CPU architecture/OS/available parallelism; CUDA discovery unavailable |
| GET `system/resources` | Scheduler unimplemented, empty reservations, unavailable memory metrics |
| GET `inference/models` | Empty registered-model list; loading unimplemented |
| GET `schemas/openapi.json` | OpenAPI document |
| GET `schemas/events.json` | Proposed event JSON schema |

The explicitly listed inference session, training dataset/job and agent task
routes in OpenAPI return **501 capability_unimplemented**, including SSE/WebSocket
paths. They do not create state, validate execution requests, contact providers,
download models or execute tools. Unknown routes/versions return structured 404;
unsupported methods return structured 405. The dashboard is at `/`.

Examples (the shown payloads are documentation examples, not measured outputs):

```sh
cargo run -- serve
curl http://127.0.0.1:8080/api/window-v1/system/status
curl -i -X POST http://127.0.0.1:8080/api/window-v1/inference/sessions \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"kidra-local","max_windows":8,"seed":42}'
```

```json
{"code":"capability_unimplemented","message":"This execution backend is not implemented in milestone 1. Inspect system/status capabilities; use the CLI for CPU reference replay.","details":{}}
```

Failure envelopes always have `code`, human-readable `message`, and structured
`details`. Future body validation uses 422, unknown logical entities 404, lifecycle
or resource conflicts 409, permissions 403, missing authentication 401, capacity
429 and internal failure 500. Error messages must exclude credentials and prompts.

## Event ordering and reconnection (planned)

Events identify logical session, monotonic session-local `sequence`, optional
tick/window/operation, schema version, typed category and data. Sequence numbers
start at one and totally order events within a session. They do not order unrelated
sessions or imply physical GPU placement. Tick completion is published only after
the shared D commit; tool wall-clock events may have `tick: null`. Tool output is
recorded separately from its later B-phase context insertion.

Initial categories cover opens, closes, predictions (including non-durable NOOP),
external insertions, completed ticks and terminal sessions. Tool/subagent/paused
categories will be added with their implementing backend and schema review; they
are not falsely advertised by the initial six-variant schema.

SSE will use `id: <sequence>`, `event: <type>` and JSON `data`. Reconnect with
`Last-Event-ID`; replay events strictly after that cursor. Retention expiry must
return an explicit cursor-expired/resync error, not quietly skip history. Heartbeats
are transport comments, not logical events. Slow clients have bounded queues;
disconnect and replay/resync instead of dropping model/tool events silently.

WebSockets will use the same ordered event envelopes and correlated command IDs.
Commands require per-operation authorization and acknowledgments. Retry/idempotency
rules must be finalized before mutation endpoints are enabled. Browser disconnect
does not cancel a session/job. Explicit cancellation is acknowledged at a safe
logical boundary. A `session_terminated` event ends the event stream permanently;
no predictions appear after it. These transports are **not implemented** yet.

## Authentication and compatibility

Milestone 1 permits local read-only access and rejects all non-loopback binds.
No credential/configuration/tool mutation route exists. No permissive CORS headers
are emitted. Host restrictions protect against DNS rebinding. This is not remote
multi-user authentication. See [security](security.md).

Before remote use, add revocable bearer credentials, HTTPS deployment guidance and
scopes separating system read/admin, inference read/mutate, training read/control,
agent tasks and individually authorized tool dispatch. Prompt submission never
grants host execution. Keep tokens out of query strings; define a suitable browser
streaming authentication mechanism before implementing EventSource/WebSockets.

Additive optional response fields may preserve `window-v1`; changes to required
fields, ordering, visibility, loss interpretation, event meaning or lifecycle
require explicit protocol/schema version review. Readers must reject unsupported
artifact versions. Do not silently reinterpret older checkpoints or exports.
