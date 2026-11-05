# Tesslot architecture — milestone 1

Tesslot executes Kidra; it is not the model. The current implementation is a
small CPU correctness oracle and a local status service. It cannot train or run
pretrained Qwen3 weights. The public API never delegates numerical work to a UI.

## Dependency direction

```text
cli ──> api ──> core
 ├────> dataset ──> core
 └────> model ────> core
backend-probe ──> optional burn-tensor (independent experiment)
dashboard ──HTTP──> api
```

* `core`: typed rows, phases, windows, removal boundaries, causal predicate,
  configuration and device/layer placement selectors. No tensors or HTTP.
* `dataset`: bounded JSONL streaming, tokenizer compatibility, complete sample
  reconstruction, loss arrays, lifecycle validation, deterministic split/order keys.
* `model`: seeded f64 embeddings, RoPE, RMS normalization, dense single-head local
  attention, SwiGLU feed-forward and global attention after each local block.
  Separate prediction queries, immutable per-layer memory and weighted loss.
* `api`: Axum routes, stable errors and proposed execution/event types, embedded assets.
* `cli`: configuration, validation, numerical replay and server entry points.
* `backend-probe`: opt-in executable experiments, never imported by the engine.

No empty storage, agent, tool, training or scheduler crates exist. Add these when
their first executable feature is ready. There is no Python runtime dependency.

## Tick and window invariants

Window IDs are immutable and never reused within a sample. Slots identify learned
window embeddings, not array positions or physical GPUs. Capacity is configurable.
Local positions advance only for durable tokens, reset to surviving initialization
length on external content replacement, and remain independent between windows.

1. A opens windows and commits excluded-loss BOS/headers atomically.
2. B evaluates each external query before inserting its token, in input-rank order.
3. The after-B snapshot produces every D prediction and every durable D memory
   representation. Each prediction starts from a learned query embedding plus its
   window embedding; it never embeds its label.
4. D commits all durable outputs together. NOOP contributes loss without memory.
5. Closure and replacement eviction release unavailable reference cache entries.

The runtime is **teacher-forced replay**: lifecycle and target rows come from a
validated export. It computes real logits, not a production sampling/session loop.
Opening/closing is exercised by rows at their scheduled ticks, including late opens.

`core::visible_to` ports the pinned upstream predicate, including `removed_at`.
True means allowed. Loss exclusion is unrelated to visibility. EOS keeps context
available through its closing tick. D cannot read any same-tick D output, including
its own target. B reads A and earlier ranked B. A rows have no attention keys in
the upstream predicate; their initialization representation retains their own
embedding through the residual path. Future rows never become keys for past queries.

Every memory layer obeys that predicate. Masking only the output queries would
allow contaminated representations and is insufficient. Immutable insertion-time
memory means future tokens cannot rewrite old representations. Previously learned
information in another window is not retroactively erased by a closure/replacement.

## Numerical paths

Full forward builds all durable memory layer by layer with causal masks. Tick
execution builds the same per-layer states at insertion time and holds them in
`Cache`. It stores states rather than projected CUDA K/V and is intentionally
inefficient. Tests compare both paths and selected uncached prefix recomputations.

The oracle uses f64 and one head, tiny dimensions, no trainable normalization
scales, no optimizer, no autograd and no pretrained compatibility claims. It is
not an attempt to build a general-purpose tensor framework. The small private
matrix implementation is exclusively a test baseline. Burn remains isolated until
the capability gates in [backend evaluation](backend-evaluation.md) are met.

The reference query seed is an explicit provisional architectural decision. A
production implementation may use another causally valid prior-state query, but
must version the architecture and test teacher-forced/incremental equivalence.
The oracle's seed must not be equated with the upstream adapter's previous-token
metadata; upstream intentionally leaves the decoder design unspecified.

## Hardware, scheduling and persistence boundaries

`Device` and `LayerPlacement` describe contiguous, asymmetric layer groups.
Validation rejects missing coverage, zero budgets and unavailable CUDA execution.
Budgets are declarative in milestone 1; allocation enforcement and memory
estimation arrive with the scheduler. CPU offloading is never silently enabled.
Windows are not assigned to separate GPUs. VRAM remains per-device capacity.

Planned scheduler reservations account independently for parameters, gradients,
optimizer states, activations, checkpoints, temporaries and transfers. Initially
one train/eval update owns its reserved layer groups. Stop/pause takes effect only
between completed optimizer updates; cancellation releases reservations.

SQLite with transactional migrations will store projects, tasks, operations, event
cursors and job manifests. Artifact/model blobs live outside the database.
Agent persistence is distinct from rebuildable numerical cache state. Milestone 1
does not open a database or claim restart recovery.

## Security and observability

The server binds loopback only, checks Host, serves same-origin assets under CSP
and exposes no tool/filesystem/shell executor. Remote configuration fails before
binding. API status uses monotonic elapsed time and actual CPU parallelism.
Missing GPU/memory metrics are explicitly unavailable/null. JSON tracing logs
startup and operator errors; no training/inference telemetry is fabricated.

See [security](security.md), [API contract](api.md), [testing](testing.md), and
[decisions](decisions/0001-foundation.md) for extension gates.
