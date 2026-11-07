# Tesslot

Tesslot is the Rust execution platform for **Kidra**, an experimental multi-window
transformer. Tesslot and the trainable Kidra model are separate components.

The foundation provides a deterministic CPU reference, native Trajectory Remix
validation, Qwen-derived numerical execution and opt-in full-parameter training
experiments, plus a local API/dashboard. User-supplied HF checkpoints can be loaded
and converted; production inference/training services and real tools are future work.
CUDA execution is optional and still awaits hardware qualification.

## Run

Requires Linux and Rust 1.95+ (tested here with 1.99.1). Python, Node, CUDA and
large model downloads are not needed to build or run the core.

```sh
cargo build --locked
cargo run --locked -- demo
cargo run --locked -- validate fixtures/canonical.jsonl --tokenizer fixtures/tokenizer-identity.json
cargo run --locked -- config > tesslot.json
cargo run --locked -- --config tesslot.json serve
```

Open **http://127.0.0.1:8080**. The dashboard polls the public API for actual uptime,
CPU information and backend availability. Inference, Agents, Training and
Evaluation navigation shows the implementation boundary. Execution routes return
structured 501 errors. Non-loopback binds fail until authentication exists.

`demo` performs actual FP32 multi-window teacher-forced forward passes without
recording an autograd graph, and reports loss, effective weights and decision
previews. Full/cached numerical agreement is verified in tests. It is not pretrained generation. `replay <export> --tokenizer <identity>`
uses the same oracle, with a 4096-row CLI safety limit. `validate` streams larger
exports and checks all chunks against an explicit trusted tokenizer contract.

## Check

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --features tesslot-backend-probe/burn-cpu --locked -- -D warnings
cargo test --workspace --features tesslot-backend-probe/burn-cpu --locked
```

The separate CPU backend probe requires no GPU:

```sh
cargo test -p tesslot-backend-probe --features burn-cpu --locked
```

Ordinary tests consume checked-in actual-exporter fixtures. Optional fixture
regeneration uses a disposable Python development environment, documented in
[trajectory compatibility](docs/trajectory-compatibility.md).

For automatic hardware skipping, run `./scripts/test.sh`. See
[optional NVIDIA tests and CPU fallback](docs/hardware-tests.md).

## Milestone 2 progress

The differentiable CPU model and native Qwen3 config inspection are now available.
`cargo run -- inspect-qwen-config examples/qwen3-0.6b.config.json` inspects
metadata without downloading weights.
See the [completion increment and commands](docs/milestone-2-completion.md), plus
[historical milestone-2 results](docs/milestone-2.md). Production
Qwen execution, CUDA and the training service remain unavailable. Native tokenizer
loading and SafeTensors inspection/import are described in the
[FP32 and artifact import increment](docs/milestone-2-import.md).

## Design and status

* [Model architecture diagram](docs/model-architecture.md)
* [Architecture](docs/architecture.md) and [technical decisions](docs/decisions/0001-foundation.md)
* [Numerical backend evaluation](docs/backend-evaluation.md) and [capability gates](docs/decisions/0002-backend-gates.md)
* [Training semantics](docs/training.md), [model conversion plan](docs/model-format.md), and [tool compatibility matrix](docs/tools.md)
* [Native API](docs/api.md), [OpenAPI](docs/openapi.json), [event schemas](docs/events.schema.json), and [security](docs/security.md)
* [Testing and tolerances](docs/testing.md), [milestone report](docs/milestone-1.md), and [roadmap](docs/roadmap.md)

The top risks remain causal attention, attached training gradients, training/cache
equivalence, bandwidth-constrained asymmetric GPUs and faithful export semantics.
No GPU functionality has been verified in this environment. MIT licensed.

Real base models must be supplied separately by the user. Only a tiny synthetic
5,184-parameter Qwen-shaped SafeTensors fixture is bundled for tests. See
[user-supplied models and conversion instructions](docs/base-models.md).

## Opt-in tiny Qwen experiment

`cargo run -- serve --tiny-experiment` enables real token generation, a repeating
`a b c` SGD experiment and checkpoint save/restore in the dashboard.
`./scripts/test-tiny-qwen.sh` runs the heavier tests explicitly; ordinary CI skips
them. See [experiment scope and checkpoint instructions](docs/tiny-qwen-experiment.md).

The same opt-in server now includes a [two-window echo experiment](docs/two-window-echo.md):
12 automatic random source letters, a trainable echo window, actual per-tick
replay/accuracy, and independent echo checkpoints. The combined opt-in test script
checks causal isolation, source gradients, learning, ablation and resume.

The [varied-pattern echo experiment](docs/echo-generalization.md) adds a separate
corpus, full-parameter AdamW, held-out development/test reports, and checkpointed
sampling/moments. With 600 updates it copies all 16 development and 16 test
patterns exactly on CPU. It uses the same opt-in server/test commands above.

A [native Qwen/Remix CPU training gate](docs/qwen-remix-training.md) now consumes
validated trajectory exports with attached projected KV across chunks, external-first
queries, weighted loss and resumable model/optimizer/dataset state. It is explicitly
synthetic/headless: `cargo run -- train-remix-demo --help`. Numerical and CLI
experiments remain opt-in through `./scripts/test-qwen-remix.sh`.
