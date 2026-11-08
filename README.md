# Tesslot

Tesslot is the Rust execution platform for **Kidra**, an experimental multi-window
transformer. Tesslot and the trainable Kidra model are separate components.

The foundation provides a deterministic CPU correctness reference, native Trajectory
Remix export validation, and a local API/dashboard foundation. It does **not** yet
execute pretrained Qwen weights, execute tools, or run CUDA. Tiny CPU
full-parameter training is available through the CLI.

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

The only optional feature is a CPU Burn experiment. It remains outside normal
CLI/server dependencies and requires no GPU:

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
See [milestone-2 results and remaining gates](docs/milestone-2.md). Production
Qwen execution, CUDA and the training service remain unavailable. Native tokenizer
loading and SafeTensors inspection/import are described in the
[FP32 and artifact import increment](docs/milestone-2-import.md).

## Milestone 3 progress

Tiny CPU full-parameter SFT now supports SGD/AdamW, warmup/decay,
weighted gradient accumulation, clipping, deterministic split/shuffle, evaluation
and final SafeTensors weight export:

```sh
cargo run --locked -- train fixtures/autograd.jsonl --tokenizer fixtures/tokenizer-identity.json --training-config examples/train-tiny.json --output tiny-trained.safetensors
```

The example trains one trajectory for four updates. Choose a new output path for
each run. Final weights are separate from resumable epoch-boundary checkpoints, enabled
with `--checkpoint-dir` and loaded with `--resume`. Persistent jobs, pretrained
training and GPU feasibility remain future increments. See
[Milestone 3 implementation and limits](docs/milestone-3.md).

The initial 0.6B memory preflight is also available via `preflight-training`.
See [3c assumptions, command and remaining experiment](docs/milestone-3c.md).
It estimates per-device state and checks optional local source headers; pretrained
training remains blocked on the open Qwen/CUDA gates.

For Manjaro local source/hardware checks and a tiny training/restart smoke run,
see [local training preparation](docs/local-training-preparation.md).
This handoff does not launch pretrained Qwen CUDA training.

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
