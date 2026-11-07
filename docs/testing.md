# Testing and numerical tolerances

Ordinary CI needs Rust and a CPU. The new `tesslot-tensor` CPU crate uses Burn/Flex; the
`burn-cpu` feature of `tesslot-backend-probe` remains optional; no test downloads model weights or
requires CUDA, Python, Node or a browser. Optional development fixture/schema/UI
checks have separate dependencies and are not engine runtime dependencies.

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --features tesslot-backend-probe/burn-cpu --locked -- -D warnings
cargo test --workspace --features tesslot-backend-probe/burn-cpu --locked
cargo build --locked
cargo run --locked -- demo
cargo run --locked -- validate fixtures/canonical.jsonl --tokenizer fixtures/tokenizer-identity.json
cargo run --locked -- validate fixtures/long.jsonl --tokenizer fixtures/tokenizer-identity.json
```

## Coverage

| Area | Executed checks |
|---|---|
| Core | Configuration parsing, unknown keys, dimension validation, loopback policy, layer coverage and unavailable CUDA rejection |
| Dataset | Actual exporter chunk reconstruction, single/chunked equality, 184041 upstream visibility pairs, hash integrity, tokenizer/special IDs, masks/labels/weights, malformed JSON/lifecycle/positions/removal, reordered/duplicate/missing first/middle/final chunks, metadata changes, line bounds, deterministic buffer shuffle/split keys, multi-sample and 2050-tick exports |
| Reference | Seed determinism, window identities and ablations, local RoPE, cross-window influence, separate label-free queries, 3-layer future/same-tick perturbations, supervised B self-target exclusion, full/incremental/cached/prefix-recomputed equality, EOS eviction, non-durable NOOP cache size, weighted loss, finite-difference logit derivatives, invalid inputs and loss overflow |
| Burn CPU probe | f64 matmul plus both parameter gradients, same-device copy, all-parameter finite differences through an attached prefix, explicit allowed-mask conversion and zero gradient to blocked values, operation-checkpoint output/gradient equality |
| HTTP | Versioned routes, truthful status/device/resource data, structured 404/405/501, unavailable mutation/streaming routes, CSP/no CORS, DNS-rebinding Host restrictions, schema serving and serialized event fields |
| Browser/schema smoke | Live local server, actual JSON responses against published schemas, example event schema, real Chromium navigation, live status/CPU, invalid hash fallback, mobile width and disconnect/reconnect |

Full/cache/prefix logits use **absolute tolerance 1e-12** on the tiny f64 CPU
oracle. They perform equivalent operations with identical masks and parameters;
this tight tolerance catches ordering/visibility mistakes without demanding
bitwise portability of floating-point math. Current default demo observed zero
max absolute error, but the acceptance test permits this explicit tolerance.
RoPE squared-norm tolerance is 1e-12; loss normalization uses 1e-14 on simple
analytic examples. Finite differences use epsilon 1e-6 with 1e-9 logit-gradient
and 1e-8 framework-parameter tolerances, allowing central-difference roundoff.
Operation-checkpoint parity uses 1e-12 on f64 CPU.

These are not FP16/BF16/GPU tolerances. Establish dtype- and operation-specific
tolerances from real experiments in milestone 2, while retaining exact boolean
causal-mask equality. A loose tolerance cannot excuse semantic differences.

## Optional live checks

Start `cargo run -- serve`, then:

```sh
# Development Python environment containing jsonschema:
python scripts/check-api-schemas.py
# Development Node environment containing Playwright; Chromium installed:
TESSLOT_CHROMIUM_PATH=/usr/bin/chromium node scripts/dashboard-smoke.cjs
```

The browser check saves `/tmp/tesslot-dashboard.png`. It waits for hash navigation
and polling transitions; it does not substitute local fixture responses for the
live API. It temporarily aborts one status request to verify disconnection, then
restores real network access to verify reconnection.

## Unsupported test areas

No NVIDIA device/driver utility is exposed here. CPU/GPU parity, BF16/FP16 CUDA,
single/two-GPU gradients, memory estimates/allocation/OOM recovery, actual PCIe
profiling and heterogeneous GPU compatibility **were not tested**. Do not add
empty passing GPU tests. The new optional hardware gates contain executable
forward/backward/update assertions and report absence explicitly; see
[hardware tests](hardware-tests.md). Ordinary CI continues to run CPU-only.

The bounded echo/corpus and [Qwen/Remix](qwen-remix-training.md) suites now test
actual updates, CPU model/optimizer checkpoint continuation, cursor resumption
and cache ownership/isolation. Transactional job persistence/migrations, packed
batches, production tool permissions/cancellation, remote authentication,
SSE event replay and WebSocket backpressure remain unimplemented. Their acceptance suites are gates in [the roadmap](roadmap.md).

Milestone 2 adds full-reference parameter derivatives, attached chunks, operation
checkpointing, causal tensor perturbations and Qwen config checks; see
[the current numerical gate results](milestone-2.md).

Optional CUDA tests now have an executable harness and default to ignored.
[Hardware test instructions](hardware-tests.md) describe automatic CPU fallback,
explicit skip messages and the exact limits of these small gates.

## Qwen/Remix opt-in gate

Run `./scripts/test-qwen-remix.sh` for dense/attached projected-KV logits and
gradients, two-layer finite differences, retained gradients after replacement,
weighted loss/NOOP/lifecycle causality, canonical tool/header exports and actual CLI
training/resume. These six computational tests are ignored by default; they do
not download weights or require CUDA. The [report](qwen-remix-training.md) states
limits and tolerances. `./scripts/test-tiny-qwen.sh` remains the echo/corpus
regression suite after the shared numerical-kernel extraction.
