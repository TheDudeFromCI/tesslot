# Milestone 2 — local checkpoints, runtime placement and recomputation

The remaining implementation is available with CPU correctness gates. NVIDIA
qualification is still open: this cloud host has no GPU, and a skipped hardware
test is not a passing result. Burn 0.22.1 remains provisional until the model-level
single/two-device tests pass on real hardware.

## Complete local pretrained conversion

Models remain operator supplied. No download command, pretrained weights, or Python
runtime was added. The sole tracked SafeTensors file remains the 5,184-parameter
random Qwen fixture. `tesslot-artifacts::pretrained` supports:

- HF `config.json` and either `model.safetensors` or
  `model.safetensors.index.json` plus its referenced shards.
- Exact Qwen3 names, GQA shapes, tied/untied heads, finite payloads and an explicit
  aggregate decoded FP32 budget. Headers are inspected first; payloads are read
  one tensor at a time. Source arrays are collected for backend allocation;
  this is not an out-of-core weight loader.
- Duplicate index names, missing/extra tensors, wrong-shard assignments,
  incompatible shapes, missing shards, path traversal and escaping symlinks fail.
  Source index and complete shard digests are recorded.
- Existing tokenizer IDs and **every** original embedding/output row survive.
  Padded rows between tokenizer vocabulary size and `config.vocab_size` receive
  explicit reserved spellings. Distinct `<|kidra_bos|>`, `<|kidra_eos|>` and
  `<|kidra_noop|>` controls are appended after those rows. Existing base BOS/EOS
  spellings/IDs remain available; target model lifecycle controls use the new IDs.
- Only appended embedding/output rows and seven Kidra components are seeded.
  A tied output remains an alias to the embedding parameter, including new rows.
  New special spellings intentionally acquire special-token encoding semantics;
  existing token IDs are not renumbered. Sparse/ambiguous tokenizer IDs and
  existing Kidra control spellings require an explicit migration and are rejected.
- An atomically published `kidra-pretrained/1` directory contains config,
  tokenizer, SafeTensors and conversion metadata. Loading validates config/weight
  digests, normalized tokenizer identity, control mapping and initialized names.

Example (paths are supplied by the operator; the budget is deliberately explicit):

```sh
cargo run --locked -- convert-pretrained --input /path/to/qwen \
  --output /path/to/kidra --windows 4 --seed 42 --max-bytes 4294967296
cargo run --locked -- forward-qwen --input /path/to/qwen \
  --tokens 151643,64,65 --max-bytes 4294967296
```

`forward-qwen` runs the base causal model and reports an actual next-token ID;
its IDs must match the supplied base tokenizer. `Model::from_pretrained` loads
the complete converted multi-window model and binds its training identity.
The explicit-budget tensor loader removes the tiny dashboard's dimensional caps;
Tests exercise head dimension 128, vocabulary 300, hidden width 80 and context
capacity 32768; its opt-in short-context gate executes forward, checks causality
and applies an all-parameter training update without downloading model weights. This does not
establish 0.6B/1.7B capacity, performance or conversational quality. The existing
web experiments retain their original bounded random models; general pretrained
HTTP inference is Milestone 5.

The subsequent [workstation log and finite-guard correction](cuda-finite-checks.md)
provide real initial model forward/gradient evidence in all precision modes.
Optimizer updates and checkpoint continuation remain pending the corrected rerun.

## Device ownership and precision

The [CUDA precision correction](cuda-precision.md) supersedes the original
assumption that FP32 storage implied strict FP32 GEMM. `f32` now uses a slower
reference multiply/reduce on CUDA; `tf32` explicitly permits the native accelerated
path. CPU FP32 and BF16/FP16 keep their existing paths. Corrected hardware results
are pending a workstation rerun.

`qwen::runtime::Plan` assigns a root and exactly one owner to each layer. Local
Qwen weights and projected KV stay on that layer's owner. Residual activations
move at layer boundaries. Embeddings, final norm/head and shared exchange weights
have a root owner; exchange parameters are differentiably transferred to each
layer where used. This is a capability implementation, not optimized replication
or an automatic placement planner. Shared transfers can be expensive over PCIe.
Windows remain synchronized together on each layer.

Example plan for a **three-layer test model**, with one layer on GPU 0 and two
on GPU 1 (use the actual layer count for other models):

```json
{
  "root": {"kind": "cuda", "index": 0},
  "layers": [{"kind": "cuda", "index": 0},
             {"kind": "cuda", "index": 1},
             {"kind": "cuda", "index": 1}],
  "precision": "bf16",
  "parameter_budget_bytes": 134217728
}
```

CUDA is optional (`--features cuda` on the CLI or tensor crate). CPU FP32 is the
default. Explicit unavailable devices fail with an actionable error; hardware
scripts skip CUDA when discovery fails. There is no implicit offload or VRAM pool.
Mixed CPU/CUDA plans are rejected because Burn distributed backward requires a
common compute backend; CPU optimizer offload is not implemented.
The current budget applies separately to each unique target and counts FP32
parameters only. `estimate()` separately reports gradients, AdamW moments and
candidate update arrays. Activations, temporary casts, attention workspaces,
allocator overhead and checkpoint serialization remain unestimated: this is
**not** a full training-admission memory estimator or OOM recovery service.

BF16/FP16 select reduced-precision GEMMs for projections and attention.
CUDA `f32` is a numerical qualification path; use BF16 for intended large-model
work. Native FP32-storage GEMMs may use TF32 and require explicit `tf32` selection. Norms,
residuals, returned logits, parameter masters and moments stay FP32. FP16 is
unscaled, and nonfinite losses/gradients/updates abort; dynamic loss scaling is
future work. Scalar finite/norm reductions avoid downloading entire gradient
and parameter arrays for each update. Diagnostic gradient snapshots and saving
checkpoints deliberately copy data to the host. RoPE now uses elementwise
split-half rotations, avoiding per-position square rotation matrices.

Checkpoint writes remain atomic and checksummed. Runtime placement is recorded
as provenance; restore defaults to CPU, with an explicit larger per-file/decoded
budget for bigger checkpoints. Call `configure_runtime` afterward to move both
masters and optimizer moments. Existing caches are invalidated. Raw tensor
construction without a tokenizer identity cannot publish a misleading checkpoint.
Large checkpoint encoding still materializes host payloads and encoded bytes;
streamed/sharded checkpoint writing is future memory work.

## Exact recomputation

`train_remix_recomputed` replays the full durable causal prefix for each effective
target, computes only that target's vocabulary head, backpropagates its globally
normalized loss, and accumulates parameter gradients on their owners. One optimizer
update and sample-cursor advance occur after all targets. Earlier representations
still obey their own masks at **every layer**. Replaced/closed history needed by
surviving descendants remains attached inside each replay. There is no chunk
detachment or independent chunk optimization.

This is exact target-prefix replay, not whole-block activation checkpointing. It
avoids keeping all prediction branches/live vocabulary heads together at the cost
of repeated attention/forward work. It does not eliminate the durable-prefix graph
and is deliberately slow. On this host the tiny one-layer fixture measured
14,404 KiB peak process RSS / 0.103 s for ordinary replay and 12,020 KiB / 0.161 s
for recomputation, both with loss 4.59374523. These single-run process measurements
are workload-specific and do not establish GPU savings. The two-layer regression compares loss, every named
parameter gradient, AdamW updates and deterministic checkpoint continuation.
Tolerance is `3e-4 * (1 + |reference gradient|)` for different FP32 summation orders.
Run `scripts/measure-remix-memory.sh` for separate-process Linux peak RSS and actual
step durations. Process RSS includes the runtime and allocator; it is not GPU
memory and does not establish production memory savings.

A bounded opt-in local smoke trainer uses the complete conversion and **exports
compiled with that exact converted tokenizer**:

```sh
cargo run --locked -- train-remix-local /path/to/export.jsonl \
  --pretrained /path/to/kidra --tokenizer-file /path/to/kidra/tokenizer.json \
  --output /path/to/checkpoint --steps 2 --recompute --max-bytes 4294967296
# For CUDA add --features cuda before `--`, and --runtime /path/to/plan.json after it.
```

To resume, replace `--pretrained` with `--resume /path/to/checkpoint` and choose a
new output. This remains a synchronous experiment with a 64 MiB export limit,
512 rows/sample and at most 64 window slots, not the scheduled training service.
The sequential whole-sample cursor, dataset fingerprint and AdamW state persist.
Schedules, accumulation, evaluation, production job control and full 0.6B runs
remain Milestone 3.

## Reproducible verification

Ordinary workspace tests include local conversion, exact source-row preservation,
shard validation, budget errors and converted-checkpoint load/restore. All numerical
experiments are opt-in with `scripts/test-milestone-2.sh`. It runs the previous
Qwen/echo/Remix gates, exact recomputation, the actual local-forward CLI, converted
local training/resumption, memory measurements and hardware discovery.

`pretrained-autograd.jsonl` is generated by the unmodified pinned Remix exporter
with the Rust-converted tokenizer and dedicated controls. Its independent fixture
manifest records the upstream revision, generator, tokenizer identity and file
checksum. See `scripts/generate-pretrained-fixture.py` for reproduction; Python
is only a fixture-development dependency.

`scripts/test-cuda.sh` now adds **model-level** FP32, BF16, FP16 and asymmetric
two-GPU tests to the existing small transfer probe. These compare dense/cached
logits, all parameter gradients, first-partition and shared-exchange gradients,
AdamW updates and device-independent checkpoint continuation. FP32 tolerance is
`4e-4 * (1 + |CPU|)`; BF16 is `0.04`, FP16 `0.01` with the same scaling, reflecting
quantization over three layers. These tiny-fixture tolerances are not production
quality guarantees. Compilation/kernel/numerical failures after hardware discovery
fail the script; dual-device execution skips explicitly when only one GPU exists.

This host cannot execute those tests. A normal CUDA build also encounters the
pinned LLVM bundler's hardcoded home-cache path on the cloud read-only filesystem.
A `DOCS_RS=1 cargo clippy -p tesslot-tensor --features cuda --all-targets
--locked -- -D warnings` source check skips native
LLVM setup and is only a Rust type-check, not a linked CUDA build. Do not use that
environment variable to run hardware tests. No dependency code or lockfile was
patched to bypass this limitation.

## Remaining qualification and Milestone 3

Before declaring Milestone 2 fully hardware-qualified, run `scripts/test-cuda.sh`
on the workstation and retain the device/version/results log. Burn's cross-device
autograd and reduced precision remain unverified until then. If these fail, keep
the CPU reference and artifact contracts and revisit Burn/Candle/LibTorch against
the concrete failure; do not assume another backend solves cross-device autograd.
Next implement correct gradient accumulation and trajectory-level evaluation,
then persisted training jobs/schedules/checkpoint lifecycle. Large pretrained
experiments require explicit local weights and resource measurements; layer-budget
planning, block checkpointing and PCIe transfer optimization remain Milestone 4.

## Verification of the original completion increment on this cloud host

- Formatting and workspace Clippy (`-D warnings`): pass.
- Ordinary CPU workspace: 57 passed, 20 computational tests intentionally ignored.
- `scripts/test-milestone-2.sh`: all 20 opt-in CPU tests passed, including API
  experiments, converted-model CLI training and exact checkpoint continuation.
- Separate-process memory example: completed in both modes with equal loss.
- CUDA-feature Clippy with `DOCS_RS=1`: pass as a source/type check only.
- Native CUDA compilation: blocked by the LLVM bundler cache on the read-only
  cloud filesystem. Actual single/two-GPU and BF16/FP16 execution: not verified;
  the hardware script reported an explicit skip.
- No large model weights were downloaded; no upstream repository was modified.
