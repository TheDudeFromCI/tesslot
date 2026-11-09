# Opt-in local model regression tests

Use one entry point for the permanent model-sized regression checks:

```sh
python3 scripts/test-pretrained.py \
  --model-dir "$HOME/AI Models/Kidra-Qwen3-0.6B" \
  --base-model-dir "$HOME/AI Models/Qwen3-0.6B-Base" \
  --gpu-uuid "$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)" \
  --precision tf32
```

The converted model and its tokenizer must use 24 window slots. The original HF
directory is optional; without it, base source inspection/forward are explicitly
skipped. No models are downloaded. Python 3.10+ standard library, the existing
Cargo/CUDA toolchain and the local weights are sufficient. TF32 in this example
is an explicit performance choice for hardware that supports it; the portable
default is strict FP32. No GPU model name or VRAM capacity is a pass criterion.

## Hardware and placement profiles

The same tests run on larger or smaller supported GPUs. Memory observations and
parameter counts come from the selected hardware/model, not a workstation preset.
The fixed fixtures describe model workloads and tokenizer identity, not hardware
requirements. Use `--fixtures-dir` for a regenerated compatible tokenizer collection.

Repeat `--gpu-uuid` in desired visible-ordinal order to select multiple physical
GPUs. Without a runtime file, layers are split in contiguous blocks across those
GPUs with shared/root parameters on ordinal zero. For asymmetric cards, supply an
explicit native runtime plan that assigns each layer and the root deliberately.
The native schema is described in [runtime.rs](../crates/tesslot-tensor/src/qwen/runtime.rs):
`root`, one entry per layer in `layers`, `precision`, and
`parameter_budget_bytes` (a weight-only bound per owner, separate from total state).
Devices unused by a profile are hidden; effective ordinals are remapped while
preserving physical ownership. Requested and effective maps are both recorded.

Several profiles can be tested in a single invocation:

```sh
python3 scripts/test-pretrained.py \
  --model-dir /path/to/converted-model \
  --gpu-uuid GPU-first --gpu-uuid GPU-second \
  --runtime-plan profiles/single-gpu.json \
  --runtime-plan profiles/split-gpus.json \
  --runtime-plan profiles/cpu.json \
  --runtime-plan profiles/cpu-layer-offload.json
```

These are caller-supplied plans with the actual model's layer count; filenames
are illustrative. Each profile reports independent results. One working profile
does not qualify the others. Precision comes from each file; without a file use
`--precision f32|tf32|bf16|f16`. Conflicting explicit precision settings are rejected.
Only the selected precision's numerical gates run. CUDA split profiles also run
the corresponding two-device gradient/checkpoint gate. Tests requiring hardware
that is absent do not become evidence that that hardware works.

Use `--cpu` for an explicit CPU-only invocation without CUDA/toolchain requirements.
CPU execution is not limited to tiny fixtures, though large models can be very
slow and need sufficient RAM and a raised `--max-bytes` import budget. The default
4 GiB import budget is configurable for larger artifacts and is not a VRAM cap.
There is no automatic CPU fallback when a GPU profile fails.

**Mixed CPU/CUDA layer offload is currently unsupported by the runtime.** Such a
profile is reported as unsupported and its execution tests skip; metadata still
records its requested ownership. A normal CPU contract test verifies the native
rejection before GPU initialization. This accounts for offload in the capability
matrix without claiming it works. Enabling it requires backend implementation
and then enabling positive numerical/placement tests for those profiles.

Memory is never pooled. Parameter bytes are assigned to the same root/layer
owners used by the native runtime. Admission and monitoring use each selected
GPU's free memory minus its reserve, and the CPU's available memory minus
`--host-reserve-mib` (default 1,024 MiB). A large GPU's spare memory cannot conceal
another GPU's shortfall. Unknown available CPU memory does not qualify AdamW
admission. Host/import/checkpoint staging and sequence-dependent workspaces are
additional to the known state categories. Per-device memory traces retain UUIDs
and visible-ordinal mapping through the profile and command provenance.
Known insufficient SGD state or dense host-import storage prevents that training
profile from launching. This does not claim that inference alone cannot fit.

This replaces the separate preparation, forward/backward smoke, minimal restart
and readiness-sweep wrappers. No compatibility wrappers are retained. The native
Rust commands remain the implementation exercised by the tests, and the authored
fixture generators remain development tools.

## Execution and scratch data

The tests use Python `unittest`, with named groups and individual capacity subtests.
They are opt-in: ordinary discovery does not load a model or initialize CUDA.
The suite shares one sequential collection so compilation and source inspection
are not repeated for every assertion. A built binary is reused across profiles.
It runs independent stages even after a
workload fails, then reports the test outcomes together.

All logs, raw memory traces, manifests and test checkpoints live in a unique
`tesslot-pretrained-*` directory under the OS temporary directory (`/tmp` on
Manjaro). The directory is removed at the end of successful and failed runs.
Checkpoint payloads are reclaimed between profiles so they do not accumulate.
Source models are never overwritten. A success prints a summary and leaves no
logs or checkpoints behind.

**Only a failed test exports diagnostics**, as one archive under `test-failures/`.
The archive contains logs, reports and checkpoint manifests; it excludes all model
and optimizer SafeTensors payloads. Its exact filename is printed. This directory
is gitignored and is not created on a successful run. Previous runs' exported
failure archives and pre-existing `work/` data are never deleted automatically.

Known runtime/context bounds and analytically insufficient budgets are skips.
The default capacity sweep discovers per-profile limits: a resource-limited
capacity case skips without declaring that Tesslot is broken on that hardware.
Nonfinite results, timeouts, source corruption and restart mismatches always fail;
executed mandatory backward/update/pilot checks also fail on resource exhaustion.

For a previously qualified configuration, use `--strict-capacity`: selected
capacity cases must fit, and a newly insufficient state budget fails before
launching. Explicit `--capacity-case ID` selections likewise require those
known-fit cases to pass. This catches memory regressions without hardcoding one
workstation's GPU model or requiring every workload to fit every device. Exit
codes are standard: zero for success with possible skips, one for failure;
invalid CLI arguments return two. Only failures export diagnostics.

The three temporary pilot checkpoints occupy about 7.2 GB for the ~600M model.
Before execution, the suite requires free space on the temporary filesystem for
four source-weight-file sizes plus 512 MiB margin. `/tmp` may be a RAM-backed
filesystem: if it lacks that space, select a suitable scratch filesystem with
`--tmp-dir /path/to/scratch`. Cleanup and failure-only export still apply there.
The suite does not change the machine's temporary-directory or mount settings.

## Checks included

- Existing small CUDA numerical/backend tests on the selected visible GPU, plus
  embedding gather/backward parity. Tests requiring another visible GPU can skip;
  this suite isolates the selected physical UUID for the model-sized runs.
- Original HF source-header inspection and one-token base forward, when supplied.
- Converted-model backward with finite nonzero gradients and unchanged counters.
- Stable SiLU values/gradients at large positive and negative inputs, with a small
  CUDA gate before model-sized execution. The backend activation avoids the NaN
  derivative caused by directly differentiating `x / (1 + exp(-x))` when its
  exponential overflows even though the forward value remains finite.
- One ephemeral clipped SGD update with finite losses/gradients/candidates.
- Ten capacity cases with multiple supervised targets and exact attached-prefix
  recomputation: two windows with 1/16/64/128 payload tokens each, eight with
  1/16/32, and 24 with 1/16. The tenth case has 24 windows each holding 16 payload
  tokens and all 24 predicting on the same tick. BOS is additional to payload.
- A four-trajectory training/four-trajectory held-out pilot covering copying,
  selection, updated observations and waiting with supervised NOOP/EOS controls.
- Twelve continuous updates versus six, checkpoint publication, fresh-process
  restore and six more updates. Final weights/manifests, update/sample sequence,
  cursor and held-out losses around restart must match exactly.
- Independent AdamW admission: the existing transactional FP32 update requires
  seven weight-sized arrays before activations/workspaces. If those categories
  fit after reserve, two actual updates run; otherwise the optimizer test skips.
- Before/after source and fixture checksums. Validation UUID overlap is rejected
  before updates; evaluation never updates parameters, moments or dataset cursor.
  Failure reports retain final hashes and the names of changed source files.

The full suite can take hours; cases include 24-target backward replay and two
pilot comparison branches. A process times out after two hours by default.
The 1,024 MiB sampled free-VRAM guard remains enabled. Sampling can miss transient
peaks and is not an allocator cap. Host availability is sampled too. CUDA phase
markers describe host control flow, not synchronized kernel timings.

The current native limit is **512 rows per complete trajectory**. The largest
compiled case has 456 rows and 408 live durable tokens at prediction. A separate
full-context admission result explicitly skips 24 windows at the model's full
configured context length; this is not a passing full-context test. AdamW's known
state requirement may also cause a skip on a smaller card; larger cards or
suitable layer splits can admit it. No nominal GPU capacity is assumed.
The synthetic pilot tests mechanics and held-out loss, not broad behavioral or
tool-use competency. Loss improvement is reported separately from correctness.

## Controls and CI

`--capacity-case ID` selects cases, `--pilot-steps N` selects 4..64 updates per
comparison branch, and `--timeout-seconds N` changes the per-process timeout.
`--failure-dir DIR` selects the failure archive destination. `--tesslot-bin PATH`
uses an explicitly supplied binary. `--json` emits full structured results to
stdout for CI without saving successful artifacts. There is no output-log-directory
option and no success-export mode.

CPU CI executes the ordinary Rust workspace suite and:

```sh
python3 -m unittest discover -s tests -t .
```

The Rust CLI integration test opts into this same runner with `--cpu`, a tiny
converted artifact and `--fixtures-dir fixtures/pretrained-sweep-tiny`. It checks
actual training/restart, optional base forward, 24 simultaneous targets, no CUDA
qualification, and cleanup with no successful exports. The CI instance uses small
fixtures; explicit CPU execution also accepts larger
compatible artifacts within the configured import budget. Process/artifact unit
tests require no GPU or large model.

The core suite lives in `tests/local_training/`; harness regression tests live in
`tests/test_local_training.py`. Fixture sources and manifests are under
`fixtures/pretrained-sweep-{24,tiny}` and `fixtures/pretrained-backward-{24,tiny}`.
They use the pinned real Trajectory Remix compiler at
`d2c8083536d56a33d876e2650ae855722d491566`; the actual tokenizer fingerprint is
`dd428951d25e3b94c907ea4a8e0de493ffd6c96045eace8a18cb56e1cb41cc78`.
The uploaded tokenizer and pretrained weights are not committed. Generators
`scripts/generate-training-sweep.py` and `scripts/generate-backward-fixture.py`
regenerate compatible native exports with the existing compiler dependencies.

Rerun this suite after architectural, attention/cache, optimizer or memory changes
that may affect correctness or capacity. The public inspection/conversion/training
CLI remains available for normal model work; test scratch is always temporary.
