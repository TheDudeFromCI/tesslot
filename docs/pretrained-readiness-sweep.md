# One-sweep pretrained readiness tests

After the minimal SGD/update/restart test passes, run the remaining capacity and
pilot tests unattended with one command. Python 3.10+ standard library, Cargo and
the existing working CUDA toolchain are sufficient. No tokenizer package, model
download, live simulator, provider calls or manual trajectory generation is needed.

```sh
python3 scripts/pretrained-readiness-sweep.py \
  --model-dir "/path/to/Kidra-Qwen3-0.6B" \
  --gpu-uuid "GPU-your-3060-uuid" \
  --output-dir work/pretrained-sweep-01
```

Use the converted 24-slot artifact and a new output directory. The helper builds
once, runs the tiny CUDA embedding gather/backward gate once, then launches each
case in a separate process, sequentially. It keeps the **1,024 MiB VRAM reserve**
and logs GPU usage plus Linux host-memory availability about every 250 ms.
Polling can miss transient peaks and is not an allocator cap. Source artifact
files are hashed before/after and never overwritten. The per-file/model load
budget defaults to 4 GiB; it is separate from the VRAM budget.

The helper automatically creates `work/pretrained-sweep-01-logs.tar.gz` at the end,
including on handled failures. **Upload that archive.** It contains the report,
logs, memory traces and checkpoint manifests; it excludes all SafeTensors weights
and optimizer payloads. Keep the actual pilot checkpoints locally.

## Capacity cases

Each of the nine base cases performs one full-parameter clipped SGD update with **two supervised
targets in parallel**, using exact target-prefix recomputation and full gradient
accumulation. This exercises the multi-target path needed beyond the earlier
single-target diagnostics. Window capacity stays 24 throughout.

| Active windows | External payload tokens per window | Live durable tokens at prediction, including BOS | Complete trajectory rows |
|---:|---:|---:|---:|
| 2 | 1 | 4 | 8 |
| 2 | 16 | 34 | 38 |
| 2 | 64 | 130 | 134 |
| 2 | 128 | 258 | 262 |
| 8 | 1 | 16 | 32 |
| 8 | 16 | 136 | 152 |
| 8 | 32 | 264 | 280 |
| 24 | 1 | 48 | 96 |
| 24 | 16 | 408 | 456 |

All declared windows are open and populated before the supervised tick. A tenth
case, `capacity-w24-t016-all-generating`, uses the last row's context size with
**all 24 windows predicting simultaneously**, testing 24 weighted targets from the
same pre-D snapshot. Exact target-prefix recomputation computes these gradients
sequentially while preserving that parallel objective; it is not 24 simultaneous
GPU streams. This case can take substantially longer than the two-target cases.
Native compiler-assigned slots are preserved. All windows close
with masked EOS after the prediction. Capacity cases discard updated weights
explicitly with `--no-checkpoint`, avoiding ten extra model-sized files.

The runtime currently admits **at most 512 complete rows per trajectory**. A
separate `full-context-admission` result records the requested 24 windows at the
model's full configured context length and reports `unsupported_by_runtime`.
The sweep does not allocate that enormous graph, increase the admission bound or
claim a smaller case qualifies full context. Cases exceeding the supplied model's
position limit are also reported as unsupported. Expanding this limit and making
large full contexts practical requires subsequent memory/runtime engineering.

A reserve breach terminates the current child. Allocator OOMs remain explicit
`out_of_memory` results; numerical errors and timeouts remain failures. Later
independent cases still run. Failed cases never become passing capacity results.

## Multi-trajectory pilot and restart

The pilot uses four training trajectories and four disjoint held-out trajectories
with different UUIDs and values. It covers copying a value, selecting one of two
values, reading an update and waiting with a supervised non-durable NOOP until an
external observation arrives. Answers have multiple target tokens, EOS targets
have fractional weight and observations have zero loss. The pilot is curated
synthetic mechanics data, not broad Kidra competency, tool-use or free-running
behavioral validation. It does not run external specialists or tools.

The trainer evaluates the held-out set before/after each process without autograd,
optimizer updates, dataset rebinding or sampler advance. Train/validation UUID
overlap is rejected before any update. Reported validation loss aggregates weighted
numerators and effective target weights consistently across complete trajectories.

Three processes compare:

1. Twelve continuous updates from the converted source, then checkpoint.
2. Six updates from the same source, then checkpoint and process exit.
3. A fresh process restoring step/cursor six, then six more updates and checkpoint.

The four-sample dataset makes this a real cursor check: step six resumes partway
through the next dataset pass. The comparison checks exact serialized final
weights/manifests, the complete update/sample sequence and held-out losses around
the restart. Finite held-out loss is a mechanical gate; whether it improved is
reported separately and is not silently treated as model-quality success.

The three saved pilot checkpoints occupy about **7.2 GB** for 600,299,520 FP32
parameters. Admission requires four source-weight-file sizes plus 512 MiB free
disk to allow margin. Checkpoint save/restore may require substantial host RAM;
host availability is sampled but not attributed solely to the child. Checkpoints
are saved at process endpoints, not periodically or automatically on interruption.

## AdamW admission and optional execution

AdamW is assessed independently of SGD. The current transactional FP32 optimizer
holds weights, gradients, old/new moment maps and candidate parameters. Its known
steady-step tensor categories total **seven weight-sized arrays**; persistent
weights/gradients/moments alone total four. Activations, the recomputation gradient
accumulator, allocator/workspace and host/checkpoint costs are additional.

If these known transactional categories exceed observed free VRAM minus the
reserve, the report records `insufficient_budget` and `executed: false`. For the
current ~600M model on 12 GB, this is expected: seven FP32 arrays total about
15.65 GiB. The sweep does not deliberately allocate an optimizer already known
to exceed the device budget, change precision, lower the reserve or use the second
GPU. That result means AdamW remains unqualified and needs a memory strategy.

When the categories fit, a separate two-update AdamW process executes the smallest
capacity fixture. Two updates exercise retained moments and their candidate
copies. It has the same reserve/finite-metric guards and discards weights. Passing
this minimal gate does not qualify AdamW for larger capacity cases or the pilot.

## Report and controls

`sweep-report.json` retains each workload's status, process exit code, phase markers,
finite loss/gradient metrics, timing and memory summaries. `*.memory.jsonl` contains
the raw sampled traces. The report records model/fixture/binary hashes, pilot
restart checks, held-out loss change and separate SGD/CUDA/AdamW qualification flags.
Phase markers describe host control flow; CUDA kernels may finish asynchronously.
`pretrained_training_ready` stays false because these tests have explicitly limited
workload/optimizer scope and do not qualify arbitrary full contexts or production jobs.

Exit codes are:

- `0`, `passed`: all selected executed gates passed with no declared limits.
- `2`, `completed_with_limits`: collection completed but capacity/admission limits
  or dependent unrun stages remain. The full-context bound and expected AdamW
  shortfall normally produce this result on the current workstation.
- `1`, `failed`: an unexpected numerical/process/checksum/restart/setup failure.

The output directory and archive must both be new. The sweep may take hours,
depending on kernel compilation, context size, checkpoint I/O and update cost.
Every process has a two-hour timeout by default. Controls include:

- `--pilot-steps N` (4..64, default 12 per comparison branch; twice that many updates overall).
- `--timeout-seconds N` (per process), `--learning-rate RATE` and `--reserve-mib N`.
- Repeated `--capacity-case capacity-w02-t001` selections for focused reruns.
- `--fixtures-dir DIR` for an explicitly generated compatible fixture collection.
- `--tesslot-bin PATH` for an existing explicitly supplied binary.

The explicit `--cpu` regression mode is restricted to artifacts below 128 MiB and
must use the matching tiny fixtures. It never qualifies CUDA or model-sized GPU
execution. It is used by CI to execute the actual full orchestration against the
tiny converted model, including held-out evaluation, restart and AdamW updates.

## Fixture provenance and verification

`fixtures/pretrained-sweep-24/` was compiled against the actual converted tokenizer
with upstream Trajectory Remix at `d2c8083536d56a33d876e2650ae855722d491566`.
Its fingerprint is `dd428951d25e3b94c907ea4a8e0de493ffd6c96045eace8a18cb56e1cb41cc78`.
The compiler validates scheduling, assigns slots, tokenizes content and writes
the native chunks. Authored sources, token/row counts and checksums are included.
The uploaded tokenizer and weights are not committed. `pretrained-sweep-tiny/`
contains the equivalent cases for the tiny converted fixture tokenizer.

To regenerate a new collection during development (requires the pinned compiler
and its existing Python dependencies):

```sh
python3 scripts/generate-training-sweep.py \
  --upstream /path/to/pinned/trajectory-remix \
  --tokenizer /path/to/converted/tokenizer.json \
  --output-dir work/generated-sweep-01
```

CPU regression tests compare non-recording multi-target evaluation against attached
replay and verify parameter/optimizer/cursor preservation; execute the actual Python
orchestrator and native CLI, reject validation leakage, compare restart weights and
state, enforce real child termination on a reserve breach, distinguish OOM/timeouts
from numerical failures, check AdamW accounting and verify logs-only archiving.
Actual GPU capacity and pilot outcomes must come from the workstation sweep.
