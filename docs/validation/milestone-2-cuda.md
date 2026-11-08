# Milestone 2 CUDA qualification — passing workstation run

The maintainer-supplied `results.log` reports **nine CUDA tests passed, zero failed,
zero ignored**, with no SKIP branches. This closes the tiny-model hardware gate
for Milestone 2. The run was executed on the maintainer's NVIDIA workstation,
not in the CPU-only cloud development environment.

[Retained log](milestone-2-cuda.log) replaces the local checkout path with
`<checkout>`; all test results and numerical values are preserved.

- Original uploaded log SHA-256:
  `70508210808179c12a6c6cda802aba6120f78cecc82d675fb7e58570719b6fd9`.
- Retained redacted log SHA-256:
  `660c82b07c264f7b03563e715aa5da97ec53acabd5dbfa655eef0e888c52628c`.
- Harness: the gates present after PR #10, including the Boolean readback correction.
  The upload does not include its checkout commit SHA, GPU identities, driver or
  CUDA runtime versions; these details remain unrecorded. Target workstation
  specifications are not substituted for measured device information.

## Results and scope

| Suite | Passed | Failed | Harness runtime |
|---|---:|---:|---:|
| Single/two-device strict FP32 SGD probes and native TF32 diagnostic | 3 | 0 | 1.42 s |
| CUDA finite-value guard, including NaN and both infinities | 1 | 0 | 0.89 s |
| Qwen-derived model execution/training/checkpoint gates | 5 | 0 | 120.89 s |

These runtimes are whole-suite measurements, not training throughput or per-device
kernel benchmarks. The basic probes both produce post-update loss `12.969219`;
the separate native GEMM diagnostic produces the expected TF32 value `-3.6047745`.
The finite guard accepts finite values and rejects NaN and positive/negative
infinity on actual CUDA tensors.

All five model cases report `full/cached/all-gradients/AdamW/checkpoint` passing:

| Model arithmetic | Device placement | Result |
|---|---|---|
| Strict FP32 reference | Single CUDA device | Pass |
| TF32-permitted native GEMM | Single CUDA device | Pass |
| BF16 GEMMs, FP32 masters/state | Single CUDA device | Pass |
| FP16 GEMMs, FP32 masters/state | Single CUDA device | Pass |
| Strict FP32 reference | Two CUDA devices, one layer on ordinal 0 and two on ordinal 1 | Pass |

The harness uses a three-layer tiny Qwen-derived model with configurable GQA and
shared full-rank cross-window exchange, replaying the canonical Remix autograd
fixture. It compares dense/cached logits, every named parameter gradient with CPU,
and nonzero first-partition/shared-exchange gradients. It then performs two
full-trajectory AdamW updates, writes a checkpoint, restores through CPU, moves
masters and optimizer state to the original CUDA plan, and compares continued
logits with the uninterrupted run and CPU baseline at the declared tolerances.

The test explicitly exercises backward propagation across devices and shared
parameter use across layer owners. It does not establish that a different layer
assignment resumes identically: the restored model returns to the original plan.
The [precision policies](../cuda-precision.md) and [finite-check correction](../cuda-finite-checks.md)
remain part of the qualified implementation; no tolerance was enlarged to hide
the previously reported defects.

## What remains outside this gate

- Actual Qwen 0.6B/1.7B pretrained runs or conversational quality.
- BF16/FP16/TF32 model training across two GPUs; the dual-device model case is strict FP32.
- GPU memory savings from recomputation or whole-block checkpointing.
- Asymmetric VRAM admission, placement heuristics, OOM recovery or offload.
- Transfer/PCIe efficiency, peer-to-peer versus host staging, or useful throughput.
- Production strict FP32 GEMM; the current CUDA FP32 path is a slow reference.
- Production training job scheduling, accumulation, evaluation and full resume
  lifecycle with schedules/RNG beyond the implemented bounded experiments.

Burn 0.22.1 has passed the numerical capability gates needed to continue Milestone 3
SFT development. Production memory/performance suitability remains provisional.
The next implementation work is gradient accumulation and trajectory-level
validation/evaluation, followed by persisted training jobs, schedules and safe
checkpoint/pause/resume lifecycle. Large-model feasibility follows explicit
operator-supplied artifacts and memory measurements; full asymmetric scheduling
and communication optimization remain Milestone 4.
