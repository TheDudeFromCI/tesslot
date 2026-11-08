# Milestone 3c — 0.6B feasibility preflight

The allocation-free preflight is implemented. **The real 0.6B full-parameter
training experiment has not run, and Milestone 3c is not complete.** Pretrained
Qwen execution/Kidra conversion, model-level CUDA/mixed-precision verification,
large-model checkpoint support and device resume remain prerequisites. The
current managed environment exposes neither `nvidia-smi` nor `/dev/nvidia*`.
No model weights were downloaded and no model-sized allocations were attempted.

## Command

```sh
cargo run --locked -- preflight-training examples/qwen3-0.6b.config.json \
  --scenario examples/preflight-0.6b.json
```

This illustrative FP32 AdamW scenario has a single **8 GiB** budget and a **1 GiB**
reserve. It is not a description of the user's hardware. The command intentionally
returns a nonzero exit code: persistent state alone exceeds that device's budget.
It prints a JSON report before returning the failure. An invalid scenario fails
before reporting, while valid scenarios report insufficiency per device.

The supplied Qwen3-0.6B configuration implies **596,049,920** unique base
parameters, including tied embedding/output weights counted once. FP32 AdamW
weights, gradients and two FP32 moments total **9,536,798,720 bytes**, about
**8.88184 GiB**, before activations, buffers, staging, reserve or Kidra additions.
That is analytical accounting from config shapes, not a measured OOM or peak.

Use an edited scenario with explicit device labels/budgets and layer ranges to
plan other cases. Each layer must have exactly one owner; gaps, overlaps and
unknown devices fail. Embeddings and the output projection must share an owner
when tied. For untied weights, the output projection and final norm belong to the
output device. Each device's budget is checked independently; spare memory on
another device cannot satisfy it. This is accounting, not a multi-GPU scheduler.

Optional `--weights /local/model.safetensors` inspects one local SafeTensors
header and reports compatible, missing, incompatible and unexpected source tensors
against the explicit Qwen3 manifest. An invalid mapping produces a nonzero exit.
No source tensors are silently discarded. The command does not read the payload,
prove its finite values, convert controls or execute pretrained weights. The
report always declares `weights_payload_verified: false` and
`hardware_measured: false`. Sharded metadata inspection is now available separately through
`inspect-model-directory`; see [local preparation](local-training-preparation.md).
Sharded tensor import/conversion is still unimplemented.

## Parameter and memory accounting

The manifest uses HF Qwen3 embedding, final norm, per-layer RMS norms, Q/K head
norms, GQA Q/K/V/O projections and SwiGLU up/gate/down tensors. Query projection
width is `num_attention_heads * head_dim`, not assumed equal to hidden size;
KV width is `num_key_value_heads * head_dim`. A tied `lm_head` is an alias and
is counted only as the embedding tensor. An artifact containing a second copy
must be reconciled by conversion; it is reported as unexpected here.

The proposed Kidra production adapter is not finalized, so no global/window/query
parameters are guessed. `additional_parameters` is an explicit per-device scalar
count supplied by the scenario. Zero means a base-Qwen-only planning assumption,
not a claim that conversion adds no parameters.

| Scenario | Weights | Gradients | Master weights | AdamW moments | Persistent bytes/parameter |
|---|---:|---:|---:|---:|---:|
| FP32 SGD | 4 | 4 | 0 | 0 | 8 |
| FP32 AdamW | 4 | 4 | 0 | 8 | 16 |
| BF16/FP16 SGD | 2 | 4 | 4 | 0 | 10 |
| BF16/FP16 AdamW | 2 | 4 | 4 | 8 | 18 |

These are explicit future device-resident backend assumptions. They do not
estimate the existing tiny CPU trainer's peak memory: host f64 interchange,
FP32-valued host parameter maps, gradient readback, candidate optimizer copies
and current checkpoint encoding have additional costs. In this planning scheme,
reduced precision still retains FP32 gradients and masters. Other optimizer
schemes require different verified accounting rather than assuming BF16 saves
all categories. No BF16/FP16 training or loss scaling is implemented here.

Activation, temporary/transfer and device checkpoint-staging allowances are
separate optional byte values. Null means unknown, not zero. Supply values only
for a declared workload/implementation; do not infer them from the parameter count.
The report includes:

- `persistent_bytes`: weights + gradients + master + moments.
- `known_required_bytes`: persistent bytes + all supplied allowances + reserve.
- `estimated_required_bytes`: the full scenario total, null if any allowance is unknown.
- `insufficient_budget`: known bytes exceed this device's budget.
- `incomplete_estimate`: known bytes fit but a workload allowance is missing.
- `estimate_within_budget_unverified`: all allowances were supplied and their sum fits.

Even a complete estimate within budget is not memory admission or evidence of a
working training run. `training_ready` remains false, with explicit execution
blockers. Checkpoint tensor-payload storage assumes FP32 saved model plus optimizer
moments and any separate master copy. It excludes JSON expansion, manifests,
filesystem overhead and publication/staging duplication. The tiny trainer's
64 MiB checkpoint-file limit cannot store a 0.6B model.

## Verification

Analytic tiny-shape tests independently count tied/untied tensors, precision and
optimizer components, additional parameters and overheads. Placement tests prove
budgets are not pooled. Unknown costs stay incomplete. Tests reject invalid
placements and checked-integer overflow before tensor allocation. A config-sized
0.6B test checks the exact count and persistent-memory shortfall without weights.
The actual CLI tests verify failure status/report JSON, successful incomplete
reports, and valid/invalid local tiny source headers. All 58 workspace CPU tests,
formatting and Clippy with warnings denied pass.

## Remaining experiment

Before a real experiment, identify target GPU models/VRAM, obtain authorized local
weights and a converted native tokenizer/export, complete pretrained model and
hardware gates, and lift the tiny-only checkpoint/optimizer constraints. Then
measure actual peak per-device allocations during forward, backward, update and
checkpoint save on a declared attached-trajectory workload. Record effective target
weight, train/validation loss, precision, finite gradients/updates, checkpoint
restart behavior, and actual OOM outcomes. No measured OOM, validation-quality,
GPU-memory or 0.6B throughput claim is made by this preflight.

## RTX 3060 / RTX 5060 planning scenarios

Two further examples target nominal RTX 3060 12 GB and RTX 5060 8 GB capacities.
Budgets assume 12/8 GiB before a 1 GiB reserve per device; replace these with
measured available capacities before admission. Both use base-Qwen-only FP32
AdamW accounting, zero declared Kidra additions, and unknown workload overheads.

- `examples/preflight-0.6b-3060.json`: all layers and tied embeddings/output on
  the 12 GB RTX 3060. This is the simplest hardware target for initial single-GPU
  correctness work, before introducing cross-device execution.
- `examples/preflight-0.6b-3060-5060.json`: layers 0–13 and tied embeddings/output
  on the RTX 3060, layers 14–27 on the RTX 5060. This is a prospective placement
  for later multi-GPU support, not an implemented runtime configuration.

Pass either file to `--scenario` in the preflight command above.

| Placement | Device | Persistent state | Headroom after 1 GiB reserve |
|---|---|---:|---:|
| Single GPU | RTX 3060 | 8.882 GiB | 2.118 GiB |
| Split 14/14 | RTX 3060 | 5.600 GiB | 5.400 GiB |
| Split 14/14 | RTX 5060 | 3.282 GiB | 3.718 GiB |

Headroom still needs activations, kernel/transfer buffers, checkpoint staging and
new Kidra parameters. Both reports remain `incomplete_estimate` with
`training_ready: false`. No GPU allocation, precision or PCIe throughput was
measured. Single-GPU feasibility is unproven; the split adds an unimplemented
scheduler and heterogeneous-device compatibility gates. Host RAM and SSD capacity
do not extend either device's VRAM budget.

## Provisional 0.8B target and training K/V overhead

The current experiment order prefers the smaller converted model on one GPU if
feasible, then both GPUs for the 1.7B model if the smaller model trains well.
The previous base-Qwen scenarios remain comparisons, not fit evidence for a
converted model. Use `examples/preflight-0.8b-provisional-3060.json` to test the
provisional smaller-model size on the RTX 3060. This adds 203,950,080 assumed
parameters to the config-derived 596,049,920, giving exactly 800,000,000. It is a
planning assumption, not a measured Kidra architecture or implemented conversion.
Actual growth depends on window count, global-attention dimensions and sharing,
and whether conversion retains tied embedding/output weights. Count the final
unique tensor manifest before treating 0.8B as the real model size.

FP32 AdamW persistent state at 0.8B is 12,800,000,000 bytes (11.921 GiB).
With the example's 1 GiB reserve it exceeds the nominal 12 GiB budget before any
activation allowance. The preflight intentionally exits nonzero. Smaller contexts
or activation recomputation cannot remove that persistent-state requirement.
A successful single-GPU plan may require optimizer-state offload, a verified
lower-memory optimizer/state representation, or another explicit memory strategy.
These must preserve full-parameter training; no such optimization is implemented
or selected by this scenario. BF16 alone does not guarantee fit under the current
FP32-gradient/master accounting assumptions.

For scale, an explicitly stored inference-style projected K/V bank has payload

`2 * layers * live_durable_tokens * kv_heads * head_dim * bytes_per_element`.

For the base config (28 layers, 8 KV heads, head dimension 128), 4,096 total live
durable tokens require 0.4375 GiB in BF16, or 0.875 GiB in FP32, for K/V values
alone. Token count is across all windows in one sample; sum across independent
samples in a batch. Reusing the same local projected K/V for global reads need
not allocate a second bank. These numbers exclude gradients, tensor/allocator
metadata, saved intermediates, attention workspaces and any changed production
adapter geometry. The current tiny implementation reconstructs projected K/V from
retained states, so the bank formula is not a measured allocation in that trainer.

Attached full-trajectory training retains differentiable prefix dependencies,
including across storage chunks. It cannot simply detach an inference cache.
Backward may retain hidden states, projections, MLP intermediates and attention
intermediates; a dense attention implementation can have quadratic context costs.
Visibility eviction alone does not prove that earlier graph storage is released.
Recomputation and memory-efficient attention can reduce saved activations, but
need numerical/gradient verification. Avoid double-counting K/V already included
in an activation estimate; supply a measured combined allowance once a real
training implementation/workload exists.

The larger model also needs a state-memory strategy: at exactly 1.7B parameters,
FP32 AdamW persistent state alone is about 25.332 GiB, exceeding the two devices'
combined nominal 20 GiB even before per-device placement constraints and overhead.
Conversion may increase that count further. Using two GPUs is the planned target,
not sufficient evidence of fit; optimizer offload or another verified memory
reduction will likely be necessary with this optimizer scheme.
