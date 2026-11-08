# Implementation roadmap

Milestone 1 delivers a runnable reference baseline. It does not make production
training or GPU claims. [Testing](testing.md) and the completion report record
executed checks and unsupported areas.

| Milestone | Dependencies | Executable completion criteria |
|---|---|---|
| 1: foundation | Pinned Remix contract | Workspace/CLI/config, multi-window CPU oracle, strict native validator, canonical fixtures, causality/equivalence/loss tests, operational status API, live dashboard, versioned schemas/ADRs |
| 2a: numerical gates | 1 | Port tiny oracle to an autograd backend; all trainable parameters have finite gradients; finite differences and CPU equivalence; explicit mask and attached-chunk gradient tests; checkpoint/recompute parity |
| 2b: Qwen3 metadata/import | 2a for execution; metadata work can start from 1 | Actual HF config/tokenizer/SafeTensors metadata parsing; tiny shape fixtures; explicit conversion report, tied weights, native controls and seeded new components; no silent discarded tensors |
| 2c: initial CUDA | 2a/2b | Single-device forward and backward, FP32/BF16/FP16 comparisons with declared tolerances, training/incremental/cache equality; hardware tests opt in; real dual-device capability gate before production backend commitment |
| 3a: full-parameter SFT | 2 | Whole-trajectory weighted objective and attached graph; SGD/AdamW, schedules, accumulation/clipping, finite-gradient checks, deterministic split/shuffle and evaluation on tiny exports |
| 3b: checkpoint/job lifecycle | 3a | Atomic checksummed checkpoint + optimizer/RNG/schedule state; interruption/restart and changed-device resume; persistent jobs, measured metrics, safe stop/pause |
| 3c: 0.6B feasibility | 3a/3b; often 4 memory mechanisms | Authorized optional weights; preflight measured/estimated memory report; real full-parameter experiment with validation, documented OOM/precision limitations |
| 4: asymmetric multi-GPU | 2 gates, 3 tiny trainer | Explicit layer map, resource admission, per-device budgets, correct forward/backward/optimizer parity, activation recomputation, transfer/memory measurements; 1.7B feasibility without pooling VRAM |
| 5: production inference | 2, resource interface; 4 for multi-GPU | Native sampling, incremental projected KV, replacement invalidation, persistent sessions, cancellation/model management, SSE/WS reconnect and backpressure tests, inference workspace |
| 6: agent runtime | 5, SQLite/events/auth | Real scoped tools/registry, async scheduling/deadlines/cancellation, persistent projects/tasks/artifacts, external specialist adapters, audit/authorization tests and agent workspace |
| 7: evaluation/optimization | 3/5/6 | Exact and behavioral metrics, regression reports, benchmark reproducibility, measured GPU bottlenecks, optimized attention parity and evaluation workspace |

## Immediate milestone-2 priorities

1. Port the tiny model to Burn/Flex without changing the versioned query/memory
   semantics. Compare every row's logits, all parameter gradients and one update
   with finite differences. Keep the f64 oracle as a separate implementation.
2. Run two real NVIDIA GPUs through `to_device` forward/backward and optimizer
   parity, including heterogeneous generations and constrained asymmetric budgets.
   Profile actual transfer/staging. This gate may precede full GPU scheduling.
3. Prove graph-preserving recomputation and attached storage-chunk gradients on
   the real model. Escalate a framework limitation before committing the architecture.
4. Implement config/tokenizer/SafeTensors metadata validation and tiny conversion
   fixtures before any optional pretrained download. Inspect actual names/shapes.
5. Implement single-device Qwen-derived local blocks, global attention and
   conversion initialization, then CUDA numerical tolerance/equivalence tests.

Maintain a backend decision log with results, versions, devices and limitations.
Do not declare production selection based solely on the CPU probes. Full SFT at
0.6B with Adam-family state may require milestone-4 memory techniques; the schedule
does not promise it fits one workstation GPU before measuring.

Future work: approximately 4B feasibility, RL rollout/reward storage and policy
updates, client libraries (Python deferred), alternative backends/platforms and
distributed cluster execution. Keep decision/observation/outcome records versioned
so future RL does not require redesigning the agent protocol.

## Current milestone-2 increment

The first CPU autograd gate and native Qwen3 config inspection are implemented;
see [results and explicitly outstanding gates](milestone-2.md). This does not
mark Qwen import, full checkpoint/recomputation support or CUDA complete.

## Current milestone-3 increment

The tiny CPU 3a training path is implemented independently of the open Qwen/CUDA
gates: attached trajectory objectives, full-parameter SGD/AdamW, schedules,
accumulation/clipping, deterministic data ordering and evaluation. See
[Milestone 3 results and remaining work](milestone-3.md). Final weight export
remains distinct from the new epoch-boundary 3b checkpoints and CLI resume.
Persistent jobs, arbitrary optimizer-boundary recovery and placement migration
remain open. The initial [3c preflight](milestone-3c.md) now reports per-device
parameter/state estimates and source-header compatibility; the real 0.6B
experiment remains blocked on pretrained execution and hardware gates.
