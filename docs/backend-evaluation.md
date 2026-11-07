# Numerical backend evaluation

Evaluated for milestone 1 on 2026-11-05. No NVIDIA device or `nvidia-smi` is
available in this environment. GPU conclusions below are explicitly unverified.
The oracle remains backend-independent; this document recommends an investigation,
not a completed backend integration.

| Candidate | Useful capabilities and evidence | Constraints and unresolved gates |
|---|---|---|
| Burn 0.22.1 / Flex CPU, CUDA via CubeCL later | Native Rust autodiff, explicit masks, tensor placement; source implements backward device copy to original device. CPU probes exercise f64 matmul, both parameter gradients, same-device copy, attached prefix gradients, explicit allowed-mask conversion and operation checkpointing. | No real multi-GPU, CUDA, BF16/FP16 or optimizer-state migration test performed. Balanced checkpointing recomputes selected operations, not necessarily whole transformer blocks. Backward consumes shared intermediates; combine losses for one attached-sample backward. New dynamic Device API differs from older backend-generic examples. |
| Candle (source inspected at `cfcbd7186d13271843ce7eb41d25659bd0835850`) | Native Rust CPU/CUDA, SafeTensors and Qwen3 implementation; differentiable `Op::ToDevice` copies gradients back. Concrete Qwen3 `head_dim`, Q/K normalization and separate Q/K/V projections are present. | No executable Candle test performed. No general transformer activation-checkpoint API found in inspected core source. Cross-device correctness/performance, full SFT optimizer/checkpoint behavior and needed custom backward ops remain unverified. |
| Rust tch / C++ LibTorch | Mature underlying tensor/autograd ecosystem; potential alternative if Burn fails capability gates. C++ LibTorch can be used without embedding Python. | Not installed or executable-tested here. C++ ABI/CUDA deployment, checkpoint/recompute access through Rust bindings and heterogeneous GPU transfers require independent verification. No claims based on multiple device constructors. |
| Native CUDA interoperability | Could implement a specialized masked attention or transfer primitive if measurements justify it. | Must supply a correct backward integration, tests and clear ownership/stream lifetimes. Not a substitute for an entire tensor/autograd engine. Deferred. |

## Recommendation

Continue the Burn evaluation with the exact lockfile and opt-in probe. Use Flex
for CPU experiments: Burn 0.22 deprecates NdArray. Depend only on `burn-tensor`
with the required features; the experiment is not linked into normal CLI/server
builds. Burn's Rust-native deployment and custom-operation path fit the intended
platform. Do not finalize the production backend before
[ADR 0002's gates](decisions/0002-backend-gates.md) pass.

CPU experiments cannot establish heterogeneous CUDA compatibility, asynchronous
copy/stream behavior, direct peer-to-peer transfers, the absence of host staging,
efficient graph partitioning or real memory savings. In particular, a same-device
CPU `to_device` test provides no multi-GPU evidence.

Burn's operation-checkpointing probe is deliberately small. The real architecture
must preserve identical lifetimes/masks during recomputation and retain gradients
across storage chunks. If whole-block checkpointing cannot be expressed safely,
document the limitation and evaluate LibTorch before implementing full training.

## Inspected primary sources

* [Burn 0.22.1 autodiff documentation](https://docs.rs/burn-autodiff/0.22.1/burn_autodiff/)
  and registry source `ops/tensor.rs::float_to_device`,
  `checkpoint/strategy.rs`, and `burn-tensor::tensor::api::autodiff`.
* [Burn tensor 0.22.1](https://docs.rs/burn-tensor/0.22.1/burn_tensor/), including
  Device autodiff/checkpoint configuration and explicit dtype options.
* [Candle device-copy backward](https://github.com/huggingface/candle/blob/cfcbd7186d13271843ce7eb41d25659bd0835850/candle-core/src/backprop.rs)
  and [Qwen3 model](https://github.com/huggingface/candle/blob/cfcbd7186d13271843ce7eb41d25659bd0835850/candle-transformers/src/models/qwen3.rs).
* [tch upstream](https://github.com/LaurentMazare/tch-rs) as the alternative to
  investigate; not an executable result in this milestone.

The workspace declares Rust 1.95 or newer; only the installed 1.99.1 toolchain
has been tested in this environment. Burn registry sources record commit
`907ab111a02c2dea47bc41633754f095c6b2220e`.

## Milestone 2 completion increment

[Current implementation and verification](milestone-2-completion.md) supersede the
initial CPU-only probe status above. Burn/Flex is now linked into the reference
CLI/server path; CUDA remains feature-gated. Exact target-prefix recomputation
passes CPU loss/gradient/update/resume parity and has independent-process RSS
measurements. This does not supply whole-block GPU allocator measurements.

Registry `burn-tensor` 0.22.1 `tensor/api/base.rs::to_device` explicitly states that
distributed backward requires parameters to use the loss backend. Mixed CPU/CUDA
placement is rejected; two CUDA device ordinals remain supported by the plan and
await actual model-level verification. Backend transfers may stage through host
memory. `to_device` preserves autodiff association even when the destination device
has autodiff disabled; inference snapshots therefore use `Tensor::without_autodiff`.
Production selection is still conditional on the real NVIDIA results.
