# ADR 0002 — defer production backend selection until executable gates pass

Status: provisional. Burn is the recommended candidate, not the engine backend.

The user requires correct asymmetric layer partitioning across RTX 3060/5060
devices, full-parameter SFT, explicit masking and attached cross-chunk gradients.
Multiple device constructors alone do not establish any of these properties.

Use the opt-in `tesslot-backend-probe` crate to test pinned Burn 0.22.1 on CPU.
Use `burn-tensor` directly, with Flex/autodiff and default features disabled, to
avoid coupling the core to a large training umbrella. The deprecated NdArray path
is not the recommended CPU backend for this release. Burn's newer dynamic Device
API must not be confused with older backend-generic Tensor examples.

Before a production commitment, run and publish:

1. A tiny causal multi-window network with gradients for every original and added
   parameter; finite differences on f64 CPU and comparisons with the oracle.
2. Two real CUDA devices: forward 0→1 and backward 1→0, all parameter gradients
   and one optimizer step compared to a single-device baseline. Include asymmetric
   budgets, unavailable ordinals, both GPU generations, explicit transfers and
   measured communication. No silent CPU fallback.
3. Checkpoint/recomputation equivalence on the same network, including B insertions,
   parallel D decisions, closures, replacements and attached chunks. Measure peak
   allocated/reserved memory. Balanced operation recomputation is not evidence of
   arbitrary transformer-block activation checkpointing.
4. BF16/FP16 matmul, normalization, masking and finite gradients on real hardware;
   optimizer masters/state precision, clipping and serialization/resume checks.
5. SafeTensors/shape conversion, deterministic new-parameter initialization, tied
   embeddings, optimizer state movement and resume under a changed device layout.

If Burn cannot pass cross-device/recomputation gates, prototype LibTorch through
Rust `tch` with a separately installed C++ LibTorch runtime. That need not require
Python at runtime. Verify the same gates, rather than assuming a mature upstream
engine eliminates integration risk. Candle is attractive for native Qwen loading
but needs comparable training/recomputation evidence. Native CUDA is permitted
only for a demonstrated missing operation or measured bottleneck with backward
and numerical tests; do not design a new autodiff framework.
