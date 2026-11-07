# CPU fallback and optional NVIDIA tests

Run the portable suite followed by any available hardware gates:

```sh
./scripts/test.sh
```

CPU tests always run first. When NVIDIA discovery is unavailable, the script
prints **SKIP CUDA tests** and exits successfully without compiling CUDA features
or attempting CUDA initialization. Ordinary `cargo test --workspace` remains CPU
only. CI explicitly enables the CPU probe feature rather than `--all-features`.

On an NVIDIA workstation, run only the hardware gates with:

```sh
./scripts/test-cuda.sh
```

The optional `tesslot-backend-probe/cuda` feature enables Burn/CubeCL CUDA. This
requires a working driver and the compiler/runtime dependencies of that backend
(including NVRTC and applicable LLVM tooling); `nvidia-smi` alone cannot establish
kernel/compiler readiness. A discovered GPU followed by a build, kernel, gradient
or numerical failure is a real failure. It is never swallowed as a skip.

Two ignored-by-default integration tests compare FP32 forward loss, gradients of
both parameter partitions and the next forward after one full SGD update against
Flex CPU. The dual-device test transfers activations from device 0 to 1 and
requires backward propagation to the first partition. Tolerance is absolute plus
relative `1e-4 * (1 + abs(cpu))`, permitting FP32 CUDA reduction differences while
rejecting missing gradients. Tests respect the visible-device count; fewer than
two visible GPUs produce an explicit skip message for the dual-device test.

To invoke Rust's hardware harness directly after installing the CUDA toolchain:

```sh
cargo test -p tesslot-backend-probe --features cuda --test cuda --locked -- --ignored --nocapture
```

Rust's test harness represents the default state as `ignored`. During an explicit
hardware run, a missing-device branch prints SKIP and returns; the harness counts
that return as a success, **not as evidence of CUDA execution**. Capture the output
when reporting results. `CUDA_VISIBLE_DEVICES=""` or `-1` forces CPU-only execution
in the script. `nvidia-smi` is the conservative discovery preflight; environments
without it skip even if another application could access CUDA.

These small optional gates are not the full Kidra CUDA backend. They do not yet
verify model-level mixed precision, asymmetric memory admission, checkpoint
migration, GPU memory accounting or PCIe performance. Production CUDA placement
remains unavailable until those integration gates are implemented. CPU-only
execution is the default, while explicit production CUDA placement continues to
fail clearly rather than silently changing the requested device plan.

In the current cloud environment, the skip paths and all 32 CPU tests passed.
An optional CUDA `cargo check` was attempted, but stopped in
`tracel-llvm-bundler 23.1.0-4`: its default installation directory under the user
data folder is read-only here. Thus neither CUDA-feature compilation nor GPU
execution is certified. This native dependency failure does not affect the
CPU suite or the no-hardware skip path.

## Model-level Milestone 2 gates

`test-cuda.sh` now also runs the three-layer Qwen/Remix FP32/BF16/FP16 and
asymmetric two-device gates, including all gradients, AdamW and changed-device
checkpoint restore. See [exact scope and tolerances](milestone-2-completion.md).
They require actual hardware; a skipped test supplies no GPU evidence.

## CUDA precision correction

See [the TF32 diagnosis and corrected precision policies](cuda-precision.md).
The strict FP32 probes keep their original tolerances and now use an explicit
FP32 multiply/reduce reference. The script also reports native FP32/TF32 behavior
and tests the explicitly selected TF32 model path. CUDA FP32 reference execution
is intended for tiny correctness gates, not large-model throughput.
