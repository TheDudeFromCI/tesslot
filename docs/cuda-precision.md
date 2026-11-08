# CUDA FP32 storage and TF32 arithmetic

The workstation failure reported on 2026-11-07 was in the post-update probe loss:
CPU `12.969219`, CUDA `12.994399`, for both single- and two-device execution. The
probe compared initial loss and both parameter gradients before that final value;
those comparisons had passed. This excerpt provides useful initial CUDA evidence,
but does not qualify the full model or the corrected tests.

The numerical difference is reproduced exactly by TF32 operand rounding. For the
updated hidden vector `[2.04, 5.96]` and output weights `[1.016, -0.952]`:

| Arithmetic | Output | Squared output |
|---|---:|---:|
| FP32 | -3.6012802 | 12.969219 |
| TF32-rounded operands, FP32 accumulation | -3.6047745 | 12.994399 |

This is a precision-policy error in the original integration, rather than evidence
of a broken gradient transfer in this example. FP32 storage alone did not guarantee
full-precision matrix products. The independent CPU regression preserves the
original tolerance and explicitly rejects the TF32-rounded loss as strict FP32.

## Pinned backend evidence

In CubeK 0.3.1, `matmul::strategy::auto` tries `SimpleCyclicCmma` before a SIMT
fallback. `multi_level::definition::blueprint::adjust_dtypes` replaces F32
stage/register operands with TF32 when the accelerator supports it, retaining
F32 global storage. CubeCL LLVM 0.11.3 lowers that rounding to
`llvm.nvvm.f2tf32.rna` / `cvt.rna.tf32.f32`. Burn CubeCL 0.22.1 calls the default
strategy from `float_matmul`. Its high-level `Tensor::matmul` and `DeviceConfig`
do not expose a strict-FP32 strategy selector. These findings come from the exact
locked registry sources, not from assuming cuBLAS or PyTorch configuration applies.

## Corrected policies

`qwen::runtime::Plan.precision` now distinguishes:

| Value | CPU | CUDA |
|---|---|---|
| `f32` | Existing native FP32 GEMM | Strict FP32 reference multiply/reduce |
| `tf32` | Rejected | Backend-native FP32-storage GEMM; permits TF32 acceleration and FP32 SIMT fallback |
| `bf16` | Rejected | Existing BF16 GEMMs, FP32 masters/state |
| `f16` | Rejected | Existing FP16 GEMMs, FP32 masters/state |

The CUDA `f32` implementation composes safe differentiable tensor slices,
broadcast products, and reductions. Each product tile holds at most 16³ values;
it never calls GEMM in forward or backward, so CubeCL cannot substitute TF32
operands there. Ordinary FP32 reduction-order differences remain possible.
It introduces no unsafe code, native kernels, FP64 computation, host offloading
or detached gradients. CPU FP32 and BF16/FP16 retain their existing paths.

**Strict CUDA FP32 is a slow numerical reference, not a production GEMM.** Its
extra kernel launches and retained autograd nodes can be substantial on a real
model. Use it for tiny qualification tests; target production BF16 remains the
preferred direction. `tf32` explicitly opts into native accelerated arithmetic
when FP32 storage with reduced operand precision is acceptable. Production strict
FP32 requires a backend extension/upgrade exposing an appropriate kernel strategy
and separate hardware verification; it is not claimed by the native GEMM path.
Existing runtime plans specifying `f32` now receive the strict arithmetic they
requested, so CUDA performance and continuation numerics can change from the
previous implicit TF32 behavior. Shared cross-device semantics are unchanged.

## Verification and rerunning

```sh
cargo test -p tesslot-backend-probe --features burn-cpu --test cuda --locked
cargo test -p tesslot-tensor --test matmul --locked
./scripts/test-cuda.sh
```

The first two run without NVIDIA hardware. They check the original analytic
forward/gradient/SGD example, reproduce the exact TF32 failure, and compare the
reference forward and both operand gradients against independent scalar arithmetic
for a transposed rectangular matrix crossing every tile boundary. Invalid shapes
and dtypes are rejected.

The hardware script retains the strict probe's `1e-4 * (1 + |CPU|)` tolerance and
reports loss, both gradients, both updated parameter arrays, the updated hidden
vector, output and loss by stage. The Qwen strict-FP32 gate also retains its
existing `4e-4` scaled tolerance. An independent native-GEMM diagnostic recognizes
the two known FP32/TF32 dot products at tight tolerance; accepting TF32 there does
not relax either strict gate. The explicitly named TF32 model gate uses `0.01`
scaled tolerance for its reduced operand precision; BF16/FP16 tolerances remain
unchanged. GPU cases run serially to keep stage logs clear.

CPU tests and CUDA-feature source checks are possible in the cloud. Actual corrected
single/two-device execution still requires a workstation rerun. Burn remains
provisional pending those model-level results.

Verification for this correction: workspace CPU tests 61 passed (20 heavy cases
ignored by default); seven opt-in Qwen/Remix tests passed; formatting, workspace
Clippy and CUDA-feature source Clippy with `DOCS_RS=1` passed. That source check
skips native LLVM setup and is not a linked GPU execution result. Hardware discovery
on this cloud host still reports an explicit skip. No corrected NVIDIA pass is claimed.

## Subsequent workstation evidence

The [follow-up log](cuda-finite-checks.md) passes the corrected strict probes and
initial forward/gradient comparisons in all model modes. A separate Boolean
readback bug blocks the first optimizer update and is fixed in that increment.
