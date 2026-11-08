# ADR 0006: distinguish strict FP32 from TF32-permitted GEMM

Status: CPU and [workstation CUDA gates](../validation/milestone-2-cuda.md) pass.

The workstation's CUDA probe produced post-update loss 12.994399 versus CPU
12.969219. TF32 rounding reproduces this exact value. Pinned CubeK promotes F32
stage/register operands to TF32 for its accelerator path; Burn's high-level API
does not expose a strict strategy selector. Storage dtype was an insufficient
precision contract.

Do not enlarge the strict tolerance to accommodate this change. Keep native CPU
FP32, implement CUDA strict FP32 with a deliberately slow tiled multiply/reduce
reference, and introduce an explicit `tf32` policy for native accelerated GEMM.
The reference uses Burn's existing tensor operations and autograd, including
cross-device transfer, without unsafe/native kernels or a new differentiation
engine. BF16/FP16 paths remain as before.

This corrects the stated `f32` semantics at a cost: reference CUDA FP32 throughput
and graph memory are unsuitable for large production models. Efficient strict
FP32 needs backend strategy control and an independent hardware gate. Model
weights/masters stay FP32; CPU fallback never moves to FP64. The tiny-model
NVIDIA rerun now passes; production memory/performance suitability remains provisional. See [evidence and commands](../cuda-precision.md).
