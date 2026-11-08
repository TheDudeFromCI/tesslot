# CUDA finite-check Boolean readback correction

**Current result:** [all nine corrected workstation CUDA gates pass](validation/milestone-2-cuda.md),
including model AdamW/checkpoint continuation. The diagnosis and original cloud-only
verification below are retained as implementation history.

The supplied `results.log` shows all three basic CUDA probes passing, including
the strict single/two-device SGD comparison and the native TF32 diagnostic. All
five Qwen cases then fail at the **first optimizer update**, reporting
`nonfinite gradient kidra.exchange.identities.weight`.

The location matters: the tests had already completed dense/cached forward
comparisons, downloaded and compared every parameter gradient with CPU, and
checked nonzero first-partition/shared-exchange gradients. The error is raised
by `gpu.train_remix_step`, not those initial gradient comparisons. The log does
not establish that AdamW updates or GPU checkpoint/resumption work yet.

## Reproduced cause

Burn's CubeCL backend supports Boolean storage as `Bool(U8)` or `Bool(U32)`, while
Flex CPU uses `Bool(Native)`. `TensorData::try_to_vec::<bool>()` performs **no dtype
conversion** and rejects both integer-backed Boolean representations. The old
finite guard did this after its on-device `is_finite().all()` reduction:

```rust
.try_to_vec::<bool>().is_ok_and(|flags| flags == [true])
```

For a CUDA result containing logical `true`, readback returns a dtype mismatch.
`is_ok_and` turns that error into `false`, and the optimizer reports a nonfinite
gradient. The shared identity matrix is simply the first parameter in the sorted
map. Every precision/device arrangement reaches the same faulty guard.

The CPU regression constructs the exact integer-backed Boolean TensorData and
reproduces the old conversion error, without simulating GPU arithmetic or needing
NVIDIA hardware. Both true and false flags are tested for Native/U8/U32 storage.
This is an integration bug in Tesslot's use of Burn's readback API.

## Correction

A shared `qwen::checks` helper explicitly decodes the logical Boolean with
`try_to_vec_as::<bool>()`, validates its Boolean dtype and scalar result, and returns
`Result<bool, String>`. All gradient, candidate-parameter and AdamW moment guards
use it. Readback errors propagate with parameter/moment context; they are never
silently classified as numerical overflow.

The full tensor is still reduced on its owning device and only the scalar flag
is copied to the host. There is no whole-model download, gradient sanitization,
new clipping policy, changed attention mask or relaxed tolerance. Genuine NaN,
positive infinity and negative infinity still reject an update. A regression also
checks that rejecting a bad gradient preserves model weights, optimizer moments,
and the completed-step count.

## Rerun

```sh
./scripts/test-cuda.sh
```

The script now includes a small opt-in GPU finite-guard test before the full
Qwen gates, checking ordinary values, `f32::MAX`, NaN and both infinities. This
isolates readback/validation failures from model math. The CPU-only regression is:

```sh
cargo test -p tesslot-tensor --lib finiteness --locked
```

The uploaded run is evidence of successful initial CUDA forward/gradient comparisons
in all five model modes, including the asymmetric two-device path. Corrected
optimizer updates and checkpoint continuation still need a workstation rerun.
No corrected GPU execution is claimed from the cloud CPU/source checks.

Cloud verification for this correction: 65 ordinary CPU tests passed (20 heavy
cases ignored by default); nine opt-in Qwen/Remix training, finite-difference and
checkpoint/resumption cases passed. Formatting, workspace Clippy and CUDA-feature
Clippy with `DOCS_RS=1` passed. The latter is a source/type check that skips native
LLVM setup, not GPU execution. Hardware discovery still explicitly skips CUDA on
this host.
