# Milestone 2 increment — FP32 CPU execution and native artifact import

This increment continues the CPU path without waiting for workstation CUDA results.
The hardware gates remain unverified: proceeding with other work does not turn a
skipped CUDA check into a passing hardware result.

## CPU precision

`Model::new` now stores parameters, activations, attention intermediates and
backward graphs in **f32**. `Model::inference` uses f32 without autograd recording;
normal CLI `demo` / `replay` use this inference path. Model initialization still
uses the reproducible scalar parameter generator and releases its temporary
high-precision payload after tensor construction. Prediction previews and weighted
loss may be converted to f64 for diagnostic output; no dense f64 forward is run
by normal replay and the CLI does not repeat the forward merely to check itself.

The independent scalar oracle remains f64, as does the explicitly named
`Model::validation_f64` diagnostic constructor used by exhaustive central-difference
checks. This provides a stronger reference for evaluating FP32 and future BF16.
Reference parameter interchange / gradient diagnostic payloads still use f64;
they are not runtime tensor storage or an implemented production optimizer state.
CPU FP32 does not assert that BF16 execution or GPU numerics have been verified.

The precision regression checks actual logits' dtype, non-recording inference,
nonfinite FP32 conversion rejection, f32/f64 logits and gradients, chunk parity
and causal noninterference. FP32-versus-f64 tolerance is
`2e-4 * (1 + abs(f64_logit))`; gradient tolerance is
`2e-3 * (1 + abs(f64_gradient))` on the tiny normalization-sensitive test model.
The f64 derivative check retains its existing 1e-6 perturbation and tolerances.
No performance or peak-memory improvement is claimed from an unmeasured benchmark.

## Native tokenizer loading

`tesslot-artifacts::NativeTokenizer` uses Rust tokenizers 0.23.1 with a pure-Rust
regex backend, with HTTP support and optional native tokenizer acceleration
features disabled. It reads supplied local bytes, disables padding/truncation,
validates dedicated BOS/EOS/NOOP encodings and distinct IDs, derives vocabulary
size and hashes the normalized tokenizer serialization. The actual Remix fixture's
identity is checked, rather than substituting a hash of the original JSON bytes.
Tokenizer input is bounded to 64 MiB. This loader does not modify/extend Qwen's
base tokenizer automatically; native Kidra controls must be deliberately added by
an architecture converter before training compatibility is established.

```sh
cargo run -- inspect-tokenizer fixtures/demo-tokenizer.json
```

## SafeTensors inspection and import

The new native library uses the upstream SafeTensors 0.8.0 codec. Header inspection
reads at most 16 MiB and uses the codec's metadata validation for tensor dimensions,
contiguous offsets and byte counts. Duplicate top-level tensor names, invalid
shapes, truncation, trailing bytes and overflow are rejected. A seekable input's
length is checked without reading the payload. A test represents a one-GiB tensor
with only a small header and proves no payload read/allocation occurs.

```sh
cargo run -- inspect-weights /path/to/model.safetensors
cargo run -- inspect-weights /path/to/model.safetensors --mapping mapping.json
```

No command downloads weights. Inspection reports actual tensor names, shapes,
dtypes, offsets and a **header** SHA-256, not a misleading whole-file checksum.
It does not validate payload float values until those payloads are read.

An explicit mapping names each destination's expected shape and source name, or
marks it for separate initialization. Preflight reports all compatible, missing,
unexpected and incompatible entries; invalid plans produce a nonzero CLI exit.
There are no heuristic Qwen tensor-name guesses or implicit discarded weights.
For example, using arbitrary illustrative names rather than claiming HF mappings:

```json
{
  "expected": {"model.weight": [2, 2], "window.embedding": [4, 2]},
  "sources": {"model.weight": "source.weight"},
  "initialize": ["window.embedding"]
}
```

The library's `import` function applies valid mappings, copies actual source
values and requires explicit initialization values for all new tensors. It
supports F32/F16/BF16/F64 source floats, decoded to finite f32, with per-source and
aggregate output byte budgets. Missing/extra initialization values and nonfinite
or out-of-FP32-range values fail. It retains source identity in the mapping report;
multiple destinations may explicitly reference a source, but a future model
converter must implement actual runtime parameter tying rather than treating the
copied arrays as an alias-preserving tensor graph.

Tests use tiny upstream-codec-generated artifacts for round trips, BF16/FP16
conversion, invalid headers, byte budgets and explicit reuse/new initialization.
No model-sized weights, network services or Python runtime are needed.

## Remaining production work

This is an artifact foundation, not a complete Qwen-to-Kidra converter. Outstanding
work includes real-checkpoint-specific tensor manifests, sharded index traversal,
tokenizer vocabulary extension, tied parameter handling in the model, reproducible
architecture-specific initialization metadata, atomic checkpoint serialization,
trainable Qwen norms/GQA blocks and end-to-end pretrained execution. CUDA/BF16 and
multi-device tests remain separate optional hardware work.

Dev/test profiles disable debug symbols to fit dependency-heavy builds in small
workspaces. Override `CARGO_PROFILE_DEV_DEBUG=1` or `CARGO_PROFILE_TEST_DEBUG=1`
when debugging. This changes build artifact size, not runtime numerical semantics.

## Verification for this increment

The CPU workspace and final focused suites pass (40 tests total), including the
actual CLI tokenizer/weight commands and invalid-mapping exit status. Formatting,
Clippy with warnings denied, documentation links and whitespace checks pass. The
73-tick/four-window FP32 demo completes with loss about 5.37206312. No CUDA result
is asserted, and no large model weights were downloaded.
