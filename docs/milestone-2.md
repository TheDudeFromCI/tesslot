# Milestone 2 — numerical implementation progress

**Current status:** [completion increment](milestone-2-completion.md) adds complete
pretrained tokenizer conversion, sharded local loading, explicit runtime placement,
reduced-precision GEMMs, model-level CUDA gates and exact recomputation. CPU gates
are executable; NVIDIA hardware qualification remains pending. The sections below
are historical measurements from earlier increments.

Latest gate: [native Qwen/Remix CPU training](qwen-remix-training.md) now verifies
external-first weighted trajectories, attached projected KV and checkpoint/cursor
resumption. Production SFT jobs and NVIDIA capability gates remain outstanding.

Latest update: the [tiny Qwen experiment](tiny-qwen-experiment.md) provides
single-window FP32 CPU generation in the dashboard, full-parameter SGD and
verified checkpoint continuation. The [two-window echo experiment](two-window-echo.md)
adds full-rank GQA exchange and attached source-memory gradients. The
[varied-pattern increment](echo-generalization.md) verifies held-out copying and
exact AdamW/sampler resumption on CPU. [Base-model conversion](base-models.md) is
implemented using the synthetic fixture. The sections below retain the original
autograd gate's measurements; outstanding production gates remain on the
[roadmap](roadmap.md).

This is the first independently testable milestone-2 increment. It completes the
initial CPU autograd gate and begins config inspection. **Milestone 2 as a whole
is not complete.** The initial gate did not include Qwen imports or CUDA; these historical
limitations are superseded by the completion report linked above.

## Implemented

`tesslot-tensor` uses pinned Burn/Flex 0.22.1 and f64 tensors to implement
`kidra-reference/2`. It is independent of HTTP and the CLI; ordinary CLI linking
does not acquire Burn. The existing scalar model remains a separate numerical
implementation. Its new named-parameter snapshot/import methods validate names,
shapes and finite values before constructing a replacement model. These payloads
are test/interchange values, not a versioned production checkpoint format.

All current reference parameters are differentiable tensor leaves: token and
window embeddings, prediction seed, every local attention/FFN matrix, shared
global query/output and output vocabulary matrix. Fixed reference normalization
constants are still fixed; this is not a substitute for Qwen's trainable norms.
Memory reads remain causal at every layer. Same-tick D queries and memory states
are computed before commit. Closed/replaced numerical entries are evicted while
existing dependent graph nodes remain attached. No detach or optimizer action
occurs at a dataset chunk boundary. Callers use a fresh cache per model/sample.

Stable log-softmax and the effective-weight denominator reproduce the unshifted
upstream objective. A single backward returns named gradients, reporting nonfinite
values as errors. Unused parameters receive explicit zero gradients. Test-side
SGD updates all parameter matrices; there is no operational training job/optimizer
service, scheduler, checkpoint or resume claim.

## Executable results

The tiny trajectory is produced by the real pinned Remix compiler/exporter using
`scripts/generate-fixtures.py`. Checked-in `autograd.source.json` and
`autograd.jsonl` exercise external observations, concurrent D decisions, weighted
supervised B input, replacement, NOOP and EOS across multiple storage chunks.

* All 616 scalar parameter derivatives of a two-layer, width-two, vocabulary-128
  model are compared against central differences of the independent scalar oracle.
  Epsilon is 1e-6; tolerance is 2e-5 times (1 + absolute numerical derivative),
  allowing finite-difference roundoff and normalization sensitivity. This checks
  unused coordinates as well as parameters with nonzero gradients.
* Every predicted logit and weighted objective agrees within absolute 1e-10 on
  f64 CPU; the tensor matmul/reduction order differs from scalar iteration.
* Ordinary and operation-checkpointed outputs/gradients agree within 1e-10.
  This tests Burn's operation strategy on the actual reference architecture. It
  does **not** establish whole-block activation checkpointing or memory savings.
* Explicit chunk-by-chunk attached execution matches the unchunked objective's
  gradients within 1e-10. The excluded initial observation has a nonzero later-loss
  gradient, proving that its prefix is attached rather than detached.
* A full-parameter SGD step decreases the same tiny objective. Future/current-tick
  target perturbations cannot alter earlier tensor predictions.

Run `cargo test -p tesslot-tensor --locked` or the existing full workspace CI.
The engine remains CPU-reference only; no CUDA/precision capability is advertised.

## Qwen3 config inspection

`model::qwen::Config` parses and validates the actual 0.6B/1.7B HF configs captured
in `examples/qwen3-*.config.json`. Source URLs were rechecked on November 5, 2026:

* https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/config.json
* https://huggingface.co/Qwen/Qwen3-1.7B-Base/raw/main/config.json

The loader rejects unknown fields, unsupported architecture/activation,
attention bias/dropout, sliding-window or RoPE-scaling features, invalid dimensions,
GQA head ratios, controls, dtypes and overflowing projection widths. It does not
assume `hidden_size / heads` determines head dimension. Both candidates have
Q width 2048 and KV width 1024; 0.6B hidden width is only 1024. The base's identical
BOS/EOS IDs remain base metadata and cannot pass as distinct Kidra controls.
This config parser is deliberately separate from the tiny oracle's configuration;
acceptance does not imply that weights can be loaded or executed yet.

## Remaining gates and next increment

1. Actual tokenizer and SafeTensors metadata inspection/import, tensor name/shape
   mapping and reproducible conversion reports with tiny fixtures. No large model
   weights have been downloaded or authorized by this work.
2. Native Qwen-derived blocks, trainable norm weights, GQA/head dimensions, dedicated
   Kidra controls and a versioned production exchange adapter. The tiny reference
   still implements the original ProjectKidra exchange, not its low-rank Qwen fork.
3. Model-level CUDA execution, real single/two-device gradients/updates, BF16/FP16,
   heterogeneous cards, copy profiling and capacity failure handling. This host
   exposes no NVIDIA devices; these results cannot be supplied by CPU simulation.
4. Whole-block graph-preserving recomputation and measured peak-memory accounting;
   SafeTensors serialization and optimizer/RNG migration before resumable training.

Burn remains provisional under ADR 0002. Passing CPU autograd is evidence for
continuing the evaluation, not evidence that asymmetric multi-GPU training works.

## Subsequent precision / import increment

Normal tensor CPU execution has moved to FP32, with FP64 retained explicitly for
validation. Native tokenizer and SafeTensors inspection/import are now implemented.
See [the detailed update](milestone-2-import.md); the f64 measurements above
describe the original validation path, not the current CPU default.
