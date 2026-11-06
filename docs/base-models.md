# User-supplied base models and synthetic conversion tests

Real Qwen model weights are **not bundled with Tesslot**. Users must supply base
model config, tokenizer and weight files separately, preferably outside the source
checkout. There is no automatic weight download. Conversation with a real model
will require those separately supplied artifacts when that runtime is implemented.
Never include model-sized downloads in ordinary CI. `.gitignore` excludes
SafeTensors files except the tiny synthetic fixture, plus local models/checkpoints
folders. Ignore rules prevent accidental additions; they do not restrict paths
that the trusted operator may intentionally pass to the CLI.

## Tiny test model

[fixtures/tiny-qwen](../fixtures/tiny-qwen/README.md) contains a real HF-shaped
`model.safetensors` file with 5,184 locally generated parameters and its matching
config. Its projection names and shapes follow Transformers Qwen3 v4.51.3, including
GQA, per-head Q/K norms and the explicit head-dimension exception. The source file
has 13 tensors; the tied output embedding is represented logically, not duplicated.
This random model is a mechanical import fixture, not a useful conversational base.

## Actual tensor conversion

```sh
cargo run -- convert-qwen \
  --model-config fixtures/tiny-qwen/config.json \
  --weights fixtures/tiny-qwen/model.safetensors \
  --output /tmp/kidra-demo-converted --windows 4 --seed 42
```

The output directory must not exist. The command validates every expected HF
source tensor, preserves the source values, adds seven separately initialized
Kidra tensors and writes `model.safetensors`, the source `config.json`, and
`conversion.json`. The manifest includes full source-file SHA-256, seed, source
configuration, slot capacity, reused/initialized names and tied output aliases.
Missing, unexpected, wrong-shaped, unsupported and nonfinite tensors are errors.
New components do not silently overwrite pretrained tensors.

The added tensor contract `kidra-qwen-tensors/1` contains input window identities,
a label-free query seed, shared full-rank exchange query/output matrices, exchange
normalization scale/bias and KV-width window identities. Query and output widths
come from Qwen's explicit head dimensions, while distinct KV identities accommodate
GQA. This is a **tensor-conversion contract**, not a claim that the current tiny
single-head reference can execute Qwen's GQA network. It retains original Qwen
weights/norms rather than flattening them into that reference architecture. No LoRA
or quantized adapter is substituted for these parameters.

Source parameters remain FP32 in this fixture; import also supports float payloads
already handled by the native reader. By default conversion budgets 128 MiB for
FP32 output and each source read; `--max-bytes` can explicitly raise that bound for
user-supplied files. Large source parameters fail preflight before allocation.
This bound is not a complete peak-memory or GPU feasibility estimate.

## Verified and remaining

CPU tests verify source schema/byte reproducibility, exact loaded values,
value-preserving conversion, logical tied/untied output handling, deterministic
new initialization, seed changes affecting only new components, output
serialization/reload, budgets, malformed shapes and source completeness. CLI tests
exercise real files and refuse to replace an existing destination.

Tokenizer vocabulary/control extension, loading tied aliases into an executable
parameter graph, Qwen numerical forward/backward and conversation are still pending.
The manifest explicitly records `tokenizer_conversion_required: true`; base
BOS/EOS metadata must not be mistaken for Kidra's distinct BOS/EOS/NOOP controls.
Sharded pretrained checkpoints and resumable training checkpoints remain separate
work. No GPU verification is implied by successful fixture conversion.
