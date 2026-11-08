# Model definitions and versioned artifacts

Current Milestone 2 supports native SafeTensors imports, complete pretrained
conversion and checksummed experimental optimizer checkpoints; see the
[artifact/runtime contracts](milestone-2-completion.md). The initial design below
records Milestone 1 and is superseded where those implementations now exist.

Milestone 1 supported only `kidra-reference/2` configuration and seeded tiny
parameters in memory. No SafeTensors import/conversion/checkpoint command existed at that milestone.
`cargo run -- config` prints the reference configuration. Unknown config fields,
invalid dimensions, unavailable CUDA placement and non-loopback binds are errors.

Version axes remain separate: Tesslot application `0.1.0`, API `window-v1`, Kidra
architecture `kidra-reference/2`, export `kidra-token-chunk/1`, tokenizer SHA-256,
future checkpoint version, and future production tool-schema version. A token
fingerprint/control mismatch fails validation even if vocabulary sizes coincide.

## Actual Qwen3 configuration inspection

Inspected the real [0.6B Base config](https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/config.json)
and [1.7B Base config](https://huggingface.co/Qwen/Qwen3-1.7B-Base/raw/main/config.json)
on 2026-11-05, without downloading weights:

| Field | Qwen3-0.6B-Base | Qwen3-1.7B-Base |
|---|---:|---:|
| hidden_size | 1024 | 2048 |
| intermediate_size | 3072 | 6144 |
| num_hidden_layers | 28 | 28 |
| num_attention_heads | 16 | 16 |
| num_key_value_heads | 8 | 8 |
| head_dim | 128 | 128 |
| vocab_size | 151936 | 151936 |
| max_position_embeddings | 32768 | 32768 |
| rope_theta | 1000000 | 1000000 |
| rms_norm_eps | 0.000001 | 0.000001 |
| tie_word_embeddings | true | true |
| bos_token_id / eos_token_id | 151643 / 151643 | 151643 / 151643 |
| torch_dtype | bfloat16 | bfloat16 |

Both use SiLU, no attention bias, no sliding-window attention, and null rope
scaling in these configurations. Do **not** derive head dimension by dividing
hidden size by head count: for 0.6B, the Q projection is wider than hidden size.
The inspected Candle Qwen3 implementation explicitly models Q/K per-head RMS
normalization and grouped-query heads. Exact weight names/shapes must still be
verified against the model's actual SafeTensors index/header in milestone 2.

## Conversion plan

Parse architecture metadata with supported-feature validation, load tokenizer and
SafeTensors metadata before allocating tensors, then construct an explicit report
of each expected source/destination tensor and validated shape. Reuse compatible
embedding, attention, FFN and normalization weights; account for tied output
embeddings explicitly. Do not invent source tensor names from examples.

Add dedicated Kidra control vocabulary/embeddings rather than reusing the base
model's identical BOS/EOS ID. Validate tokenizer extension and embedding/output
shapes together. Initialize window identities, global attention and prediction
query parameters reproducibly with recorded seed, policy and architecture version.
Missing, unexpected and incompatible weights are errors or explicit reviewed
conversion dispositions; never silently discard them. Preserve source revision,
hashes, mapping and initialization metadata in the converted artifact manifest.

Ordinary CI uses tiny generated SafeTensors fixtures. Optional multi-gigabyte
weights require explicit operator authorization and are never downloaded by tests.

## Synthetic Qwen fixture and tensor conversion

The [user-supplied base model policy](base-models.md) excludes real weights from
this repository. A tiny Qwen3-shaped fixture now tests actual loading and
value-preserving tensor conversion, with explicit initialization and provenance.
The older no-import statement above describes the milestone-1 baseline. Tiny
Qwen execution and [native Remix training](qwen-remix-training.md) are now
implemented. Pretrained tokenizer/control extension and real-size execution
remain outstanding. The Remix demo initializes a new synthetic vocabulary aligned
with the actual export tokenizer; it never reinterprets ASCII embedding rows.
