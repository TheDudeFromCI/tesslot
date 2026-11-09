# Tiny converted-tokenizer readiness sweep

Equivalent compiler-authored cases for the tiny converted fixture tokenizer.
Used by CPU CI to execute the actual sweep, including multi-target evaluation,
checkpoint continuation and AdamW updates. CPU results never qualify CUDA capacity.

Sources, checksums and exact tokenizer identity are included. See the
[one-sweep handoff](../../docs/local-regression-tests.md) and development generator.
