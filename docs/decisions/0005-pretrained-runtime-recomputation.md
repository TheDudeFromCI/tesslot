# ADR 0005: append-only controls, explicit runtime placement and exact replay

Status: implemented on CPU; CUDA backend selection remains provisional.

Base Qwen tokenizer IDs and padded embedding rows carry pretrained meaning. We
append dedicated Kidra controls **after config.vocab_size**, reserving padded
rows explicitly rather than overwriting them. The output preserves tied heads
and records complete shard hashes, normalized tokenizer identity and seeded new
parameters. Sparse tokenizers or existing dedicated spellings require a migration.
New exports must use the resulting tokenizer; older exports are never relabeled.

A runtime plan owns layers, not windows. FP32 masters remain on their owners;
shared exchange tensors move through differentiable device transfers. BF16/FP16
are GEMM compute modes. This isolates physical placement from persisted agent
state and artifact schemas, but does not establish PCIe efficiency, fit, recovery
or cross-device autograd correctness without real hardware tests.

Burn's operation checkpointing probe did not establish whole-block memory savings.
We add exact target-prefix replay as a deliberately expensive recomputation mode.
It preserves the globally weighted full-trajectory objective and all durable-prefix
gradients while releasing prediction branches between backwards. Gradient sums
are accumulated before one optimizer update; storage chunks do not detach graphs.
CPU parity and independent-process RSS measurements are executable. Full-block
checkpointing and GPU allocator measurements remain open memory work.

See [implementation, commands and qualification boundaries](../milestone-2-completion.md).
