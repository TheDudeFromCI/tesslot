# Global exchange alignment

Reference: ProjectKidra commit `5e78cfb765aed2b54e575c6bc85f9648007b0596`,
[model/exchange.py](https://github.com/TheDudeFromCI/ProjectKidra/blob/5e78cfb765aed2b54e575c6bc85f9648007b0596/project_kidra/model/exchange.py),
with `layer.py` and `model.py` for cache production and invocation.

The original Tesslot oracle independently projected global K/V from post-local
hidden states, with RMS normalization and global RoPE. That was a different
mechanism. Architecture `kidra-reference/2` replaces it with the single-head
specialization of the original ProjectKidra GlobalExchange:

```
Q = W_query LayerNorm(post_local_block_state)
K = concatenate(local_RoPE_K[layer, window] + window_identity[window])
V = concatenate(local_V[layer, window])
output = state + W_out softmax(Q K^T / sqrt(head_dim) + causal_mask) V
```

Global query/output parameters are shared across layers, as in AgentModel's single
`full_attn` module. Exchange follows every local block (the upstream default
`n_global_layers=1`). Both own-window and other-window visible keys participate.
There is no extra global K/V projection, query RoPE, value identity offset or
post-local-state K/V bank. The CPU oracle reconstructs local projected K/V from
immutable layer inputs rather than retaining an optimized projected KV cache.
The reconstruction uses precisely the local attention's projection parameters.

Trajectory Remix remains authoritative for visibility: only durable, live,
non-removed memory is readable, with external-first ordering and no same-tick
generated targets. The older unmasked Python call is not copied literally.
Initialization rows have no visible keys; empty exchange returns the residual
without adding synthetic placeholder memory.

This matches the exchange equation, not every aspect of the Python model. The
tiny reference uses one head, fixed identity LayerNorm affine parameters with
epsilon 1e-5 and zero exchange linear biases. It retains its existing seeded
parameters, dense local blocks and separate label-free prediction queries. No
trained Python checkpoint or complete multi-head/MoE model parity is claimed.

The repository also contains a distinct `qwen/exchange.py` implementation:
grouped-query cached K/V, low-rank query/output adapters, separate KV-width identity
embeddings, zero-initialized output up-projection and masks supplied for other
windows. This change targets the original `GlobalExchange`, not that distinct
Qwen adapter. Production Qwen conversion must explicitly choose/version its
adapter rather than silently equating the two.

Tests cover the exchange equation analytically, full/cached/recomputed parity,
cross-window influence, identity ablation, and multilayer future/same-tick
noninterference. They do not establish PyTorch checkpoint or GPU parity.

## Experimental Qwen execution

The [two-window echo experiment](two-window-echo.md), `kidra-qwen-echo/1`, now
executes the full-rank GQA extension from `kidra-qwen-tensors/1`. It reuses local
Qwen K/V, adds KV-width identities to K and uses trainable shared LayerNorm/Q/out.
Input identities and KV identities are separate when their widths differ. It
retains zero linear biases, label-free decision queries and explicit causal
masks. This is a versioned extension of the original exchange equation, not the
repository's distinct low-rank Qwen adapter or an exact PyTorch checkpoint port.
