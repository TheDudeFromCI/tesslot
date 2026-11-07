# ADR 0003: varied-pattern corpus and explicit experimental AdamW

Status: accepted for the CPU experiment; production numerical backend remains
provisional under [ADR 0002](0002-backend-gates.md).

The single-pattern echo demonstration could memorize one sequence. Its held-out
accuracy did not establish copying. The next gate must use disjoint pattern seeds,
avoid training on evaluation data, retain source causality, and resume both sampling
and optimization state. The per-layer full-rank exchange architecture remains
`kidra-qwen-echo/1`; changing it to hardcode a copying strategy would invalidate
the experiment.

Use a small procedural corpus with versioned seed namespaces and reproducible
Fisher–Yates epoch shuffling. Keep development monitoring separate from explicit
final-test evaluation. Report teacher-forced and free-running metrics separately,
with source-attention ablation and exact per-pattern outputs. Serialize held-out
seed identities as strings to avoid JavaScript integer rounding.

On the same 600-pattern shuffled curriculum, SGD at learning rate 0.2 reached
5/192 development letters (2.60%) versus 192/192 with AdamW at 0.003. This compares
optimization recipes including learning rate, rather than isolating one factor. Introduce explicit FP32 AdamW
for a **separate** corpus experiment, keeping previous SGD defaults/formats intact.
Use constant learning rate 0.003, β₁=0.9, β₂=0.999, ε=1e-8, decoupled decay 0.01
on all trainable parameters, and clipping before the moment update. Tests compare
known optimizer values and require exact continuation of parameters and moments.

The small optimizer is written using the already selected Burn tensor operations;
it does not introduce a second numerical framework or a Python dependency. Dense
named FP32 moment maps are easy to inspect and serialize in SafeTensors, at the
cost of two extra parameter-sized buffers. This does not provide fused kernels,
reduced-precision optimizer state, distributed placement or memory-efficient large
model training. Production optimizer selection should be revisited with backend,
precision and multi-device gates; any changed numerical/checkpoint semantics must
receive an explicit version.

Save the sampler definition/cursor atomically with model weights, optimizer count,
hyperparameters and independently checksummed moment tensors. Reject unsupported
format combinations, missing/wrong-shaped/invalid moments and invalid cursors.
Advance sampling only after a complete successful update. Running jobs and reports
remain ephemeral; checkpoint restoration is an explicit operator action. No claim
of complete production SFT resumption or GPU portability is made.

The default 600-update run achieves exact copying on all 16 development and 16
test patterns. Source masking reduces test accuracy to 7/192, providing evidence
that the model uses source-window information. General Trajectory Remix integration
and actual NVIDIA verification remain required before claiming the next broader
training milestone complete.
