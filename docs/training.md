# Training design and milestone boundary

Milestone 1 had no trainer. The current [Qwen/Remix CPU gate](qwen-remix-training.md)
performs actual full-parameter AdamW updates with weighted trajectory loss and
attached projected KV across export chunks. The opt-in echo/corpus experiments
remain separate. Production jobs, large pretrained training and GPU execution
are still later gates.

Full-parameter SFT is mandatory. Every compatible pretrained parameter and all
new window/global parameters must be eligible for optimization. LoRA is not a
replacement. The model owns differentiable execution; the trainer owns schedules,
optimizer updates and normalization; datasets own ordered replay contracts.

## Full trajectory objective

Read a validated complete sample, then execute its tick groups in storage-chunk
order. Compute B targets before insertion and D targets from a shared after-B
snapshot. Accumulate weighted loss numerators and effective weights through the
entire attached graph. Normalize by `max(total_weight,1)` once, backward once,
then update once. Never step or detach at a chunk boundary. If gradient
accumulation spans samples, aggregate numerators/denominators consistently rather
than naively averaging differently weighted sample/chunk means.

Future truncated backpropagation requires a distinct mode and documented objective.
Training/inference equality must hold for equivalent causal context at every
architecture revision, with cache invalidation for closure and replacement.

## Safe job lifecycle (planned)

Jobs are backend-owned and survive browser disconnection. States:
queued → running → pausing → paused → running, or stopping → stopped,
completed, failed. Pause/stop acknowledge only a completed optimizer boundary.
Interrupted mid-update state is not resumable. Before admission, calculate
per-device weights, gradients, optimizer/master states, saved/recomputed
activations, temporary buffers, checkpoint overhead and margin. Explicit offload
configuration is required; PCIe memory is never treated as pooled VRAM.

Milestone 3 starts with seeded SGD/AdamW, learning-rate schedule and warmup,
gradient accumulation/clipping, finite-gradient checks, validation, true metrics
and atomic checkpoints. Mixed precision must verify FP32 accumulation/master
weights as appropriate, finite updates and dtype-specific tolerances.

## Production checkpoint protocol (planned)

Bounded experimental [Qwen/Remix checkpoints](qwen-remix-training.md) already
preserve model/optimizer/tokenizer/data state and exact continuation. The broader
production protocol below still requires scheduler/RNG/device-placement work.

Version checkpoints independently. Write model SafeTensors, optimizer states,
schedule/step, deterministic RNG states, architecture config, tokenizer identity,
dataset fingerprints/export version, training config and placement provenance into
a temporary same-filesystem directory. Flush payloads and a checksum manifest,
atomically publish a completed directory, then atomically update the latest-valid
pointer and fsync the parent. Validate all hashes/shapes/versions on load; ignore
unpublished partial directories. Keep the prior checkpoint until the new manifest
passes validation. Do not deserialize untrusted executable pickle.

Logical tensor names must not encode original device ordinals. Reallocate on a
new supported layer map when resuming; reject impossible capacity/dtype changes
clearly. Backend optimizer/RNG limitations must be recorded, never concealed.

Evaluation will separate exact replay diagnostics from behavioral outcomes:
token accuracy/weighted loss, control/lifecycle, NOOP, slots, tools, visibility,
tick efficiency and task completion. Valid alternative coordination strategies
must not fail simply because their token sequence differs from a reference.
Persist decisions/observations/outcomes in versioned rollout records suitable for
future reward attachment, without selecting or implementing an RL algorithm now.
