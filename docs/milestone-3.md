# Milestone 3 — tiny CPU full-parameter SFT

This first 3a increment trains `kidra-reference/2` on validated native exports.
It builds on the existing CPU autograd gates. Milestone 2 tiny-model Qwen/CUDA gates now pass on main; this increment does not establish model-sized pretrained training.
The first 3b checkpoint increment is also available below. Persistent jobs and
the real 3c training experiment remain unimplemented. The
[3c preflight](milestone-3c.md) now provides analytical memory accounting.

## Run

```sh
cargo run --locked -- train fixtures/autograd.jsonl \
  --tokenizer fixtures/tokenizer-identity.json \
  --training-config examples/train-tiny.json \
  --output tiny-trained.safetensors
```

The optional global `--config` selects the reference model, including its
initialization seed. The training config selects optimizer/data controls; unknown
keys and invalid hyperparameters fail. Omitting `--training-config` uses one
epoch, AdamW at 0.001, norm clipping at 1, and a 90%/10% train/validation split.
The example assigns its single trajectory to training, runs four epochs and
warms up over two optimizer steps. Its validation metric is explicitly `null`.
Do not interpret training loss on this fixture as a generalization measurement.

The command validates the complete export before any update, then reopens it for
each pass. Keep the export unchanged during a run. Splits use the existing seeded
UUID hash, and each epoch uses a bounded deterministic whole-trajectory buffer
shuffle with `seed + epoch_index` (wrapping u64). Test data is validated but never
used for updates or evaluation. An empty training partition fails clearly.
Validation uses fresh, non-recording models/caches and never changes the trainer.

## Objective and optimizer

Every sample executes its complete ticks in storage-chunk order using one
attached cache. B queries precede insertion; parallel D queries share the after-B
snapshot. There is no detach, backward or optimizer step at a chunk boundary.
Every named reference parameter is eligible for updates, including token/window
embeddings, query, local blocks, output and global exchange.

For each sample, backward runs once on the unnormalized weighted loss numerator.
Across `accumulation_samples`, host f64 sums those numerator gradients, then divides
once by `max(sum(effective_weight), 1)`. Independent samples use fresh graphs and
caches. This avoids retaining several sample graphs and avoids averaging unequal
sample means. Each epoch flushes an incomplete final optimizer batch. Batches
with no target weight fail without advancing optimizer state.

Global L2 clipping follows normalization and precedes optimizer updates. SGD is
plain SGD. AdamW uses bias-corrected first/second moments and decoupled weight
decay on all leaves, including embeddings. JSON optimizer names are `sgd` and
`adam_w`; AdamW requires explicit beta1, beta2, epsilon and weight_decay if supplied.

Runtime parameters/activations and Adam moments are FP32. The current backend
interchange uses host f64 matrices whose parameter values are rounded to FP32 at
initialization and after every update. Gradient accumulation, norm calculation,
and Adam bias-correction scalars use host f64; optimizer moment recurrence and
committed parameter updates use FP32. This is a tiny CPU implementation, not a
scalable device-resident optimizer or mixed-precision training path.

Rates use one-based optimizer steps. With warmup N, step k <= N uses
`peak_rate * k/N`. Without warmup, step 1 uses peak. Optional `decay_steps: D`
linearly reaches zero D updates after the peak step and stays zero thereafter.
Without decay, the rate remains at peak. Schedule position advances only after
a successful optimizer update.

Nonfinite loss, logits, gradients, moments and updates fail explicitly. Candidate
parameter/moment maps are committed together only after every value passes;
an error preserves the previous parameters, moments and step number. This
in-memory guarantee is not crash recovery or an atomic checkpoint protocol.

## Metrics and saved weights

Stdout contains JSONL evaluation, optimizer-step and completion records. Metrics
include measured weighted cross-entropy, effective weight and weighted argmax
token accuracy (first maximum wins ties). Evaluation runs before training and
after every epoch. Step metrics describe that batch before its update. Completion
includes elapsed wall time, split counts, model/training config, tokenizer identity
and export format. Save stdout if you need the run report.

`--output` writes all final logical parameter names/shapes to a new FP32
SafeTensors file; existing files fail before training and are never overwritten.
This file contains final weights only. It lacks optimizer/RNG/schedule state,
checksum manifest, dataset fingerprint and atomic directory publication. It is
not resumable, and a process interruption while saving may leave a partial file.
Use the separate checkpoint directory described below for resumable state.

## Verification and observed result

The numerical suite checks an independent batch SGD update with unequal sample
weights, fractional total weights below one, global clipping, multi-step AdamW
against an independent f64 recurrence, deterministic repeated runs, loss reduction,
chunk/recomputation parity, prefix gradients and transaction rollback. Config,
schedule, empty/oversized trajectory and nonfinite failure paths are exercised.

The actual CLI test constructs multiple tiny trajectories to exercise separate
train/validation/test partitions, multi-sample accumulation and epoch tail flushes.
It verifies decreasing train loss, repeated identical SafeTensors bytes, logical
parameter coverage, decoded weight reload/evaluation, refusal to overwrite and
late-invalid-export preflight failure.
These copied fixture trajectories test mechanics, not held-out model quality.

The CPU workspace suite passes all 58 tests with the optional Burn CPU probe
enabled. Formatting, Clippy with warnings denied, Markdown links and whitespace
checks pass, as do the existing demo and canonical-export validation commands.

The command above completed four AdamW updates with default model seed 42 and
reduced measured training loss from approximately **4.849997** to **4.650741**.
No large weights were downloaded and no CUDA or BF16 result is claimed.

## Next increments

3b still needs persistent jobs, pause/resume/stop controls at arbitrary completed
optimizer boundaries, and supported placement migration. The HTTP/dashboard
training service remains unavailable. Epoch-boundary checkpoints below complete
only the initial recovery increment.

Before 3c, finish the pretrained architecture/import and hardware gates and add
measured/estimated per-device memory admission. The default 512-row CPU admission
bound and bounded shuffle limit data/graph size, but are not byte-accurate memory
budgets. Host parameter copies and candidate optimizer maps add memory overhead.
No 0.6B feasibility, GPU optimizer, placement migration, mixed precision or OOM
recovery has been implemented or measured here.


## Epoch-boundary checkpoint and resume increment

Add `--checkpoint-dir checkpoints` to `train` to save after each fully completed
and evaluated epoch. This is independent of `--output`, which remains final
weights only. For a clean resumable stop, add `--stop-after-epochs 1`; if training
is unfinished, it emits `stopped` and leaves the final weight output uncreated.
Restart with the same model/training/tokenizer/export arguments, the same
`--checkpoint-dir`, and `--resume` (omit `--stop-after-epochs` to finish).
`--stop-after-epochs` counts additional epochs in the current invocation.

For example, using the same fixture and training config as the first command:

```sh
cargo run --locked -- train fixtures/autograd.jsonl \
  --tokenizer fixtures/tokenizer-identity.json \
  --training-config examples/train-tiny.json \
  --output tiny-resumed.safetensors --checkpoint-dir tiny-checkpoints \
  --stop-after-epochs 1
cargo run --locked -- train fixtures/autograd.jsonl \
  --tokenizer fixtures/tokenizer-identity.json \
  --training-config examples/train-tiny.json \
  --output tiny-resumed.safetensors --checkpoint-dir tiny-checkpoints --resume
```

`tesslot-checkpoint/1` stores FP32 model SafeTensors and JSON optimizer moments,
step/schedule config, completed epoch, architecture/export versions, tokenizer
identity, exact export SHA-256, model config and CPU placement provenance.
There is no stochastic model or optimizer RNG in this architecture. The seeded
buffer shuffle is reconstructed from the recorded base seed and next epoch index;
no partial accumulation or shuffle buffer exists at the saved boundary.

Payload files and a SHA-256 manifest are flushed inside a temporary directory on
the same filesystem. The complete candidate is loaded and verified, its directory
is atomically published and synced, then a synced temporary `latest` pointer is
atomically replaced and the root synced. Prior checkpoint directories remain.
The loader follows only `latest`; partial directories and unpointed candidates
are ignored. Invalid hashes, versions, tensor/moment names/shapes/nonfinite values,
unsupported RNG/placement or mismatched input/config identities fail explicitly.
A corrupt latest checkpoint fails rather than silently selecting older state.
The hashes detect corruption; they are not signatures authenticating a source.

The implementation targets local Linux filesystems with one writer per root and
a 64 MiB bound per checkpoint file. Keep the export immutable during a run; it is
fingerprinted before training and checked again before each epoch evaluation/save.
Resume requires identical configs, including the original total epoch count.
CPU state is reallocated on load; no GPU/changed-placement resume is supported.

A crash during an epoch loses that epoch's unsaved updates. Restart replays from
the last complete epoch, never from partially updated state. Process-kill and
clean-stop CLI tests both require resumed final SafeTensors bytes to equal an
uninterrupted run. Numerical tests cover SGD and AdamW moment/schedule restoration,
corrupt payloads, correctly checksummed malformed state, failed publication,
unsafe pointer paths and ignored interrupted/unpointed directories. This does not
yet provide persistent job ownership, mid-epoch checkpoints or browser controls.

## Integration with current main

Main now provides bounded pretrained conversion and Qwen/Remix execution plus
[passing tiny-model CUDA qualification](validation/milestone-2-cuda.md). Use the
[local pretrained workflows](milestone-2-completion.md) for those experiments.
The preparation script here exercises the separate tiny reference trainer and
collects evidence; model-sized 0.6B/0.8B training readiness remains unverified.
