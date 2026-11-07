# Echo generalization: varied patterns and held-out evaluation

This increment keeps the `kidra-qwen-echo/1` model architecture and its two windows,
12 lowercase letters and one-tick lag. It adds an independent corpus experiment
with full-parameter FP32 AdamW, deterministic training sampling and held-out
evaluation. It does not change the single-pattern SGD experiment or its checkpoints.

```sh
cargo run -- serve --tiny-experiment
```

Open **Echo across random patterns** in Inference, Training or Evaluation. The
default 600 updates use different training patterns. The backend automatically
reports eight development patterns at the end of a job. Separate buttons evaluate
all 16 development or final-test patterns. Per-pattern replay shows the source,
free-running echo and accuracy. Choose a training budget using development results
before inspecting the final test set. Reports identify their split, sample count,
target weight and model step. No unavailable measurements are synthesized.

## Data and sampling contract

`echo-seed-corpus/1` defines 1,024 training patterns, 16 development patterns and
16 test patterns by default. Each seed generates its 12 letters on demand using
the existing seeded source driver. Source decisions are not neural-network samples
and have no prediction loss. Every echo target has weight one; a report over S
patterns has effective target weight `12 × S`. Source/echo histories remain
separate; the existing same-tick/future masks apply in every transformer layer.

| Split | Pattern seeds | Optimizer sampling |
|---|---|---|
| Train | `0 + i`, i in 0..train_size | Fisher–Yates shuffle without replacement each epoch |
| Development | `2^62 + i`, i in 0..development_size | Never |
| Test | `2^63 + i`, i in 0..test_size | Never |

The shuffle uses seed 42 by default and a versioned SplitMix64 sequence. It can be
reconstructed from dataset configuration and the number of committed pattern
updates. Configuration bounds are 1–4,096 training patterns and 1–16 per held-out
split. The default corpus is tested for disjoint seeds **and pattern content**.
The sampler advances only after a complete successful optimizer update. Evaluation
does not change weights, moments, step or cursor. The same dataset definition
can be reapplied idempotently; a different definition is rejected rather than
silently changing a resumed objective. A SHA-256 of the complete dataset definition
identifies it in evaluation reports and the definition itself is in checkpoints.

Held-out seeds exceed JavaScript's safe integer range, so per-case seeds are
decimal **strings** in report JSON. No model files or generated dataset files
are downloaded or added to the repository.

## Optimizer and checkpoints

The corpus uses constant learning rate 0.003, AdamW β₁=0.9, β₂=0.999, ε=1e-8 and
decoupled weight decay 0.01 on all eligible parameters. Global L2 gradient clipping
at one precedes moment updates. Weights, gradients and first/second moments are
FP32. Every model tensor must have a finite gradient; candidate moments and weights
are checked before committing the update. SGD remains unchanged for the previous
ABC and single-pattern experiments. [ADR 0003](decisions/0003-echo-corpus.md)
records the optimizer comparison and the limits of this implementation.

Independent checkpoints live under `.tesslot/tiny-qwen/echo-corpus/`:

* `model.safetensors`: all 6,320 model parameters.
* `optimizer.safetensors`: named first/second moments with exact parameter shapes.
* `manifest.json`: `tiny-qwen-echo-corpus-adamw/1`, model config, step, learning
  rate, AdamW hyperparameters/update count, checksums, tokenizer, dataset and cursor.

Publication uses the existing synced temporary-directory/atomic-latest-pointer
mechanism. A fresh zero-step optimizer can be saved/restored as well. Restore
validates format/optimizer pairing, versions, bounds, tensor names/shapes/finite
values, nonnegative second moments, update count, sampler cursor and both file
checksums before replacing live state. It resumes the next pattern and moment
update, not merely the visible model output. Existing base/single-pattern formats
remain supported in their own experiments. Rust also permits an explicitly selected
SGD corpus using `tiny-qwen-echo-corpus-sgd/1`; the HTTP corpus namespace requires
AdamW to keep the experiment's meaning fixed.

Stop occurs between updates; the stopped model/cursor can then be saved. Browser
disconnects do not stop the backend worker. On server restart press Restore to
load saved state, then start another job to continue. Running jobs and evaluation
reports are not automatically resumed or persisted. This is still a bounded CPU
experimental worker, not production job scheduling or a general SFT checkpoint.

## Public API

All paths are under `/api/window-v1/`; the same localhost/Host/Origin protections
and `--tiny-experiment` switch apply. Checkpoint paths are chosen by the operator,
not supplied by HTTP clients.

| Method | Suffix | Request / behavior |
|---|---|---|
| GET | inference/experiments/tiny-qwen-echo-corpus/status | Actual availability, optimizer, sampling progress, metrics and last report |
| POST | inference/experiments/tiny-qwen-echo-corpus/generate | `{"seed":42}` for a diagnostic replay; does not train |
| POST | training/experiments/tiny-qwen-echo-corpus/jobs | `{"steps":600}`, 1–4,000; continues saved sampler, arbitrary seeds rejected |
| POST | training/experiments/tiny-qwen-echo-corpus/evaluate | `{"split":"test","samples":16}`; development/test only |
| POST | training/experiments/tiny-qwen-echo-corpus/stop | Stop between completed updates |
| POST | training/experiments/tiny-qwen-echo-corpus/checkpoint | Save model, optimizer and sampling state together |
| POST | training/experiments/tiny-qwen-echo-corpus/restore | Validate and restore latest corpus checkpoint |

One operation owns each experiment at a time; conflicts return 409. Metrics are
bounded to the last 1,000 updates. `echo-evaluation/1` reports teacher-forced loss
and accuracy, **free-running** token accuracy, exact-pattern match rate, and
source-masked loss/free-running accuracy. Source masking applies at every layer,
including earlier echo memory, preventing an indirect source path in the ablation.
Test evaluation is only triggered explicitly; training never automatically uses it.

## Measured results and verification

With the default initialization, shuffle and 600 AdamW updates (600 distinct
training patterns; no epoch repeats):

| Measurement | Result |
|---|---:|
| Initial development loss | 4.852991 |
| Initial development free-running accuracy | 0/192 |
| Final development loss / token accuracy / exact patterns | 0.021129 / 192/192 / 16/16 |
| Final test loss / token accuracy / exact patterns | 0.020397 / 192/192 / 16/16 |
| Source-masked test loss / free-running accuracy | 8.959054 / 7/192 (3.65%) |

A comparison using the same initialization, 600 shuffled training patterns and
16 development cases gave **5/192 (2.60%)** with SGD at learning rate 0.2
(development loss 3.327237), versus **192/192** with AdamW at 0.003. This compares
two optimization recipes, including their learning rates; the architecture,
sampling order and visibility rules are identical. SGD did not establish
generalization in that run. These CPU results demonstrate copying on the bounded
12-letter, two-window distribution; they do not establish arbitrary-length copying,
production trajectory SFT, mixed precision, CUDA or multi-GPU correctness.

Opt in to all computational gates:

```sh
./scripts/test-tiny-qwen.sh
# Standalone measured training/evaluation, optional step count (default 600):
cargo run -p tesslot-tensor --example echo_generalization -- 600
# Same corpus/order with SGD, development-only comparison:
cargo run -p tesslot-tensor --example echo_generalization -- 600 sgd
```

The new ignored tests train across patterns, require ≥90% held-out token accuracy,
verify source ablation, validate effective weights and exact seed identities,
compare model/moment/manifest bytes after the next resumed update, reject corrupt
moments and invalid dataset/cursor versions, and exercise zero-step and epoch-boundary
resume. The HTTP workflow verifies development/test separation, request bounds,
restart recovery, safe stopping and sampler progress. Cheap ordinary tests check
split/shuffle invariants and AdamW against known two-step values and decoupled decay.
The separately opt-in Playwright workflow includes 600 corpus updates, 16-case test
evaluation and checkpoint restoration through the actual dashboard.

Next: integrate Qwen local blocks and exchange with general Trajectory Remix
lifecycles/NOOP/weighted objectives, preserving attached chunk gradients and
training/inference consistency. Production SFT/job scheduling and NVIDIA gates
remain separate milestones.

Final verification: 49 ordinary CPU tests pass with the CI feature set; all ten
computational experiments are ignored by default and passed when explicitly run.
Formatting, Clippy with warnings denied, workspace build and the actual Chromium
workflow pass. No CUDA hardware execution was performed.
