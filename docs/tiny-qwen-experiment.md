# Opt-in tiny Qwen generation, training and checkpoints

The 5,184-parameter random Qwen3 fixture now has an executable **single-window
FP32 CPU** runtime: trainable RMS normalization, Q/K per-head normalization,
explicit head dimensions, grouped-query attention, causal masking, split-half HF
RoPE, SwiGLU and the tied vocabulary head. All source parameters participate in
SGD. This experiments with the base Qwen architecture; it does not yet integrate
production lifecycle or Trajectory Remix training. A separate [two-window echo
experiment](two-window-echo.md) now adds full-rank Kidra exchange and source/echo
replay using the same opt-in switch.

## Browser experiment

```sh
cargo run -- serve --tiny-experiment
```

Open http://127.0.0.1:8080. The “Tiny Qwen experiment” panel can:

* Generate up to 32 actual greedy tokens (the button requests 16).
* Train 1–200 steps on repeating `a b c ` using next-token cross entropy.
* Display real loss, gradient norm and completed step measurements.
* Stop training after a completed SGD update, save a checkpoint, and restore it.

The experiment intentionally uses `ascii/1`: byte IDs 0–127 rather than a production
Qwen tokenizer. Prompt length is limited to 64 ASCII bytes. Nonprinting predicted
tokens appear as `\xNN`; raw IDs are also shown. Generation stops at the fixture's
EOS. Output is expected to be gibberish. Responses are shown after the bounded
request completes; this is not a production SSE/session streaming implementation.

Without the flag, controls remain disabled and experiment mutations return 503.
Production inference/training APIs remain unimplemented. All experiment operations
use the public `window-v1` API documented in OpenAPI. Loopback/Host/Origin checks
apply; HTTP clients cannot choose arbitrary checkpoint paths or model files.

Training runs in a backend worker and continues if the browser disconnects.
One generation/training/checkpoint operation owns the experiment at a time; other
requests receive 409. Training metrics remain bounded to the last 1,000 entries.
This is a small experimental worker, not the future centralized job/resource manager.

## Checkpoint behavior

Checkpoints are stored under `.tesslot/tiny-qwen/` relative to the server working
directory (ignored by Git). Each new checkpoint gets its own directory containing
SafeTensors weights and a checksummed `tiny-qwen-sgd/1` manifest with source config,
ASCII tokenizer identity, optimizer identifier, constant learning rate and step.
The optimizer is deterministic SGD without momentum, with norm clipping at one;
there are no hidden Adam states or random sampling states to restore.

Weights and manifest are written into a sibling temporary directory and synced,
then published into a newly reserved destination. Existing checkpoint directories
are never overwritten. The latest-pointer file is replaced only after publication.
A restarted server starts from the random fixture and detects the saved pointer;
press Restore to load the saved state. An interrupted training job itself is not
restored. Checkpoint loading validates versions, bounds, config, shapes and the
whole weight-file hash. This is not the full production training-checkpoint format.

## Opt-in experiments

```sh
./scripts/test-tiny-qwen.sh
```

These tests are `#[ignore]` by default, so ordinary unit tests and CI skip them.
No GPU, real model download or Python is needed. The script explicitly runs:

1. Causal prefix noninterference for the executable Qwen network.
2. Sixty full-parameter SGD updates on the repeating text, with finite nonzero
   gradient checks and a required loss reduction below 65% of its initial value.
3. Actual token generation, checkpoint save/restore, identical logits-derived loss
   and generated IDs after restoration, the next matching SGD update, refusal to
   overwrite an existing checkpoint, and rejection of a corrupted weight file.
4. The public HTTP generation/training/status/save/restore workflow.

An additional opt-in Playwright check, `node scripts/tiny-qwen-browser.cjs`, runs
against an experiment server on port 8081 (configure `bind` in the server JSON).
It requires a separately installed Playwright/Chromium; set
`TESSLOT_CHROMIUM_PATH` if using system Chromium. This check passed with actual
browser generation, four training updates, and checkpoint save/restore. The HTTP
experiment also verifies checkpoint restoration after recreating the server state.

A measured CPU run reduced loss **4.815503 → 1.2649204** over 60 steps. This is
an actual smoke-test measurement, not a claim of conversational ability or reliable
sequence mastery. The test asserts the learning objective improves rather than
requiring a particular sampled sentence. CUDA and the production multi-window
runtime remain separate gates before proceeding to their implementation phase.

Fixture serialization also canonicalizes metadata key ordering: upstream
SafeTensors accepts a hash map whose iteration order varies. The committed
fixture was regenerated with the same seed and identical tensor values, while
metadata bytes now reproduce deterministically.

Verification for this increment: 46 CPU tests passed with the CI feature set;
the base experiments were skipped by default and passed when explicitly run.
The combined script now includes the two-window experiment and its HTTP/gradient
gates; see its linked report for measured results.
Workspace build, formatting and Clippy with warnings denied passed. No CUDA
hardware execution was performed.
