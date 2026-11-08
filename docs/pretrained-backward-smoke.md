# Converted Kidra backward-only smoke

After base-model forward and local CUDA qualification, measure an attached
cross-window backward on the converted Kidra weights before allocating AdamW.
This diagnostic does not update parameters, advance the sample cursor, allocate
an optimizer or save a model checkpoint.

## Workstation command

Pull the latest PR branch and point `--model-dir` at the **converted Kidra model**,
including its conversion.json and converted tokenizer.json. Select the RTX 3060's
UUID using `nvidia-smi --query-gpu=uuid,name --format=csv,noheader`.

```sh
python3 scripts/pretrained-forward-smoke.py \
  --model-dir "/path/to/Kidra-Qwen3-0.6B" \
  --backward-export fixtures/pretrained-backward-24/trajectory.jsonl \
  --gpu-uuid "GPU-your-3060-uuid" \
  --output-dir work/pretrained-backward-01
```

The helper builds with CUDA, isolates the selected physical GPU as visible ordinal
0, and uses the qualified TF32 matmul policy with FP32 parameters/gradients.
It retains the 1,024 MiB free-memory polling guard and 30-minute timeout. As with
the forward check, polling can miss transient peaks; it is not a hard allocator
cap. The default 4 GiB limit is a model-loading budget, not a total VRAM cap.

Share `backward-report.json`, `backward.stdout.log`, `backward.stderr.log` and build
logs, or archive the whole new output directory. Model weights/tokenizer are not
copied into the output directory.

A successful report has finite loss and nonzero global gradient norm, per-tensor
finite gradient statistics, zero updates, zero sample-cursor advance and no
optimizer allocation. `connected: false` records a leaf without a graph path;
it does not pretend every unused slot receives a nonzero gradient. Zero gradients
for unused window rows are expected. The report's `training: false` means no
training update; `backward_requested: true` selects autograd diagnostics. The
nested `backward` result is present only after a successful native backward.

The sampled device peak includes graph storage and gradient diagnostic reduction
temporaries, allocator caching and other processes. It is neither an isolated
allocator peak nor a measured optimizer-training peak. A passing backward does
not establish that AdamW candidates/moments fit, or that useful learning occurs.
A reserve-triggered abort or OOM is evidence to inspect before choosing changes.

## Authoring and tokenizer provenance

`fixtures/pretrained-backward.source.json` is a curated synthetic memory probe:

- Capacity is **24 slots**, with **two active windows**. Slots 22 and 23 are used
  by this pinned compiler/UUID. This is not a 24-concurrent-window workload test.
- A masked user observation `A` enters the context window at tick 0.
- The answer window has exactly one supervised `A` target at tick 1.
- Both windows close with excluded-loss EOS at tick 2. Implicit NOOP loss is zero.
- The eight rows are exported as three chunks, one complete tick per chunk.
  The attached cache is retained across the chunk boundaries and backward runs
  once on the complete weighted trajectory objective.

The checked-in `fixtures/pretrained-backward-24/` export was compiled using the
actual supplied converted tokenizer and upstream Trajectory Remix at
`d2c8083536d56a33d876e2650ae855722d491566`. Its normalized tokenizer fingerprint is
`dd428951d25e3b94c907ea4a8e0de493ffd6c96045eace8a18cb56e1cb41cc78`, matching
the conversion manifest. The 5.4 MB uploaded tokenizer and model weights are not
committed. Native validation rejects other tokenizer identities before model
loading. The local converted artifact loader independently checks tokenizer and
weight identities against its conversion manifest.

The source and compiler provenance, export checksum and identity are included
in manifest.json and tokenizer-identity.json. To regenerate into a new directory:

```sh
python3 scripts/generate-backward-fixture.py \
  --upstream /path/to/pinned/trajectory-remix \
  --tokenizer /path/to/converted/tokenizer.json \
  --output-dir work/regenerated-backward-01
```

This development-only generator needs tokenizers 0.23.1 and the pinned upstream
compiler's dependencies. It uses the real compiler/exporter, not manually
constructed replay rows. It verifies that the target sees the context observation,
never sees itself, and that both windows close. The companion
`fixtures/pretrained-backward-tiny/` is the same source compiled against the tiny
converted fixture tokenizer for ordinary CPU CI. No provider calls, external
messages or synthetic specialist execution are used.

## Native command and numerical checks

`backward-remix --input CONVERTED --export EXPORT [--runtime PLAN]` accepts exactly
one trajectory, at most 32 rows and a 1 MiB export. Without a runtime plan it uses
CPU FP32; explicit CUDA placement never falls back. Model gradients are reduced
on their owning device; full model-sized gradient arrays are not copied to host.
Per-tensor L2 norms use max-absolute scaling before the sum-of-squares reduction,
with scalar/global norm accumulation in host f64.

The CPU tests compare device-reduced summaries against full gradient readback,
check parameter/step/cursor preservation, reject nonfinite losses and preallocated
optimizer state, and exercise the actual CLI against a converted tiny model.
Dataset tests verify the real-tokenizer fixture's checksum, masks, chunk count,
capacity and cross-window visibility. The actual 0.6B backward must execute on the
workstation; no GPU memory or gradient result is claimed from cloud CPU checks.
