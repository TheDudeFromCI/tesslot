# Short pretrained SGD run and restart comparison

Run this after the converted-model backward and one-step SGD probes pass on the
workstation. This test exercises repeated full-parameter updates, checkpoint
serialization and fresh-process restoration on the selected GPU. It retains the
same single-target trajectory: **24 slots of capacity, two active windows, eight
rows and three storage chunks**. It does not measure 24 concurrent full contexts.

## Workstation command

Pull the updated PR branch and use the converted Kidra artifact directory. The
selected GPU must already have passed the local CUDA qualification tests.

```sh
python3 scripts/pretrained-training-smoke.py \
  --model-dir "/path/to/Kidra-Qwen3-0.6B" \
  --gpu-uuid "GPU-your-3060-uuid" \
  --output-dir work/pretrained-training-01
```

The default export is `fixtures/pretrained-backward-24/trajectory.jsonl`. The
optional `--export` must contain one complete trajectory, at most 32 rows and
exactly one effective target, within 1 MiB. Native validation checks the full
schedule, tokenizer and masks before updates. The
helper uses SGD without momentum/weight decay, learning rate 0.0001, global norm
clipping at one, TF32 matmuls and FP32 parameters/gradients. Every parameter remains
eligible for optimization. The exact single-target causal prefix retains all
ancestors; zero-loss/future prediction branches do not need to be constructed.
There is no additional model-sized gradient accumulator for this path.

The helper builds the CUDA binary and runs the existing tiny CUDA gather/backward
gate before starting three training processes, sequentially:

1. `continuous/`: start from converted weights, train four updates, save.
2. `stopped/`: start from the same converted weights, train two updates, save and exit.
3. `resumed/`: load `stopped/` in a new process, restore the SGD rate/dataset cursor,
   explicitly relocate to the same CUDA runtime, train two more updates and save.

That is **eight updates across two comparison branches**, each ending at step four.
Each branch repeatedly trains the same synthetic target. Loss reduction is useful
diagnostic evidence, not held-out accuracy or meaningful task-learning evidence.
`--steps N --split-step K` changes the comparison within `2 <= N <= 16` and
`0 < K < N`; defaults are four and two. `--learning-rate` is optional.

## Memory, files and evidence

The output directory must be new. Three model-sized checkpoints are retained:
about **7.2 GB total** for 600,299,520 FP32 parameters. The helper requires free
disk space for four source-weight-file sizes plus 512 MiB before building, to
allow serialization/build margin. Actual host RAM and checkpoint I/O behavior
are part of this workstation test. Restore loads on CPU before explicitly
relocating; the 32 GB host budget still needs this measurement.

The selected physical UUID is isolated as visible CUDA ordinal zero. Whole-device
VRAM is polled approximately every 250 ms during each build/gate/training process.
A sample below **1,024 MiB free** stops the child and records failure. Each process
has a 30-minute timeout. Sampling can miss transient peaks and does not impose a
hard allocator cap. `--max-bytes` remains a per-file/model load budget, not a total
VRAM budget. There is no CPU fallback for the training runtime.

`training-report.json` records checks, losses/norms and step counts, per-stage
commands, exit codes, timing, phase markers, sampled memory and checkpoint hashes.
All source artifact files are hashed before/after and the export is checked for
changes. Native loaders independently check tokenizer identity, weights, dataset
binding and restored state. The source converted artifact is left unchanged.

A passing report requires finite metrics, the expected step/cursor at each saved
boundary, valid checkpoint payload checksums, no AdamW state, and **identical final
weight hashes and checkpoint manifests** between continuous and resumed branches.
Any mismatch remains a failure for inspection; CUDA numerical differences are not
silently accepted through a tolerance. The report keeps `pretrained_training_ready:
false`: a passing minimal SGD test does not establish AdamW capacity, realistic
context/concurrency or useful model behavior.

Keep the checkpoint directories locally. To share evidence without the roughly
7.2 GB of weights, archive logs, runtime JSON and manifests with GNU tar:

```sh
tar -czf work/pretrained-training-01-logs.tar.gz \
  --exclude='*/model.safetensors' -C work pretrained-training-01
```

The trainer saves only at the requested final boundary of each process. If killed
before publication, that process's unsaved updates are lost; a previously published
checkpoint remains usable. This does not implement periodic checkpoints or
automatic recovery after an arbitrary interruption.

## Native commands and verification

The helper uses the existing trainer with explicit SGD and single-target options:

```sh
tesslot train-remix-local fixtures/pretrained-backward-24/trajectory.jsonl \
  --tokenizer-file /path/to/converted/tokenizer.json \
  --pretrained /path/to/converted --output /path/to/new-checkpoint \
  --optimizer sgd --learning-rate 0.0001 --single-target --steps 4 \
  --runtime /path/to/runtime.json --max-bytes 4294967296
```

For continuation, replace `--pretrained` with `--resume CHECKPOINT`; `--steps`
means additional updates. Explicit optimizer/rate values must match the checkpoint.
The existing checkpoint format is reused, including its historical `tiny-qwen-`
prefix; restore still enforces the explicit decoded byte budget and metadata.

CPU CI converts the tiny pretrained fixture, executes four steps versus two plus
two in separate CLI processes, compares every serialized weight and manifest,
checks loss reduction/cursor restoration and rejects changed datasets, rates and
corrupt cursors. Python tests cover checksum/configuration failures, malformed
metrics and termination with a failure report on a reserve breach. The GPU run
remains a workstation gate. After it passes, measure progressively longer causal
contexts and more active windows before a realistic preliminary training run;
AdamW state needs separate qualification if selected for that run.
