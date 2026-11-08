# Local preparation on Manjaro Linux

The next runnable step is a local readiness collection and tiny CPU training
smoke test. **This preparation harness does not launch pretrained Qwen/Kidra CUDA training.** A
passing preparation report does not qualify a model-sized training run.
The current `train` command initializes only the tiny CPU reference model; it
does not load the supplied Qwen weights or execute on your GPUs.

## One preparation command

From the updated repository checkout, with Rust 1.95+, Cargo and Python 3.10+
on PATH, run this with the actual local Qwen3-0.6B SafeTensors directory:

```sh
python3 scripts/prepare-training-run.py \
  --model-dir "/path/to/Qwen3-0.6B" \
  --output-dir work/local-readiness-01 \
  --cpu-smoke
```

The script builds the native CLI using locked dependencies. It does not install
system packages, download model weights, modify the model directory, or launch
pretrained training. The output directory must be new. Each command's stdout and
stderr are retained along with `readiness.json`; a failed command still leaves
its evidence. Cargo may fetch missing Rust development dependencies normally.
Use `--tesslot-bin /absolute/path/to/tesslot` to use an existing binary instead.

### Standalone SafeTensors file

If your download is one file rather than an HF directory, use:

```sh
python3 scripts/prepare-training-run.py \
  --model-file "/path/to/qwen3-0.6B-Base.safetensors" \
  --output-dir work/local-readiness-02 \
  --cpu-smoke
```

For compatibility, passing that file to `--model-dir` also works. Spaces in paths
are supported; quote the path. The source is inspected in place without renaming,
copying or loading its tensor payload. A sibling `config.json`, when present,
is used for Qwen3 architecture/tensor-shape validation. Otherwise the check is
explicitly `safetensors_metadata_only`: passing it verifies the file header and
layout, not Qwen/Kidra compatibility. If the actual HF config is elsewhere, add
`--model-config "/path/to/actual/config.json"`. The script does not infer config
from the filename or substitute a repository example. This option applies only
to standalone files; directories use their own `config.json`.

Use a fresh output directory on every attempt. The CPU smoke and static scenarios
are independent of source-header validation, so they can pass while the source
check fails. `NotADirectory` from an older script indicates that a file path was
passed to directory-only inspection; it does not establish corrupt weight data.

The initial evidence includes:

- Observed GPU names, UUIDs, PCI bus IDs, driver version, total/used/free memory
  from `nvidia-smi`. Discovery is not proof that CUDA kernels compile or run.
- A **1 GiB (1,024 MiB) reserve per GPU**. Candidate training memory is
  `max(0, min(observed_free, observed_total - reserve))`; existing OS/display
  usage is not subtracted twice. This is a snapshot, not an enforced allocator
  cap or a guarantee that later free memory will remain the same.
- Source `config.json`, actual single-file or indexed SafeTensors shard headers,
  tensor names/shapes/dtypes and header/index identities. Payload finite values,
  full-file hashes, tying/conversion and model execution remain unverified.
- Base 0.6B and provisional 0.8B static scenarios on the RTX 3060. These retain
  their supplied nominal budgets and are not automatically rewritten from the
  GPU snapshot. The 0.8B FP32 AdamW shortfall is an expected report outcome,
  not a reason to apply offload or change the optimizer during preparation.
- Tiny fixture training for four updates, a separate stop-after-one-epoch run,
  resume to completion, and byte-for-byte equality of final CPU weights.
- OS/architecture, logical CPU count, free space on the output filesystem,
  Git HEAD and dirty-file status for provenance. No environment/credential dump.

Read `readiness.json` and the referenced logs. `pretrained_training_ready` stays
false until the remaining execution gates are implemented. Preparation returning
zero means the requested implemented checks completed, not that Qwen training is
available. `insufficient_budget`, `incomplete_estimate`, `skipped` and
`partial_or_skipped` must not be interpreted as training success. Actual requested
build, source, dataset, smoke or CUDA failures return a nonzero script exit code.

The checkpoint smoke test is validated on Linux; native Windows checkpoint
filesystem behavior is not supported by the current implementation. Manjaro is
the intended local host, but it has not been exercised from this cloud session.

## Optional native dataset check

If you already have a native Trajectory Remix export and a trusted matching
native tokenizer identity, add both:

```sh
  --export "/path/to/native-export.jsonl" \
  --tokenizer-identity "/path/to/native-tokenizer-identity.json"
```

This validates the supplied export. It does not train it or infer native controls
from a base Qwen tokenizer. The CPU smoke always uses the checked-in tiny fixture.

## Optional CUDA numerical probes

After the initial report, run a new evidence collection with `--cuda-probes`:

```sh
python3 scripts/prepare-training-run.py \
  --model-dir "/path/to/Qwen3-0.6B" \
  --output-dir work/local-readiness-cuda-01 \
  --cuda-probes
```

This explicitly builds/runs the existing small single/dual-device CUDA
forward/backward/update gates. It needs the working NVIDIA driver and the
Burn/CubeCL compiler/runtime dependencies, including appropriate NVRTC/LLVM
support. Discovery alone cannot validate them; compiler/kernel failures are
recorded as failures. The script does not alter Manjaro packages or drivers.
Keep both devices visible for the heterogeneous-device probe. Use GPU UUID/PCI
identity in the report when relating physical cards to CUDA-visible ordinals;
reordered `CUDA_VISIBLE_DEVICES` can change those ordinals.

No GPU means an explicit skip. If a probe reports SKIP, the outcome is recorded
as partial/skipped rather than a passing full CUDA gate. These probes use tiny
matrices and do not certify the Qwen architecture, BF16, activation memory,
large-model checkpoints or throughput. See [hardware gates](hardware-tests.md).

## Direct source inspection

```sh
cargo run --locked -- inspect-model-directory "/path/to/Qwen3-0.6B"
```

Supports `model.safetensors` or `model.safetensors.index.json` with local shard
filenames. Index-to-header tensor coverage, shard ownership, duplicate tensors,
missing/unindexed SafeTensors files, tensor shapes/dtypes and optional index
`total_size` are checked. Source files must be regular files; symlink shards,
absolute/traversing shard names and ambiguous single-file-plus-index layouts fail.
This header command does not perform conversion; current main provides that
separately through `convert-pretrained`. No pickle loader is used.

## What remains before a pretrained run

Use the report to establish source/hardware status before choosing memory changes.
Current main supplies bounded Qwen/Remix conversion and CUDA experiments.
Model-sized activation, optimizer, checkpoint and memory behavior still require
local qualification. Single-GPU remains the preferred smaller-model target;
both GPUs are the later 1.7B target. The 1 GiB reserve remains in both plans.

## Integration with current main

Main now provides bounded pretrained conversion and Qwen/Remix execution plus
[passing tiny-model CUDA qualification](validation/milestone-2-cuda.md). Use the
[local pretrained workflows](milestone-2-completion.md) for those experiments.
The preparation script here exercises the separate tiny reference trainer and
collects evidence; model-sized 0.6B/0.8B training readiness remains unverified.

## After local CUDA qualification passes: actual-weight forward

Once the full `scripts/test-cuda.sh` suite passes on the workstation, test a
one-token forward on the **base checkpoint** before conversion or optimizer
allocation. This does not run a training update or add Kidra parameters.

Find the RTX 3060's physical UUID:

```sh
nvidia-smi --query-gpu=uuid,name --format=csv,noheader
```

Then run with that UUID and a fresh output directory:

```sh
python3 scripts/pretrained-forward-smoke.py \
  --model-dir "/path/to/Qwen3-0.6B-Base" \
  --gpu-uuid "GPU-your-3060-uuid" \
  --output-dir work/pretrained-forward-01
```

This builds the CUDA CLI, makes only the selected physical GPU visible, and maps
it to runtime CUDA ordinal 0. All layers use TF32 matmuls with FP32 parameter,
normalization and residual storage, matching the qualified TF32 policy. The
layer count and BOS token ID come from the actual source `config.json`.
The default 4 GiB FP32 parameter/source budget accommodates the 0.6B base weights;
it is not a total VRAM allocator limit. No model files are modified or downloaded.

`forward-report.json`, `runtime.json`, build logs and forward stdout/stderr are
retained. A passing report means the actual base model loaded, produced finite
one-token logits of the expected vocabulary shape and completed without an
observed reserve breach. It does not establish learned coordination, validation
quality, long-context memory or optimizer feasibility. The native loader reads
real tensor payloads here; source-header inspection alone did not do that.

Memory is sampled from `nvidia-smi` approximately every 250 ms during the forward.
These are whole-device observations including other processes, not exact peak
allocator measurements; transient peaks between samples can be missed. An
observed free-memory sample below the default 1,024 MiB reserve stops the child
process and records failure. This polling guard is not a hard allocator cap.
The forward times out after 30 minutes; failed compilation/loading/execution
keeps a failure report. No CPU fallback is performed.

The helper's plan construction/failure reporting is unit-tested and the native
forward command is exercised with the checked-in tiny CPU source fixture. The
actual 0.6B CUDA invocation must run on the workstation. After a passing forward,
measure complete conversion parameter count and plan the smallest attached
training workload before allocating optimizer state.

After the actual-weight forward passes, use the [converted-model backward-only
smoke](pretrained-backward-smoke.md) to measure attached trajectory gradients
without allocating AdamW or updating weights.

After the separate one-step SGD diagnostic passes, use the [short multi-step
SGD/checkpoint/restart test](pretrained-training-smoke.md). It keeps the same
minimal trajectory and reserve, saves three new checkpoint directories, and
checks exact continuation before expanding context/concurrency.
