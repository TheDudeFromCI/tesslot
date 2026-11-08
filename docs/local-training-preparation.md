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
