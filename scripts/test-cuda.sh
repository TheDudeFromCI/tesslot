#!/usr/bin/env bash
# No CUDA build or runtime initialization when the hardware preflight is unavailable.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${CUDA_VISIBLE_DEVICES-unset}" == "" || "${CUDA_VISIBLE_DEVICES-unset}" == "-1" ]]; then
  echo 'SKIP CUDA tests: CUDA_VISIBLE_DEVICES disables CUDA. CPU remains the default.'
  exit 0
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi --query-gpu=index --format=csv,noheader > /dev/null 2>&1; then
  echo 'SKIP CUDA tests: NVIDIA hardware/driver discovery unavailable. CPU remains the default.'
  exit 0
fi
if [[ -z "$(nvidia-smi --query-gpu=index --format=csv,noheader)" ]]; then
  echo 'SKIP CUDA tests: no NVIDIA GPUs reported. CPU remains the default.'
  exit 0
fi
# Once hardware is discovered, build/kernel/gradient failures must be visible.
cargo test -p tesslot-backend-probe --features cuda --test cuda --locked -- --ignored --nocapture --test-threads=1

cargo test -p tesslot-tensor --features cuda --test qwen_cuda --locked -- --ignored --nocapture --test-threads=1
