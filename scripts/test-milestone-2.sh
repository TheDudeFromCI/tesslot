#!/usr/bin/env bash
# Computational gates are opt-in. No base-model downloads or Python runtime.
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/test-tiny-qwen.sh
./scripts/test-qwen-remix.sh
cargo test -p tesslot-cli --test cli local_checkpoint_forward_command_executes_actual_weights --locked -- --ignored --nocapture
cargo test -p tesslot-cli --test cli local_pretrained_remix_cli_trains_and_resumes_with_extended_controls --locked -- --ignored --nocapture
cargo test -p tesslot-tensor --test runtime --locked -- --ignored --nocapture
./scripts/measure-remix-memory.sh
./scripts/test-cuda.sh
