#!/usr/bin/env bash
# Explicit CPU correctness/training gates; never part of default CI execution.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo test -p tesslot-tensor --test remix_experiment --locked -- --ignored --nocapture
cargo test -p tesslot-tensor --lib finite_differences_and_replaced_prefix_graph_remain_correct --locked -- --ignored --nocapture
cargo test -p tesslot-cli --test cli native_remix_training_cli_resumes_without_resetting_sample_cursor --locked -- --ignored --nocapture
