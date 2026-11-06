#!/usr/bin/env bash
# Opt-in CPU experiments; these are not part of the ordinary CI invocation.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo test -p tesslot-tensor --test qwen_experiment --locked -- --ignored --nocapture
cargo test -p tesslot-api --test http tiny_experiment_http_workflow --locked -- --ignored --nocapture
