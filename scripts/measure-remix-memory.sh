#!/usr/bin/env bash
# Opt-in; fresh child processes keep Linux high-water RSS measurements independent.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run -p tesslot-tensor --example remix_memory --locked -- ordinary
cargo run -p tesslot-tensor --example remix_memory --locked -- recompute
