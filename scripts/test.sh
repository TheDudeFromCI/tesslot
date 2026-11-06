#!/usr/bin/env bash
# CPU tests always run; hardware tests are conditional and never required by CI.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo test --workspace --features tesslot-backend-probe/burn-cpu --locked
exec bash scripts/test-cuda.sh
