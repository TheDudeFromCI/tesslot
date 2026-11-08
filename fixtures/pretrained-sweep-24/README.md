# Converted-tokenizer readiness sweep

Ten capacity cases and four training/four held-out pilot trajectories compiled
with the actual converted Qwen tokenizer and pinned Trajectory Remix compiler.
See `manifest.json` for counts, checksums, tokenizer identity and compiler revision;
`sources/` contains the authored scenarios. No model or tokenizer payload is included.

Run and interpret these using the [one-sweep handoff](../../docs/pretrained-readiness-sweep.md).
All 24 windows generate concurrently in the `all-generating` case. These fixtures
stay within the current 512-row bound and do not represent full configured contexts.
