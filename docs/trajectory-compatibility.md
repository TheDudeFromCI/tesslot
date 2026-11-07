# Trajectory Remix compatibility

Source of truth: [Trajectory Remix](https://github.com/TheDudeFromCI/trajectory-remix)
at **d2c8083536d56a33d876e2650ae855722d491566**. Research covered `docs/EXPORT.md`,
`docs/TOOLS.md`, `docs/SANDBOX.md`, `docs/kidra-scenarios.md`, `compiler.py`,
`training.py`, compiler/training/header/reference tests and built-in tool sources.
Tesslot never changes or imports that Python package at runtime.

## Supported contract

The native reader accepts modern, explicit-slot `kidra-token-chunk/1` records,
`external-first-v1`, the exact pinned durable-cache policy and either
`uuid-hash-v1` or `declaration-v1`. It validates rather than reallocates exported
slots. `max_window_slots: null` is supported with capacity inferred from window
count. Legacy missing-slot fallback is intentionally rejected, not reinterpreted.
Unknown schema fields/versions are rejected to avoid silent semantic changes.

The reader validates:

* Exact tokenizer fingerprint and control mapping against a supplied trusted
  identity, plus vocabulary-bounded token IDs. Hashes are lowercase SHA-256.
* UUID/sample identity, contiguous samples, zero-based chunk order, prefix row
  requirements, chunk counts, row counts, adjacent inclusive tick intervals and
  no split-tick boundaries. Duplicate sample IDs or UUIDs are rejected.
* Identical lifetime/slot/tokenizer/count metadata across chunks; unique bounded
  slot mappings; all BOS opens and EOS closes; writes within lifetimes.
* Row indices, phase ordering, unique D decisions per window/tick, ordered B
  ranks, local positions, replacement boundaries and durable-only NOOP behavior.
* Every redundant array, unshifted labels, masks, finite nonnegative effective
  weights, excluded-loss headers, and policy versions.

It preserves excluded observations, source operation/event/rank, supervised B
targets, fractional weights, implicit NOOPs, headers, global ticks and removal
metadata. `Sample::ticks` exposes complete tick groups. Chunk ends remain explicit
so a later trainer can maintain a single attached prefix graph.

Only durable rows can become memory keys. Closed windows disappear after D;
replacement removes old non-header context at its B boundary, including empty
external replacements that produce no new row. B and D visibility is checked
independently of loss masks. The all-pairs visibility fixture includes keys with
zero loss as well as non-durable NOOP targets.

Loss is `sum(effective_weight * cross_entropy) / max(sum(effective_weight), 1)`.
The clamp matters even for a total weight below one. There is no extra label shift
and no averaging of chunk means. The [Qwen/Remix CPU reference trainer](qwen-remix-training.md)
now implements this objective with attached projected KV across chunks;
production job scheduling and large-model SFT remain unimplemented.

## Fixtures and reproducibility

`scripts/generate-fixtures.py` invokes the actual pinned `Compiler`, `chunks` and
`visible_to`, with a checked-in author source, native demo tokenizer and fixed UUID.
It exports canonical chunked/single-record variants, a visibility matrix, tokenizer
identity, full tool schemas, a 2050-tick idle/late-observation export and a hash
manifest. The visibility matrix is bit-packed in little-bit order per row, so
all pairs remain auditable without a multi-megabyte boolean listing. The canonical tool output comes
from the real synthetic calculator scheduler. No hand-authored JSON stands in for
an exporter result. Fixtures are mechanical compiler examples, not human-approved
Ready training data and not evidence of real specialist competence.

Regenerate in a disposable development environment:

```sh
python3 -m venv /tmp/remix-fixtures
/tmp/remix-fixtures/bin/pip install tokenizers==0.23.3 jsonschema==4.26.0 cloudpickle==3.1.2 simpleeval==1.0.8
git clone https://github.com/TheDudeFromCI/trajectory-remix /tmp/trajectory-remix
git -C /tmp/trajectory-remix checkout d2c8083536d56a33d876e2650ae855722d491566
/tmp/remix-fixtures/bin/python scripts/generate-fixtures.py /tmp/trajectory-remix
```

Fingerprint is the upstream tokenizer's serialized normalized state after padding
and truncation are disabled, not the original file byte hash. Native tokenizer
execution/model identity derivation is a milestone-2 task; do not trust an
arbitrary identity supplied by an untrusted training client.

## Limits and deferred features

The streaming validator bounds lines to 64 MiB, samples to 500000 rows, ticks to
the upstream 100000-tick limit and slot capacity to 4096 by default. It materializes
one trajectory, not the whole corpus, while retaining seen identities for exact
duplicate checks. Callers must exhaust the iterator and handle errors to certify
the complete file. Stable UUID-based split/order keys keep chunks together.
`shuffle_samples` supplies a seeded bounded-buffer shuffle of complete samples;
its configurable sample buffer bounds the number of resident trajectories. It is
not a globally uniform permutation of an arbitrarily large disk corpus.

Disk-indexed global shuffle, sample packing, cross-sample attention isolation in
packed tensors, arbitrary evaluation metadata sidecars and huge-corpus identity
indexing are deferred. Cross-sample isolation currently follows from independent
forward/cache invocations. No SFT, truncated backpropagation or lossless resumption
is claimed. Schema changes upstream require a new audited compatibility decision
and regenerated fixtures, not a permissive fallback parser.

The general Qwen path now rechecks materialized `Sample` objects before numerical
execution, preserving the reader's validation boundary even for native callers.
Dense and real projected-KV execution agree on the original canonical/autograd
exports. The relevant upstream docs/compiler/training sources were rechecked
against current main and still match the pinned contract. See the
[Qwen/Remix report](qwen-remix-training.md) for exact coverage and CPU limits.
