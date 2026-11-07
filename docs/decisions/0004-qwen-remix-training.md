# ADR 0004: general external-first Qwen query/memory execution

Status: accepted for the bounded CPU training gate. Production backend selection
and NVIDIA capability gates remain provisional under ADR 0002.

Reuse the echo experiment's full-rank Qwen/Kidra exchange and local blocks, while
replacing the two-window task mask with the exact Trajectory Remix predicate.
Extract one shared numerical kernel accepting causal masks, durable memory rows
and optional attached projected prefix K/V. Dense full-sequence execution and
tick/chunk execution differ in their memory construction; both apply visibility
to memory and query states in every layer.

Keep target queries distinct from durable token states. A/B observations are
inserted in external-first order; supervised B targets are predicted before
insertion. Compute all D queries and pending durable states against one after-B
snapshot, then commit together. NOOP never creates KV. Closure/replacement evicts
keys but retains autograd dependencies through surviving states. A cache carries
model/sample/update identity and is invalid across optimizer boundaries.

One whole trajectory is one optimization unit. Normalize its effective weights
once and backward once across all storage chunks. Never silently implement
truncated backpropagation or ordinary next-token chat training. Initialize unused
parameter gradients to zero explicitly so full-parameter optimization remains
well-defined on sparse window activity.

Tokenizer alignment is a required precondition, not a consequence of matching
embedding shapes. The CLI creates a new synthetic Qwen with the actual export
tokenizer/control vocabulary. The lower-level aligned-weight constructor records
the supplied source hash, conversion version and initialization seed, and rejects
vocabulary/control mismatches. It does not perform pretrained tokenizer extension
or reinterpret the ASCII echo checkpoint. This avoids silently discarding or
remapping pretrained embedding rows while proving the execution/training contract.

Version the general architecture and checkpoint formats separately from echo
experiments. Preserve exact tokenizer and dataset identity, whole-sample progress,
source provenance and optimizer moments. The CLI preflights the entire file and
resumes deterministic sequential sample order from the saved cursor. An immutable
file hash is deliberately stricter than semantic-equivalence heuristics.

Bound allocations and make this a CPU-only, opt-in headless gate. Use canonical
exports from the unmodified pinned exporter, compare dense/cached gradients,
finite-difference representative coordinates, verify source dependencies survive
eviction, and compare resumed bytes. Do not advertise production HTTP jobs, online
KV replacement, mixed precision, real model loading or multi-GPU training merely
because these CPU tests pass.
