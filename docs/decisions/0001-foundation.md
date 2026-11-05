# ADR 0001 — correctness foundation and extension boundaries

Status: accepted for milestone 1. Architecture and numerical-backend choices are
provisional research decisions, not a claim of pretrained architecture stability.

## Decisions

1. **Crates:** core, dataset, model, API and CLI have executable responsibilities.
   An independent feature-gated probe evaluates Burn. HTTP never enters numerical
   dependency paths. Avoid a public general tensor trait until real backend
   requirements are known; retain backend-independent rows/config/device selectors.
2. **Numerics:** implement a tiny f64 oracle first. Evaluate Burn 0.22.1 on CPU;
   require real two-GPU transfer/backward tests before production coupling. Candle
   and LibTorch remain alternatives. See [ADR 0002](0002-backend-gates.md).
3. **Windows:** explicit logical ID, exported embedding slot, inclusive lifetime,
   local insertion position and optional replacement boundary. No hardcoded window
   count. Slot space is not a GPU assignment. Tokenizer controls are distinct native
   IDs; a base model's shared BOS/EOS ID is not a Kidra control mapping.
4. **Execution:** immutable memory and independent query seeds. Local block followed
   by global attention per layer. Compute all D decisions before committing D;
   supervised B queries precede insertion. Port exact upstream visibility for both
   memory and prediction queries. Document the query seed as an experiment.
5. **Dataset:** pinned `kidra-token-chunk/1`, explicit trusted tokenizer identity,
   strict modern slot-aware records, bounded lines/rows and one trajectory in RAM.
   Full sample is the optimization unit. Never silently use chunkwise training or
   detached caches. Preserve source event identities and boundaries.
6. **API:** `window-v1` is independent of application version. Four namespaces,
   structured errors, monotonic per-session event cursors. Reserve unimplemented
   routes with 501. Separate native session/window semantics from ordinary chat.
7. **Dashboard:** build-free semantic HTML, CSS and vanilla JS. Poll the public
   status/device routes. No alternate engine path or simulated metrics. Add a
   frontend framework only when real interaction complexity warrants it.
8. **Persistence:** SQLite metadata/migrations and external artifacts, deferred
   until the first persistent job/task. Atomic checkpoint directory + validated
   manifest + latest-pointer replacement planned before resumable training.
9. **Testing:** actual exporter fixtures, upstream all-pairs visibility oracle,
   mutation rejection, multilayer perturbation, numerical path equivalence,
   weighted-loss derivative and API security tests. Ordinary CI is CPU-only.
10. **Security:** forbid unsafe in Tesslot crates. No production tools in milestone
    1. Reject remote bind until scoped authorization/auditing are implemented.
    Synthetic author controls remain fixture inputs, never privileged API tools.

## Consequences and unresolved risks

Dense full/prefix reference replay costs quadratic attention work; default CLI
limits replay to 4096 rows, independently of the streaming validator's 500000-row
sample limit. Whole-sample reconstruction still needs memory proportional to one
trajectory, and duplicate detection retains sample IDs. Exact disk-indexed global
shuffle and sample packing are later work; stable order/split keys exist now.

The oracle can establish causal and replay correctness without proving training
gradients, CUDA performance, memory feasibility or Qwen compatibility. Production
query construction, window-embedding initialization and global-attention parameter
initialization must be finalized and versioned in milestone 2. Multi-device
activation transfer, graph lifetime, recomputation, optimizer/checkpoint migration
and heterogeneous GPU support remain principal technical gates.
