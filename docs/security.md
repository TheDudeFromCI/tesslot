# Security boundaries

Milestone 1 is a loopback-only read-only status server. Both JSON configuration
validation and `serve` reject non-loopback addresses. Host validation rejects
arbitrary domains and userinfo. Same-origin assets use a restrictive Content
Security Policy, no-store and nosniff headers. No CORS grant is emitted. This does
not isolate one untrusted local OS user from another: use OS account isolation.

There is no shell/filesystem/tool dispatcher, custom tool module loader, remote
subagent adapter, model downloader, secrets endpoint or job mutation backend.
Reserved execution requests return 501 without effects. Operator CLI file paths
are local trusted input and do not become server capabilities. Reader limits
bound JSON lines and sample expansion; trusted tokenizer metadata is mandatory.
Do not expose the status service through a public reverse proxy as a substitute
for the deferred authentication design.

## Gates before production tool execution

Tools receive explicit capabilities, a canonical workspace root and a separate
budget. Tools cannot inherit all parent credentials or host privileges. Required
contracts include JSON argument/result schemas, required permissions, timeout,
cancellation, status, intermediate observations and auditable operation identity.
Authorization occurs before dispatch and again at privileged resource boundaries.

Filesystem access needs descriptor-relative path handling, containment validation,
symlink/race defenses and normalized paths; string prefix checks are inadequate.
Separate read/write/delete grants, immutable artifact versions and stale-write
checks where applicable. Network research needs destination policy, redirect/DNS
and private-network handling, deadlines and response size limits.

Untrusted subprocess execution requires process/container isolation, resource
limits, scoped mounts, reduced credentials and process-group cancellation. A
Python venv is not a sandbox. Cancellation does not undo already completed effects;
outputs already inserted in B remain committed before a same-tick D EOS.

Persist consequential approvals, denials, dispatches and results with schema
migrations and redacted audit fields. Dangerous operations require an explicit
approval policy. Do not record secrets in prompts, exports, logs or error details.
Avoid executable checkpoint formats. Test denied paths, traversal, symlinks,
timeouts, cancellation, scoped subagents and server restart recovery before
enabling tool API routes.

Synthetic permission/failure/clock/hardware controls are training-author inputs,
not a real security policy. Preserve their observed trajectory rows for replay,
but do not expose host clock changes or failure injectors as production tools.
