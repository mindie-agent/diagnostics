# Exception evidence API

`mindie_diagnostics.integration.exception_evidence(component, exception, *, source_root=None)`
returns a bounded JSON-safe dictionary without writing files or reporting anything.
Use it in a child process while the original exception and traceback are available,
then pass the decoded dictionary to `record_failure(..., evidence=evidence)` in the
parent. `record_failure(..., exception=exc, source_root=owned_source_directory)`
can capture evidence directly instead.

`source_root` is a trusted caller-declared absolute directory containing owned
component source. Relative roots are ignored. Script filenames beneath the root
are mapped lexically to the component namespace (for example,
`release_activation.py` becomes `mindie_agent_codex.release_activation`). This is
not a filesystem ownership check; callers must not pass user-code directories.
The directory itself is never retained in evidence. Different-drive or outside-root
frames are not mapped. Existing canonical component module frames remain eligible.

The allowlist contains `stack_frames` (module, function, line), a recomputed
`stack_fingerprint`, `exception_chain` (class identifiers), and signed 32-bit
`errno` from typed `OSError` exceptions excluding `TimeoutError`. Capture walks
at most eight chained exceptions and 128 traceback links per exception, retaining
at most 24 component frames. Cyclic exception chains terminate. No exception
messages, source paths, source text, locals, stdout, or stderr are copied.

`mindie_diagnostics.evidence.validate_evidence(component, evidence)` projects
untrusted transport onto the same allowlist. Unknown or malformed fields are
removed and any supplied fingerprint is ignored and recomputed from valid frames.
Transport cannot prove an errno originated in a typed exception; producers must
use the capture API. Validation permits only component frames, not arbitrary
user-code namespaces.

Recording remains best-effort and preserves existing consent and incident-ID
deduplication behavior. Evidence does not grant reporting consent. Validated
transported fields take precedence over matching locally captured wrapper fields.
