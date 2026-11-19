# Exception Evidence API

[中文](./exception-evidence.md) · **English** · [Back to Documentation](./README.en.md)

`mindie_diagnostics.integration.exception_evidence(component, exception, *, source_root=None)`
(implemented in `mindie_diagnostics.evidence.exception_evidence`) returns a bounded
JSON-safe dictionary without writing files or reporting anything. Use it in a child
process while the original exception and traceback are available, then pass the decoded
dictionary to `record_failure(..., evidence=evidence)` in the parent. Alternatively,
`record_failure(..., exception=exc, source_root=owned_source_directory)` captures
evidence directly in the current process.

`source_root` is a trusted caller-declared absolute directory containing owned
component source. Relative roots are ignored. Script filenames beneath the root
are mapped lexically to the component namespace (for example,
`release_activation.py` becomes `mindie_agent_codex.release_activation`, with
`init=True` recorded when mapped from `__init__.py`). This is not a filesystem
ownership check; callers must never pass user-code directories. The directory path
itself is never retained in evidence. Different-drive or outside-root frames are
not mapped. Existing canonical component module frames (including co-owned
`mindie_harness.*` frames for `mindie-agent-*` host plugins) remain eligible.

Live exception capture via `exception_evidence()` extracts `stack_frames`
(`module`, `function`, `line`, plus optional `init`), a recomputed SHA-256
`stack_fingerprint` (with `stack_truncated` / `stack_incomplete` flags when capped),
`exception_chain` (at most 8 exception class identifiers), signed 32-bit `errno`
derived strictly from non-`TimeoutError` `OSError` instances, and `reason_code`
from `.reason_code` / `.code` exception attributes. Capture walks at most 64
chained exceptions and 4,096 traceback links per exception (buffering the trailing
256 links) and retains at most 24 owned frames. Cyclic exception chains terminate
cleanly. No exception messages, file paths, source text, local variables,
`stdout`, or `stderr` are ever copied.

`mindie_diagnostics.evidence.validate_evidence(component, evidence)` projects a
caller-supplied JSON dictionary (without requiring an active exception object)
onto the allowlist combined with `redact.scan_text` leak checks: it validates
owned `stack_frames` (ignoring any caller-supplied `stack_fingerprint` and
recomputing it from valid frames), `exception_chain`, signed 32-bit `errno` and
`exit_code` integers, and whitelisted string labels (`reason_code`, `substage`,
`contract_name`, plus `soc_generation` in `{"a2", "a3", "a5"}`), while dropping
unknown or malformed fields. Because raw transport cannot prove an integer
`errno` originated from a typed `OSError`, child-process producers should use
`exception_evidence()` to capture exception fields. Validation permits only owned
component (and co-owned `mindie_harness`) frames—never arbitrary user-code
namespaces.

`record_failure` remains best-effort with non-blocking lock acquisition
(`blocking=False`, never raising into callers or starting a worker, while the
JSONL log append itself writes synchronously) and preserves existing consent and
`incident_id` deduplication behavior. Attaching evidence never grants reporting
consent, and validated `evidence` fields take precedence over matching wrapper
fields captured locally on the parent `exception`.
