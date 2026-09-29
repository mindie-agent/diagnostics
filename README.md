# MindIE diagnostics

Bounded local logs and independently enabled automatic GitHub Issues for
MindIE tool failures. Every executable diagnostics path is model-free. It does
not capture transcripts or retry business operations. The knowledge-review
Grok Bot desktop application is a separate product and is not a diagnostics
command.

Active log writers stat the pressure marker once per record and reuse its
unchanged result. Creation, replacement, modification and deletion apply on
the next record; directory validation runs on changes and before segment
creation or recycling. Stable writes use their existing file descriptor.

## Use

Install the reviewed revision in an existing Python 3.11+ runtime. Adapters
provide native reporting commands; standalone components can use:

```sh
python -m mindie_diagnostics.cli reporting status
python -m mindie_diagnostics.cli reporting configure --enabled true
python -m mindie_diagnostics.cli reporting ensure
python -m mindie_diagnostics.cli reporting configure --enabled false
```

The default destination is `mindie-agent/mindie-agent`. Choose another authorized
repository with `configure --repository OWNER/REPO`. Reporting is off until the
user chooses it, independently of knowledge contribution. Configuration is a
small local write; `ensure` explicitly prepares an independent installed runtime
and starts one owned reporter outside native Hooks. `gh` and its existing login
are required. Installation does not copy transient credentials.

Use `--config /absolute/reporting.json` for an isolated policy. The default is
`$XDG_CONFIG_HOME/mindie-agent/diagnostics.json`, or
`~/.config/mindie-agent/diagnostics.json` on all supported operating systems.
`MINDIE_DIAGNOSTICS_CONFIG` also selects this file. Queue, receipts, health and
runtime are stored beside that policy; adapter runtimes are not borrowed by the
service. `reporting ensure --unit-dir /absolute/directory` isolates a native
service for acceptance. It installs the pure reporter and cannot replace
another reporting authorization.

`reporting status` is read-only. It includes recent local incident references
even when reporting is off, authorization, queue results and worker health.
An enabled setting, an installed service and a healthy running worker are
separate facts. A stopped worker requires an explicit `reporting ensure`;
macOS, Linux and Windows do not automatically restart crashed workers.

## Failure ownership

```python
from mindie_diagnostics.integration import record_failure

result = record_failure(
    "remote-dev", "mcp.request", stage="decode", category="protocol_mismatch",
    exception=error,
)
```

Call only at the component boundary that owns the failure. Pass static product
metadata, never user input as a category or operation. The result contains a
random incident ID only when writing succeeds; otherwise it exposes
`logging_failed`. Preserve a trusted inner diagnostic reference instead of
recording the same error again. Logging never publishes, installs, starts a
model, or changes the original tool result.

Ordinary caller errors, missing configuration, inactive admission, cancellation,
permissions, network/authentication failures and business commands returning
nonzero or timing out do not establish a product defect. Native model failure
alone is also not a bug. These can stay local with `reportable=False`. Confirmed
internal faults and malformed own-tool protocols can be reportable.

Normal operations can use `configure(...).operation(...)` for bounded stages,
monotonic timings and cleanup evidence. Remote stdout/stderr artifacts remain
owned by remote-dev; log maintenance never trims business output.

## Publication and privacy

Public Issues are built without a model, from a strict allowlist: component,
loaded version/revision, operation/stage, static category, duration, exit/cleanup
facts and product code frames. Raw messages, commands, output, prompts,
transcripts, environment values, credentials, session/job IDs, paths and
addresses are excluded. Evidence is observed behavior, not an asserted cause.

Each candidate has at most three persisted processing cycles, including reads
and crashes. A cycle has a total deadline, output and page limits. Unknown POST
results only reconcile; they are never blindly posted again. The public fault
fingerprint is independent of consent and local incident identity. Existing
open or closed Issues are linked without commenting or reopening. Simultaneous
independent clients can still race; this is not a distributed exactly-once
service. Reporter failures stay local.

The active queue holds at most 1,000 candidates for seven days. Terminal evidence
is compacted to small receipts (at most 1,000 / 30 days). Disabling immediately
invalidates new remote actions under the previous authorization. Re-enabling
cannot resurrect its pending evidence. Events produced while off are not
backfilled.

## Log retention

Installed writers use four stable 1 MiB segments per process/component. They
reuse closed segments only after all registered readers have consumed that
inode. Unknown readers, active or unknown writer PIDs and unconsumed evidence
are protected. Space pressure is visible as dropped records / `logging_failed`;
writes resume when safe space becomes available. The dependency-free startup
fallback is limited to one 1 MiB fault file and fails visibly when full.

Existing adapter updater checks call offline `reporting maintain`, including
when reporting is off. The default retention target is seven days, 128 MiB and
512 files per configured root, with bounded scans. A small pressure marker
prevents new growth when protected data blocks cleanup. These are offline
maintenance limits, not an instantaneous aggregate cap under arbitrary
concurrency. Registries are retained conservatively when reader shutdown cannot
be proved. There is no separate cleanup daemon or transcript scan.

```sh
python -m mindie_diagnostics.cli reporting maintain
python -m mindie_diagnostics.cli reporting maintain --update-running
python -m mindie_diagnostics.cli bundle --root /absolute/log-root --operation-id INCIDENT
```

`reporting maintain` stays offline unless `--update-running` is passed.
`--update-running` is optional. It affects only an already enabled, healthy
worker, and only to adopt a higher semantic version of the installed runtime.
It does not start or revive a stopped or unhealthy worker. `--unit-dir` is the
optional owned service directory for that inspection. There is no automatic
revival.

Adapters pass their remaining time through `--budget-seconds`; maintenance
caps this at 75 seconds and deducts time already spent on local cleanup.
Insufficient time skips the handoff before changing the service. A failed
target is recorded and suppressed on later automatic checks; inspect status
and use an explicit `reporting ensure` to recover. Automatic maintenance never
downgrades the shared worker, and different content with the same version
reports a conflict. Failure results retain the system action and return code
when available, without exposing raw commands or stderr.

Bundle export is local and does not upload. Normal updater logs go to structured
state / bounded diagnostics, not indefinitely appended service log files.

## Validation

Run `python -m pytest` for development regressions. They do not establish native
host behavior, GitHub delivery or Windows hardware acceptance. Real macOS
process, rotation, Issue and native adapter evidence is recorded separately.
Windows hardware remains deferred to the user's dedicated environment.

Diagnostics commands do not launch a model. The desktop Grok Bot remains an
external application and is not an entry point of this package.
