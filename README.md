# diagnostics · MindIE Agent

Structured logs, phase timings, sanitized support bundles, automatic GitHub
issues and an independent Grok diagnosis worker for MindIE components. The Python
package has **no runtime dependencies**. GitHub publishing uses an existing `gh`
login or a token from the process environment when `gh` is unavailable; Grok
diagnosis uses an optional, separately installed Grok Build CLI.

Logging does not make network requests, install packages, grant task ownership,
retry a workload or wait for a model. Each component retains its execution and
resource state. Importing this package does not configure Python's root logger.

## Record an operation

```python
from mindie_diagnostics import configure, wrap_context

log = configure("mindie-coordinator")
with log.operation("prepare") as operation:
    with operation.phase("lock.wait"):
        acquire_lock()
    with operation.phase("build"):
        build()
    operation.event("INFO", "reuse.selected", cache_hit=True)
diagnostics = operation.summary()
```

Exceptions are recorded and re-raised unchanged. Returned failures can use
`operation.fail("transport", retryable=False, submission_state="uncertain")`.
The summary contains the real start/end, monotonic duration, bounded phase
summary, operation/trace IDs, local record reference and diagnostic write status.
Use `wrap_context(callable)` when submitting work to a thread pool;
`current_context()` / `bind_context()` carry diagnostic association through
internal RPC metadata. These identifiers are never task or resource authority.

`DEBUG`, `INFO` (default), `WARNING` (`WARN` accepted), `ERROR`, and `CRITICAL`
follow Python logging levels. `MINDIE_LOG_LEVEL` sets the level.
`MINDIE_DIAGNOSTICS_ROOT` overrides the platform user state directory:

- Windows: `%LOCALAPPDATA%/mindie/diagnostics`
- Linux/macOS: `$XDG_STATE_HOME/mindie/diagnostics`, defaulting to
  `~/.local/state/mindie/diagnostics`

Each process writes `events/<component>/<pid>-<nonce>.jsonl`, rotating at 1 MiB
with three backups. Records are limited to 16 KiB. UTC timestamps and a random
process clock identifier accompany monotonic times; never subtract monotonic
times from different clock identifiers or add parallel phases into elapsed time.
File errors produce one bounded stderr warning and a `logging_failed` flag; they
cannot mask the business result or prevent cleanup. stderr is never MCP stdout.

Long-lived service backends can wrap their own main function with
`capture_output(operation)`. The existing process drains native stdout/stderr
into the same rotating sink; no new service owner or launcher is introduced.
Complete lines are redacted before truncation. Oversized or incomplete lines
produce explicit gap records instead of exporting potentially partial secrets.
Do not use process-wide FD capture around concurrent MCP request handlers.

## Inspect or attach diagnostics

```sh
mindie-diagnostics bundle --root /path/to/diagnostics --operation-id OPERATION_ID --output support.json
```

This command is offline. It reads existing logger segments only, without live
probes or repair. The default bundle is bounded to 1 MiB / 1,000 events, with
explicit clipping, omission and unreadable-input indicators. Files, directories,
commands, prompts and arbitrary raw output are not uploaded. The export first
selects known structured fields, then applies the shared redaction rules and a
final scan. A redaction failure prevents publishing; it does not suppress the
original local incident. The bundle includes a content hash.

Automatic issues embed a smaller bounded JSON evidence window in the issue body,
so diagnostics remain readable without a transient attachment server. Stack
locations are module/function/line records, with no source text or frame locals.
Package version/revision and exception chains help maintainers locate failures.
Private free-form output stays excluded, even if a regex scanner finds no secret.

## Enable automatic issues

The independent worker publishes only for a workspace with current explicit
community consent. Installing or running a worker alone grants no contribution
permission. Merely importing or installing the library does not upload anything.
No per-tool approval or Agent report is needed after the workspace choice.
Explicit caller input errors and user cancellation remain in local diagnostics;
they do not automatically create MindIE bug reports. Unknown failures are retained
for diagnosis rather than guessed to be caller mistakes.

```sh
gh auth login
mindie-diagnostics worker --root /path/to/diagnostics --state /path/to/reporter-state
```

The workspace onboarding owns an atomic, untracked `.mindie-local/community.json`:

```json
{"schema":"mindie.community.v1","workspace_id":"0123456789abcdef0123456789abcdef","decision":"enabled","revision":"fedcba9876543210fedcba9876543210"}
```

`workspace_id` is a stable random 32-character lowercase hexadecimal identifier.
Each explicit decision change gets a new random `revision`; idempotent setup
keeps it. Pass the canonical project's absolute receipt path through
`MINDIE_COMMUNITY_POLICY` to task processes. A task clone reuses that pointer,
not a copied consent file. No receipt is inferred from cwd, GitHub identity,
diagnostic correlation IDs or a recent task. Token-based `gh` authentication
(`GH_TOKEN` / `GITHUB_TOKEN`) works independently of the consent decision, even
without `gh`: the fallback uses fixed-host HTTPS and refuses redirects. Do not
store a token in the receipt or a Git URL.

For a process serving multiple workspaces, bind each request explicitly:

```python
from mindie_diagnostics import bind_community_policy, read_policy

with bind_community_policy(project_policy_path):
    handle_request()
# bind_community_policy(None) explicitly prevents sharing for one request.
```

The library records a private policy reference on a new operation only when the
receipt is enabled. Logging and offline support bundles continue when disabled.
Policy paths, workspace IDs and consent revisions never enter public evidence.
The worker re-reads the matching receipt before intake and each GitHub request,
model request and reply. A missing, malformed, inaccessible or changed receipt
fails closed. Windows drive paths are mapped to `/mnt/<drive>` when read by a
WSL worker; UNC and relative receipt paths are unsupported.

Revoking one workspace withdraws its queued upload and diagnosis work without
stopping the shared worker or affecting other consenting workspaces. Old
unscoped logs and queue entries remain local; enabling participation later
does not retroactively enroll them, and enabling it again does not reactivate
withdrawn revisions. A request already sent before revocation cannot be recalled;
the next external action is blocked. Existing public issues/comments are not
deleted by changing local consent. These limits are not an exactly-once or
remote deletion guarantee.

The default destination is
`mindie-agent/mindie-agent`; `--repository owner/repo` overrides
it. Repeat `--root` to watch multiple explicit diagnostic roots. `--once` runs one
cycle for Task Scheduler, systemd timers or other service managers; otherwise the
worker repeats every 60 seconds. `--interval` changes this (minimum 5 seconds).
`--since` accepts a fixed ISO timestamp with a timezone when an installation
should observe future incidents without backfilling historical logs. Keep the
same timestamp across restarts; it does not delete older local evidence.
Run it as the user whose tools produce these logs, with that user's GitHub login.
Do not put tokens in command arguments, repository files or diagnostic bundles.

For continuous operation, install the package into a permanent, non-editable
virtual environment, then explicitly enable its user service:

```sh
/path/to/venv/bin/mindie-diagnostics service install --root /path/to/diagnostics --state /path/to/reporter-state
/path/to/venv/bin/mindie-diagnostics service status
```

Linux/WSL uses systemd, Windows uses a user Task Scheduler task running `pythonw`,
and macOS uses a user launch agent. Native Windows/macOS installations can select
an independent profile with `MINDIE_DIAGNOSTICS_SERVICE_DIR`. The service runs in
the background without opening a terminal window.

The owned systemd unit restarts after failure, clears Python source overrides,
uses a private umask and rate-limits its journal. Reinstallation preserves the
initial observation timestamp. Add the same `--grok`, `--grok-home` and
`--grok-work` options to include diagnosis. `--environment-file` accepts an
explicit private 0600 file when the service needs credentials outside an
interactive login. Credential values never enter the service definition or
command line. With explicit `--save-token`, the current `GH_TOKEN` or
`GITHUB_TOKEN` is saved to `state/credentials.env` with POSIX mode 0600 or a
Windows DACL limited to the current user and SYSTEM. No token is saved by
default. The installer refuses unrelated units, editable packages and system Python.
`service remove` stops only this unit and preserves logs and queues.

First-use clients should call `service ensure` with the same installation
arguments. Under the service owner's lock it adds new log roots to an existing
local reporter and retains its state, credential file, GitHub executable,
optional Grok profile and observation start time. A requested installed Python
runtime may replace the previous runtime. Repeated calls do not restart an
unchanged worker. An existing central bot or different destination repository
is refused instead of being replaced.

For `ensure`, `--save-token` applies only when creating the service. Existing
authentication is retained even when another clone has a different token in
its environment. Use explicit `service install` to rotate authentication or
replace configuration. Credentials are never read back by the installer.

The service manager must run while observation is wanted. Linux installations
may enable user lingering through their administrator. On Windows the worker
can run in WSL; a login Task Scheduler action can start the WSL user service.
Existing WSL installations can keep their Linux service. Native Windows/macOS
installations use the same `service install`, `service status` and
`service remove` commands; a separate WSL helper is not required.

```sh
mindie-diagnostics status --state /path/to/reporter-state
```

The outbox keeps immutable sanitized evidence, occurrence counts, publication
state and last error. Explicit owner `classification=caller` or `cancelled`
remains in the bounded export and is not submitted as an automatic issue.
Only `caller`, `cancelled` and `unknown` are accepted classifications; generic
validation/permission categories and nonzero exit codes do not imply caller error.
The legacy explicit `caller`/`cancelled` categories remain supported.
It deduplicates within a workspace and consent revision by component, version,
operation, failing phase and error fingerprint. Byte cursors survive worker restarts; deduplication
also handles rereading rotated segments. The queue is capped at 1,000 unpublished incidents,
with a maximum of 10 issue submissions per hour. Full queues retain existing
incidents and expose backpressure. Published/withdrawn records and occurrence identifiers
expire after 30 days. Published history is separately bounded to 1,000 records;
the oldest published history and occurrence IDs can expire earlier at capacity.
An expired local marker still goes through GitHub reconciliation before any POST.
Worker retention removes settled log segments beyond seven
days, 128 MiB or 512 files, preserving files modified in the last five minutes;
unread segments remain until ingestion catches up. `limited` and `unread_files`
report when writers or intake backpressure prevent meeting the retention target.
An intake failure does not stop already queued issues or diagnoses from draining.

`status` also reads the worker's atomic heartbeat, current stage, last cycle and
degraded state. A stale heartbeat and a live process making no progress are
reported separately. Heartbeats never claim GitHub publication succeeded.

GitHub errors back off. A POST timeout or a crash after starting a POST is
**uncertain**, not proof that creation failed. The next attempt searches direct
REST issue listings for a stable marker before posting. If no match is found,
uncertain entries remain visible for reconciliation and are not blindly resent.
This is deliberately not an exactly-once claim. Reconciliation is bounded and
will report an exhausted listing window instead of declaring a false absence.

## Add the Grok diagnostic bot

Create a dedicated profile and log in to it using the installed Grok CLI:

```sh
mindie-diagnostics grok-profile --home /path/to/grok-bot-home
GROK_HOME=/path/to/grok-bot-home grok login
mindie-diagnostics worker --root /path/to/diagnostics --state /path/to/reporter-state \
  --grok grok --grok-home /path/to/grok-bot-home --grok-work /path/to/empty-bot-work
```

On PowerShell, set `$env:GROK_HOME` before `grok login`. A Windows installation
can also run the worker in WSL and pass the Windows `gh.exe` path through `--gh`.
The existing personal Grok configuration is never overwritten. The dedicated
configuration disables tools, memory, compatibility discovery and subagents;
the adapter inspects active hooks/plugins/MCP/project instructions before each
request. Newly materialized built-in skills are disabled in this owned profile.
The Grok Build CLI integration has been exercised with version 1.0.25.

The bot receives immutable structured evidence from this worker's locally
published, consent-bearing reporter records. It does not scan the whole
repository for other installations' issues. The same private consent reference
travels into the bot queue and is checked again before generation and comments.
Issue prose, links and comments do not become execution instructions. Grok has
no enabled business tools and returns observations, hypotheses, missing evidence
and suggested checks. Its response is bounded and leak-scanned before posting.
A separate queue caps model requests at ten per hour, caches results and reconciles comment markers after lost
responses, avoiding duplicate comments and unnecessary model reruns. The bot does
not change services, close issues or merge fixes. A model hypothesis is not a
confirmed root cause. GitHub/model availability never delays the original tool.

This is a locally supervised Grok Build diagnostic worker, not a provisioned
Grok cloud Bot or a newly created GitHub account. Comments use the configured
GitHub credential and identify the diagnostic worker explicitly.

### Maintainer-operated central diagnosis

Ordinary users need only the reporter and GitHub authentication. They do not
need a Grok account to receive diagnosis from a maintainer's central service.
A maintainer can explicitly authorize a separate central worker:

```sh
mindie-diagnostics worker --central-bot --state /path/to/central-state \
  --repository mindie-agent/mindie-agent \
  --grok grok --grok-home /path/to/grok-bot-home --grok-work /path/to/empty-bot-work
```

Use the same `--central-bot` option with `service install` for continuous
operation. This mode requires Grok, rejects local `--root` inputs, and never
ingests or uploads local logs. It reads only already-public open automatic MindIE
issues with valid bounded evidence, using a separate `central-bot.sqlite3`
queue. Repository text cannot switch an ordinary worker into this mode.

Local consent withdrawal stops the client's contributions and local queued
diagnoses; it cannot recall a public GitHub issue or cancel work already started
by a separate maintainer service. Stop that central service to stop its work.
Changing modes does not grant consent to local queue entries. Published comment
markers are reconciled before any model call, and evidence markers avoid
duplicate diagnosis across local/central queues. Use fresh state for this
release; previous database formats are not migrated. Current-format owned
service configurations support ordinary idempotent `service ensure` reuse.

## Development

```sh
uv venv
uv pip install -e '.[test]'
uv run --no-project python -m pytest -q
uv build
```

CI tests Python 3.11 and 3.13 on Linux, Windows and macOS, and installs a built
wheel without dependencies. Tests cover write failures, context propagation,
concurrent rotation, partial and malicious logs, outbox races and capacity,
GitHub rate limiting, lost POST replies and bot isolation. Synthetic public
issue/comment acceptance is separate from transport mocks.

The record model follows [Python logging](https://docs.python.org/3/howto/logging.html)
and the [OpenTelemetry logs data model](https://opentelemetry.io/docs/specs/otel/logs/data-model/),
without requiring an OpenTelemetry collector or SDK.
