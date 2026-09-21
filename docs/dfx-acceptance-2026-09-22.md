# DFX acceptance — 2026-09-22

These results are separate from development tests. They use isolated local
configuration; production automatic upload remains off. Windows hardware was
not available and is not claimed.

| Actual execution | Result and boundary |
| --- | --- |
| Codex Luna/max, native MCP | One tool call, one helper JSON failure, one matching local incident; zero SSH, knowledge activation, shell bypass or automatic retry. Model retained the unconfirmed business outcome. |
| Claude Code / configured DSV4 high | One actual MCP call and matching incident. Model did not retry. Its wording confused the status command with reporting; the shared adapter hint now explicitly says read-only and separate opt-in. That wording adjustment was not followed by a redundant model call. |
| Kimi K3/max | Native fixture plugin installed. The model request was rejected by the real OAuth grant before any tool call. Native model consumption remains unaccepted pending login. |
| Real GitHub | Native Codex incident created [acceptance Issue 8](https://github.com/mindie-agent/diagnostics/issues/8). Readback verified its selected fields; it was marked as acceptance and closed. A second authorization and actual helper failure reconciled to that closed Issue without creating another. An off event was skipped; withdrawal prevented a pending publication. |
| macOS launchd | Actual independent installed runtime started. SIGKILL of its verified owned PID did not restart it. Explicit ensure recovered it with a new PID; disable caused clean exit; removal read back absent. No model or diagnostic upload occurred in this service case. |
| Log storage | Real writer children, two SQLite readers, an exclusive SQLite lock and registry lock proved unread protection, bounded waiting, safe replacement and recovery. A long-lived writer accepted 8,000 failure events into four bounded slots; build/consent stayed intact. Blocking/full stderr never blocked recording failure. |
| Transport and queue | Real owned child cancellation cleaned descendants after the leader exited. Total paging deadline and combined output bounds held. Real SQLite verified three durable attempts, unknown POST state, compact receipts, private storage and no budget reset. Fault transport processes are not GitHub end-to-end proof. |
| Components | Real child native/config/deadline failures stayed local. Bad UTF-8, output overflow and a remote internal failure were reportable. Caller missing-required fields, business nonzero, cancellation and normal network failure were not reportable. |

The first Issue exposed a zero-valued process-identity placeholder after switching
writers. It was removed; subsequent real records contain the writer's actual
random process instance. The original Issue evidence was retained. The first
independent runtime failed with the uv macOS interpreter's relative dynamic
library linkage; the fix uses the stable base interpreter through a venv
symlink, preserves the independent venv prefix, and was verified before the
successful launchd case. Failed attempts are retained in the local audit.

Storage totals are maintained offline; arbitrary concurrent writers are not
promised a strict instantaneous aggregate cap. Unknown readers and live/unknown
writers conservatively retain evidence. Contended logging may drop a record and
expose `logging_failed`/drop counts. There is no business replay or model call in
the reporter. The real host tests used candidate native packaging; final commit
pin installation is verified separately and does not imply new Stop delivery.
