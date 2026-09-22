# DFX acceptance — 2026-09-22

These results are separate from development tests. They use isolated local
configuration; production automatic upload remains off. Windows hardware was
not available and is not claimed.

| Actual execution | Result and boundary |
| --- | --- |
| Codex Luna/max, native MCP | One tool call, one helper JSON failure, one matching local incident; zero SSH, knowledge activation, shell bypass or automatic retry. Model retained the unconfirmed business outcome. |
| Claude Code / configured DSV4 high | One actual MCP call and matching incident. Model did not retry. Its wording confused the status command with reporting; the shared adapter hint now explicitly says read-only and separate opt-in. That wording adjustment was not followed by a redundant model call. |
| Kimi K3/max | The first native model request was rejected by OAuth before tools. After the user logged in again, one new native turn completed in 29.299 seconds with one MCP call and one matching local incident; no retry, SSH, capture or upload. The model located the incident but also said business state was unaffected despite the unconfirmed result, so complete answer fidelity is not accepted. |
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

The final diagnostics package was then installed non-editably from the official
commit `4a7e50622492c92089c2318d80dbd2a24d41f145`, version `0.3.0`. Its independent
runtime ran through real macOS launchd; the running process's package source and
runtime location were read back, then reporting was disabled and the test
service removed. This final-package case made no model call or upload.

Separate native dependency installations used knowledge `87deb071`, remote-dev
`fb441aa1` and the diagnostics commit above. Fresh native packaging and upgrades
through an older packer are distinct checks: the latter exposed missing adapter
diagnostic modules/version stamps and, for Claude Code, missing new command
matchers. Adapter fixes and their actual acceptance boundaries are recorded in
[Codex](https://github.com/mindie-agent/mindie-agent-codex/blob/main/docs/dfx-acceptance-2026-09-22.md),
[Kimi](https://github.com/mindie-agent/mindie-agent-kimi/blob/main/docs/dfx-acceptance-2026-09-22.md)
and [Claude Code](https://github.com/mindie-agent/mindie-agent-cc/blob/main/docs/dfx-acceptance-2026-09-22.md).
These installation checks do not revise the native model-answer limitations or
claim Windows hardware acceptance.
