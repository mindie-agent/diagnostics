# Reporter runtime upgrade acceptance — 2026-09-23

Diagnostics 0.4.0 adds a bounded handoff for the one shared reporter and removes
the retired model-bot and knowledge-community authorization paths. Fault
reporting retains its independent opt-in policy; logging remains usable while
reporting is disabled. No diagnostic command invokes a model.

## Runtime behavior

Automatic maintenance upgrades only an already enabled, active, healthy owned
reporter to a strictly newer semantic version. Identical content is a no-op;
older content is skipped and different content at the same version is a visible
conflict. A stopped worker is not revived. Explicit `reporting ensure` remains
the operator recovery/selection action.

The parent updater passes its actual remaining time to `reporting maintain
--budget-seconds`. The shared deadline includes offline cleanup. A complete
60-second handoff window and exit margin must remain before replacement; forward
work preserves a rollback reserve. A durable attempt suppresses repeated
automatic work on a failed or interrupted target. Service locks, interpreter
verification and process calls respect the remaining deadline. Consent is checked
again after preparation, after taking the native service lock, and before each
stop/start command.

macOS replacement confirms the old label is absent before publishing replacement
descriptors and bootstrapping once. A pre-bootstrap descriptor-write failure
restores the previously verified pair. A terminal receipt-write failure after a
confirmed runtime commit reports degradation without rolling back the service
while leaving the pointer on the new version. Available native error action and
return code are preserved; raw argv/stderr are not returned.

## Real macOS evidence

An isolated launchd service used the official 0.3.0 source
`628374ba816c1e524de245b6664291398367719c`. No production policy or service was
modified, and the experiment created no incidents or GitHub Issues.

1. A real CLI maintenance call with a four-second budget skipped replacement.
   The healthy old PID, runtime source and attempt files did not change.
2. Ordinary maintenance upgraded the service to the reviewed noneditable 0.4.0
   candidate in 0.371 seconds. The native service, healthy worker and committed
   source agreed. The next same-content check retained the same PID.
3. A local, unreleased 0.4.1 copy changed only version declarations. The harness
   verified the newly started process belonged to that exact interpreter and
   manifest, then killed it once. Real OS commands and readbacks were used.
   Automatic rollback restored the healthy 0.4.0 service and old pointer within
   8.492 seconds.
4. A second check for that failed target returned `suppressed`; the combined
   bootstrap/kickstart count stayed at four and the restored PID did not change.
5. A subsequently stopped worker was not revived. Disabled reporting did not
   start it. Cleanup confirmed the label absent, no owned worker remaining and
   zero queued incidents.

The first earlier candidate failed with a generic `command_failed` and restored
its previous healthy runtime. Its failure record was retained and that case was
closed, not retried. Existing logs did not identify the exact original failing
command. The corrected candidate used a new isolated case and source; this report
does not assert a retrospectively proven cause for the earlier failure.

The native candidate source hash was
`4614bdc4ace3359a2445a04aa2a01f91d4e054a68e8c9cf336f240a796c68668`.
That snapshot still contained the already-disabled legacy modules; their later
removal was covered by the complete component suite and clean wheel verification
below. This distinction prevents treating a candidate service test as a final
package or native Windows test.

## Final package and limits

After removing the retired modules and their obsolete tests, the complete local
suite passed: **246 passed, 3 skipped**, with no test-file exclusions. The final
0.4.0 wheel installed noneditably into a clean environment; import/version,
removed-module absence, CLI help, unconfigured read-only status and low-budget
maintenance all passed. These commands did not create an authorization policy or
install a service.

Older-version and same-version/different-content decisions have component-test
evidence; they were not separately exercised against the native service.
Windows native acceptance remains outstanding. GitHub delivery, host-model task
execution and the earlier DFX acceptance are separate evidence, not implied by
these checks. Product implementation used actual local Kimi K3/max sessions,
with bounded processes and no automatic wrapper restarts; tool-driven development
involved multiple model requests. Supervising agents authored the checks and
reviewed the resulting changes.
