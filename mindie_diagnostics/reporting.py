"""Fresh consent gate for outbound diagnostics reporting actions.

[Role]
Re-validates a saved consent reference against the live `reporting.json` policy
file before and during outbound GitHub operations (`require_consent`,
`guard_consent`, `check_remote_action`).

[Invariants]
- Distinguishes `ConsentUnavailable` (missing/corrupt policy file; preserve
  evidence and retry later) from `ConsentWithdrawn` (explicit user disable or
  revision/repository mismatch; withdraw immediately).

[Timeouts & Locks]
- Synchronous bounded file read (`<= 16 KiB`) via `fallback.read_policy` with no
  locks.

[Side-Effect Policy]
- Strictly read-only policy verification + `ContextVar` guard binding.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

from . import fallback as f

current_consent = f.current_consent
consent_allowed = f.consent_allowed
consent_status = f.consent_status
scope_key = f.scope_key
read_policy = f.read_policy
policy_path = f.policy_path
state_path = f.state_path

_UNSET = object()
_guard = ContextVar("mindie_diagnostics_reporting_guard", default=_UNSET)


class ConsentWithdrawn(RuntimeError):
    """Local reporting consent does not match this reference."""

    def __init__(self):
        RuntimeError.__init__(self, "mindie diagnostics reporting consent is not enabled")


class ConsentUnavailable(RuntimeError):
    """Policy read failed; sending is blocked and prior evidence is retained."""

    def __init__(self):
        super().__init__("reporting_policy_unavailable")


def require_consent(reference):
    """Raise when a fresh policy check rejects this reference."""
    status = consent_status(reference)
    if status == "unavailable":
        raise ConsentUnavailable()
    if status != "allowed":
        raise ConsentWithdrawn()


@contextmanager
def guard_consent(reference):
    """Check consent on entry, then expose it to check_remote_action."""
    require_consent(reference)
    token = _guard.set(reference)
    try:
        yield reference
    finally:
        _guard.reset(token)


def check_remote_action():
    """Recheck the guard reference when one is set. No-op otherwise."""
    reference = _guard.get()
    if reference is _UNSET:
        return
    require_consent(reference)
