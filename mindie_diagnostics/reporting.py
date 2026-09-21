"""Fresh consent gate for the diagnostics reporting policy. Stdlib only."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

from . import fallback as f

current_consent = f.current_consent
consent_allowed = f.consent_allowed
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


def require_consent(reference):
    """Raise when a fresh policy check rejects this reference."""
    if not f.consent_allowed(reference):
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
