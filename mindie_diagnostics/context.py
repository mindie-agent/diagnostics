"""Diagnostic correlation only: none of these identifiers grants authority."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import wraps
import json
import os
import re

_CURRENT: ContextVar[dict | None] = ContextVar("mindie_diagnostics_context", default=None)
_KEYS = ("trace_id", "operation_id", "parent_operation_id", "phase_id")
_ID = re.compile(r"[0-9a-f]{32}\Z")
CONTEXT_ENV = "MINDIE_DIAGNOSTICS_CONTEXT"


def _validated(value) -> dict:
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in _KEYS
            if isinstance(value.get(key), str) and _ID.fullmatch(value[key])}


def current_context() -> dict:
    value = _CURRENT.get()
    if value is not None:
        return dict(value)
    raw = os.environ.get(CONTEXT_ENV, "")
    if len(raw) > 2048:
        return {}
    try:
        return _validated(json.loads(raw))
    except (ValueError, TypeError):
        return {}


@contextmanager
def bind_context(context):
    token = _CURRENT.set(_validated(context))
    try:
        yield current_context()
    finally:
        _CURRENT.reset(token)


def wrap_context(function):
    """Capture now; run each thread invocation in its own copy of that context."""
    captured = copy_context()
    # Environment-provided context must also be fixed before the new thread runs.
    value = current_context()
    @wraps(function)
    def invoke(*args, **kwargs):
        def call():
            with bind_context(value):
                return function(*args, **kwargs)
        return captured.copy().run(call)
    return invoke
