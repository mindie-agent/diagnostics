"""Test-only cross-process holder for the real reader registry lock."""

from __future__ import annotations

from pathlib import Path


def hold_registry_lock(root, ready, release):
    """Hold the actual reader_guard context open, report it, then wait."""
    from mindie_diagnostics import reader_registry as registry

    guard = registry.reader_guard(Path(root), timeout=5.0)
    state = guard.__enter__()
    try:
        if state["status"] != "ok":
            ready.send(("error", f"reader_guard status={state['status']!r}"))
            return
        ready.send(("locked", None))
        if not release.poll(10.0):
            raise TimeoutError("test lock release was not received within 10s")
        release.recv()
    except BaseException as exc:
        try:
            ready.send(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
        raise
    finally:
        guard.__exit__(None, None, None)
