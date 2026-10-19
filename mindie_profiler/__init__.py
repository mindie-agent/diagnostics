"""MindIE profiler core: explicit-step Ascend DB/CSV accounting + C7 JSON.

[Role]
  Public package boundary for the compact profiler core.  Re-exports the
  analysis entry point, explicit window records, source-discovery helpers, and
  actionable error types.

[Invariants]
  * ``analyze_profile`` requires an explicit window or ``full_range=True``;
    no step is inferred from kernel rows.
  * The package exposes the C7 ``mindie-profile-summary/1`` dict as its
    cross-package interface; no external submodule is imported here.
  * Missing measurements are ``None``.

[Timeouts & Locks]
  Import-time only; bounded read budgets are owned by ``analyze`` and the
  source adapters.

[Side-Effect Policy]
  No I/O at import time and no network anywhere in this package.
"""

from __future__ import annotations

from .analyze import analyze_profile
from .events import (
    PARSER_VERSION,
    REPORT_SCHEMA,
    Event,
    ProfilerError,
    ProfilerQueryError,
    ProfilerSchemaError,
    ProfilerUsageError,
    SourceRef,
    Window,
)
from .segment import (
    discover_rank_sources,
    load_steps_file,
    parse_step_windows,
    parse_windows,
    resolve_ranks,
)

__version__ = "0.5.0"
analyze = analyze_profile

__all__ = [
    "PARSER_VERSION",
    "REPORT_SCHEMA",
    "Event",
    "ProfilerError",
    "ProfilerQueryError",
    "ProfilerSchemaError",
    "ProfilerUsageError",
    "SourceRef",
    "Window",
    "analyze",
    "analyze_profile",
    "discover_rank_sources",
    "load_steps_file",
    "parse_step_windows",
    "parse_windows",
    "resolve_ranks",
    "__version__",
]
