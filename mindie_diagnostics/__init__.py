"""Zero-dependency diagnostics package; importing performs no I/O.

[Role]
  Publish the bounded MindIE Agent self-diagnostics primitives (recording,
  consent, redaction, bundles, output capture).

[Invariants]
  * Importing this package performs no filesystem, network or subprocess work.
  * Public names are re-exported from their owning modules; no logic is copied.

[Timeouts & Locks]
  Import-time only; bounded budgets are owned by the individual modules.

[Side-Effect Policy]
  None at import time.  Callers opt in to logging, files, or command execution.
"""
from .context import bind_context, current_context, wrap_context
from .logging import configure, get_recorder
from .bundle import collect_bundle, export_public_event
from .redact import redact_text
from .process_output import capture_output

__version__ = "0.5.5"
__all__ = ["configure", "get_recorder", "current_context", "bind_context",
           "wrap_context", "collect_bundle", "export_public_event", "redact_text", "capture_output"]
