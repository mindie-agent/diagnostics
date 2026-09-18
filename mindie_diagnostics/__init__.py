"""Zero-dependency diagnostics; importing this package performs no I/O."""
from .context import bind_context, current_context, wrap_context
from .logging import configure, get_recorder
from .bundle import collect_bundle, export_public_event
from .redact import redact_text
from .process_output import capture_output
from .community import bind_community_policy, current_consent, read_policy

__version__ = "0.2.1"
__all__ = ["configure", "get_recorder", "current_context", "bind_context",
           "wrap_context", "collect_bundle", "export_public_event", "redact_text", "capture_output",
           "bind_community_policy", "current_consent", "read_policy"]
