"""OSError diagnostics keep errno and an explicit safe path, never a local filename."""

import errno
import json
from pathlib import Path

from mindie_diagnostics.bundle import export_issue_event
from mindie_diagnostics.integration import record_failure
from mindie_diagnostics.local_snapshot import recent_failures


def _event(root):
    files = list(Path(root).glob("events/*/*.jsonl"))
    assert files, "expected a local diagnostic event"
    return json.loads(files[-1].read_text(encoding="utf-8").splitlines()[-1])


def test_oserror_publishes_errno_and_safe_path_only(tmp_path):
    private = "/Users/private/state/mcp-binds.sqlite3"
    exc = PermissionError(errno.EACCES, "Permission denied", private)
    exc.diagnostic_path = "mcp-binds.sqlite3"
    ref = record_failure(
        "mindie-agent-cc", "mcp.call", stage="finish_bind", category="local_storage",
        exception=exc, root=tmp_path,
    )
    assert ref["recorded"] is True
    event = _event(tmp_path)
    assert event["attributes"]["errno"] == errno.EACCES
    assert event["attributes"]["path"] == "mcp-binds.sqlite3"
    assert event["attributes"]["error_type"] == "PermissionError"
    assert private not in json.dumps(event)
    public = export_issue_event(event)
    assert public["attributes"]["errno"] == errno.EACCES
    assert public["attributes"]["path"] == "mcp-binds.sqlite3"
    assert private not in json.dumps(public)
    recent = recent_failures([str(tmp_path)])["recent"]
    assert recent[0]["incident_id"] == ref["incident_id"]
    assert recent[0]["errno"] == errno.EACCES
    assert recent[0]["path"] == "mcp-binds.sqlite3"
    assert private not in json.dumps(recent)


def test_absolute_diagnostic_path_and_timeout_are_not_published(tmp_path):
    private = "/Users/private/state/mcp-binds.sqlite3"
    disk = OSError(errno.ENOSPC, "No space left on device", private)
    disk.diagnostic_path = private
    record_failure(
        "mindie-agent-cc", "mcp.call", stage="finish_bind", category="local_storage",
        exception=disk, root=tmp_path,
    )
    event = _event(tmp_path)
    assert event["attributes"]["errno"] == errno.ENOSPC
    assert "path" not in event["attributes"]
    assert private not in json.dumps(event)
    public = export_issue_event(event)
    assert public["attributes"]["errno"] == errno.ENOSPC
    assert "path" not in public["attributes"]

    timed = TimeoutError(errno.ETIMEDOUT, "timed out", private)
    timed.diagnostic_path = "mcp-binds.sqlite3"
    record_failure(
        "mindie-agent-cc", "mcp.call", stage="finish_bind", category="local_storage",
        exception=timed, root=tmp_path / "timeout",
    )
    timeout_event = _event(tmp_path / "timeout")
    assert timeout_event["attributes"]["error_type"] == "TimeoutError"
    assert "errno" not in timeout_event["attributes"]
    assert "path" not in timeout_event["attributes"]
    assert private not in json.dumps(timeout_event)
