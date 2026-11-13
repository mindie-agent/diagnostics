"""Real Windows path aliases across policy, intake, queues and retention."""
import os
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mindie_diagnostics import fallback, maintenance
from mindie_diagnostics.ingestion import ingest
from mindie_diagnostics.integration import configure_reporting, reporting_status
from mindie_diagnostics.outbox import Outbox
from mindie_diagnostics.reader_registry import reader_guard, register_reader, unregister_reader

pytestmark = pytest.mark.skipif(os.name != "nt", reason="native Windows paths")


def extended(path):
    return Path("\\\\?\\" + str(path))


@pytest.mark.parametrize("long", [False, True])
def test_extended_paths_keep_policy_queue_and_unread_evidence(tmp_path, monkeypatch, long):
    base = tmp_path
    if long:
        for index in range(4):
            base = base / (str(index) + "-" + "nested-runtime-" * 4)
    config, root = base / "reporting.json", base / "logs"
    selected_config, selected_root = extended(config), extended(root)
    result = configure_reporting(True, config=selected_config, roots=[str(selected_root)])
    assert result["status"] == "configured"
    assert fallback.read_policy(selected_config)["decision"] == "enabled"
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_CONFIG", str(selected_config))
    event = fallback.record_failure("mindie-knowledge", "path.activation",
                                    stage="prepare", category="internal", root=selected_root)
    assert event["recorded"]
    queue = Outbox(extended(config.with_suffix(".state") / "reporter.sqlite3"))
    unread = maintenance.prune(selected_root, queue=queue)
    assert unread["unread_files"] == 1 and not unread["reader_unknown"]
    receipt = ingest(selected_root, queue, config=selected_config)
    assert receipt["enqueued"] == 1 and len(queue.rows()) == 1
    status = reporting_status(config=selected_config)
    assert status["queue"]["counts"] == {"pending": 1}
    assert status["local"]["recent"][0]["reporting_eligibility"] == "consent_allowed"
    assert not maintenance.prune(selected_root, queue=queue)["limited"]
    assert list((selected_root / "events" / "mindie-knowledge").glob("*.jsonl"))


def test_ordinary_and_extended_aliases_share_reader_and_cursor(tmp_path, monkeypatch):
    config, root = tmp_path / "reporting.json", tmp_path / "logs"
    assert configure_reporting(True, config=config, roots=[str(root)])["status"] == "configured"
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_CONFIG", str(config))
    assert fallback.record_failure("mindie-knowledge", "path.alias", stage="prepare",
                                   category="internal", root=root)["recorded"]
    queue = Outbox(config.with_suffix(".state") / "reporter.sqlite3")
    first = ingest(root, queue, config=config)
    assert first["enqueued"] == 1
    alias_queue = Outbox(extended(queue.path))
    register_reader(extended(root), alias_queue.path)
    with reader_guard(root) as state:
        assert state["readers"] == [str(queue.path)]
    second = ingest(extended(root), alias_queue, config=extended(config))
    assert second["scanned_bytes"] == 0 and second["enqueued"] == 0
    assert not maintenance.prune(extended(root), queue=alias_queue)["reader_unknown"]
    assert reporting_status(config=extended(config))["queue"]["counts"] == {"pending": 1}
    updated = configure_reporting(True, config=extended(config), roots=[str(extended(root))])
    assert updated["status"] == "configured" and len(fallback.read_policy(config)["roots"]) == 1
    unregister_reader(root, alias_queue.path)
    with reader_guard(root) as state:
        assert state["readers"] == []


def test_runtime_verification_keeps_candidate_interpreter_with_long_staging_directory(tmp_path):
    from mindie_diagnostics import reporting_runtime as runtime

    work = extended(tmp_path / ("generation-staging-" * 5) / ("nested-runtime-" * 6)
                    / ("probe-working-directory-" * 4))
    work.mkdir(parents=True)
    assert len(str(work)) > 260
    # Read the actual candidate distribution, independently of source imports.
    probe = subprocess.run(
        [sys.executable, "-I", "-c", runtime._VERIFY], text=True,
        capture_output=True, check=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    installed = json.loads(probe.stdout)
    runtime._verify(Path(sys.executable), installed["v"], Path(installed["p"]).parents[1], work)
    assert list(work.iterdir()) == []


def test_alias_cursor_reconciliation_keeps_pending_incident_and_unread_bytes(tmp_path, monkeypatch):
    config, root = tmp_path / "reporting.json", tmp_path / "logs"
    configure_reporting(True, config=config, roots=[str(root)])
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_CONFIG", str(config))
    fallback.record_failure("mindie-knowledge", "path.cursor", stage="prepare",
                            category="internal", root=root)
    queue = Outbox(config.with_suffix(".state") / "reporter.sqlite3")
    assert ingest(root, queue, config=config)["enqueued"] == 1
    segment = next((root / "events" / "mindie-knowledge").glob("*.jsonl"))
    with queue.connect() as db:
        db.execute("INSERT INTO cursors VALUES (?,?,?,?,?)",
                   (str(extended(segment)), str(segment.stat().st_ino), 0, queue.clock(), 0))
    retained = maintenance.prune(root, queue=queue)
    assert retained["unread_files"] == 1 and not retained["reader_unknown"]
    result = ingest(root, queue, config=config)
    assert result["scanned_bytes"] > 0 and result["enqueued"] == 0
    with queue.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM cursors").fetchone()[0] == 1
    assert len(queue.rows()) == 1 and not maintenance.prune(root, queue=queue)["reader_unknown"]
