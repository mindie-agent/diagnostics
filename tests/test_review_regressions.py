"""Independent bounded review regressions; no network or live worker required."""
import json
import os

import pytest

from mindie_diagnostics.outbox import Outbox, QueueFull


def test_published_queue_history_does_not_permanently_block_new_intake(tmp_path):
    queue = Outbox(tmp_path / "queue.sqlite3", capacity=1)
    queue.enqueue("old", "old-operation", {})
    item = queue.claim()
    queue.update(item, state="published", issue_number=1)
    assert queue.enqueue("new", "new-operation", {})


@pytest.mark.parametrize("discard", [False, True])
def test_rotation_between_candidate_stat_and_open_does_not_skip_new_head(tmp_path, monkeypatch, discard, reporting_consent):
    from mindie_diagnostics import ingestion

    def event(identity):
        return (json.dumps({
            "schema": 1, "timestamp": "2026-09-13T00:00:00Z", "monotonic_ns": 1,
            "pid": 1, "component": "mindie-diagnostics", "severity": "ERROR",
            "event": "operation.end", "operation": "operation." + identity,
            "operation_id": identity * 32, "trace_id": "c" * 32,
            "status": "error", "attributes": {"category": "transport"}, "reporting": reporting_consent, "reportable": True,
        }) + "\n").encode()

    folder = tmp_path / "events" / "mindie-diagnostics"
    folder.mkdir(parents=True)
    path = folder / ("1-" + "d" * 32 + ".jsonl")
    path.write_bytes(event("a"))
    queue = Outbox(tmp_path / "queue.sqlite3")
    assert ingestion.ingest(tmp_path, queue)["enqueued"] == 1
    if discard:
        with queue.connect() as db:
            db.execute("UPDATE cursors SET discard=1")
    with path.open("ab") as stream:
        stream.write(b"\n")  # Existing inode now has unread input.
    original = ingestion._candidates

    def rotate_after_stat(root, counts):
        candidates = original(root, counts)
        replacement = folder / "replacement"
        replacement.write_bytes(event("b") + b"\n" * 64)
        os.replace(replacement, path)
        return candidates

    monkeypatch.setattr(ingestion, "_candidates", rotate_after_stat)
    ingestion.ingest(tmp_path, queue)
    monkeypatch.setattr(ingestion, "_candidates", original)
    ingestion.ingest(tmp_path, queue)
    assert len(queue.rows()) == 2


def test_published_history_is_bounded_without_evicting_pending_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr("mindie_diagnostics.outbox.TERMINAL_HISTORY_LIMIT", 2)
    now = [1000.0]
    queue = Outbox(tmp_path / "queue.sqlite3", capacity=2, clock=lambda: now[0])
    for index in range(8):
        queue.enqueue(str(index), str(index), {})
        item = queue.claim()
        queue.update(item, state="published", issue_number=index + 1)
        now[0] += 1
        assert len(queue.rows()) <= 2
    queue.enqueue("pending-one", "pending-one", {})
    queue.enqueue("pending-two", "pending-two", {})
    with pytest.raises(QueueFull):
        queue.enqueue("pending-three", "pending-three", {})
    rows = queue.rows()
    assert len(rows) == 4
    assert {row["fingerprint"] for row in rows[:2]} == {"pending-one", "pending-two"}
    assert {row["fingerprint"] for row in rows[2:]} == {"6", "7"}


def test_bounded_seen_history_does_not_block_intake_and_marker_prevents_repost(tmp_path, monkeypatch, reporting_consent):
    from mindie_diagnostics.reporter import publish_one, GitHub
    monkeypatch.setattr("mindie_diagnostics.outbox.TERMINAL_HISTORY_LIMIT", 1)

    now = [1000.0]
    queue = Outbox(tmp_path / "queue.sqlite3", capacity=1, clock=lambda: now[0])
    for index in range(25):
        queue.enqueue(str(index), str(index), {})
        item = queue.claim()
        queue.update(item, state="published", issue_number=index + 1)
        now[0] += 1
    with queue.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM seen").fetchone()[0] == 20
        assert db.execute("SELECT COUNT(*) FROM incidents").fetchone()[0] == 1
    assert not queue.enqueue("24", "24", {})
    assert queue.enqueue("0", "0", {}, consent=reporting_consent)  # Outside the bounded local lookback.

    class ExistingIssue(GitHub):
        def find_issue(self, item):
            assert item["fingerprint"] == "0"
            return {"number": 1, "html_url": "https://github.com/mindie-agent/mindie-agent/issues/1"}

        def create_issue(self, *args):
            pytest.fail("evicted local history must still reconcile before publication")

    assert publish_one(queue, ExistingIssue())["status"] == "reconciled"
