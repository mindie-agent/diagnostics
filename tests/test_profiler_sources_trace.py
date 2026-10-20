"""Tier-1 tests for streaming Chrome Trace JSON and the sidecar index."""

from __future__ import annotations

import json
import sqlite3

import pytest

from mindie_profiler.events import ProfilerSchemaError
from mindie_profiler.sources_trace import (
    build_trace_index,
    index_is_valid,
    iter_trace_events,
    query_trace_events,
)


def _events(count: int = 4) -> list[dict]:
    return [
        {
            "name": f"kernel_{index}",
            "cat": "kernel",
            "ph": "X",
            "ts": index * 10,
            "dur": 5,
            "pid": 1,
            "tid": 2,
        }
        for index in range(count)
    ]


def test_streams_array_root_across_chunk_boundary(tmp_path):
    events = _events()
    events[1]["name"] = "cross_" + ("x" * 5000)
    trace = tmp_path / "trace_view.json"
    trace.write_text(json.dumps(events), encoding="utf-8")

    parsed = list(iter_trace_events(trace))
    assert [event["name"] for event in parsed] == [event["name"] for event in events]


def test_streams_object_root_with_metadata_before_trace_events(tmp_path):
    trace = tmp_path / "trace_view.json"
    trace.write_text(
        json.dumps({"displayTimeUnit": "ns", "traceEvents": _events(3), "other": {"a": [1, 2]}}),
        encoding="utf-8",
    )
    parsed = list(iter_trace_events(trace))
    assert [event["name"] for event in parsed] == ["kernel_0", "kernel_1", "kernel_2"]


def test_malformed_trace_is_an_explicit_schema_error(tmp_path):
    trace = tmp_path / "trace_view.json"
    trace.write_text('[{"name": "a"},', encoding="utf-8")
    with pytest.raises(ProfilerSchemaError):
        list(iter_trace_events(trace))


def test_sidecar_index_is_wal_bounded_and_rebuilt_on_identity_change(tmp_path):
    trace = tmp_path / "trace_view.json"
    trace.write_text(json.dumps(_events(5)), encoding="utf-8")

    first = build_trace_index(trace)
    assert first["rebuilt"] is True
    assert first["event_count"] == 5
    assert index_is_valid(first["index_path"], trace) is True

    con = sqlite3.connect(first["index_path"])
    try:
        assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert con.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        meta = dict(con.execute("SELECT key, value FROM trace_meta"))
        assert meta["schema_version"] == "mindie-profiler-trace-index/1"
        assert meta["source_size_bytes"] == str(trace.stat().st_size)
    finally:
        con.close()

    rows = query_trace_events(first["index_path"], start_ns=0, end_ns=25_000, limit=10)
    assert [row["name"] for row in rows] == ["kernel_0", "kernel_1", "kernel_2"]

    # A changed source identity (size here) forces a rebuild, not a stale hit.
    trace.write_text(json.dumps(_events(7)), encoding="utf-8")
    assert index_is_valid(first["index_path"], trace) is False
    second = build_trace_index(trace)
    assert second["rebuilt"] is True
    assert second["event_count"] == 7
    assert index_is_valid(second["index_path"], trace) is True
