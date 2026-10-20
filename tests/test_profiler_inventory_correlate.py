"""Tier-1 inventory and Host/Device correlation tests on a real sqlite shape."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mindie_profiler.correlate import correlate_profile
from mindie_profiler.events import ProfilerUsageError, Window
from mindie_profiler.inventory import inventory_profile
from mindie_profiler.__main__ import main


def _sid(con: sqlite3.Connection, cache: dict[str, int], value: str) -> int:
    if value not in cache:
        cache[value] = len(cache) + 1
        con.execute("INSERT INTO STRING_IDS VALUES (?, ?)", (cache[value], value))
    return cache[value]


def build_correlate_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cache: dict[str, int] = {}
    try:
        con.executescript(
            """
            CREATE TABLE META_DATA(name TEXT, value TEXT);
            CREATE TABLE STRING_IDS(id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE TASK(startNs INTEGER, endNs INTEGER, deviceId INTEGER, connectionId INTEGER,
                              globalTaskId INTEGER, globalPid INTEGER, taskType INTEGER, contextId INTEGER,
                              streamId INTEGER, taskId INTEGER, modelId INTEGER, depth INTEGER);
            CREATE TABLE COMPUTE_TASK_INFO(name INTEGER, globalTaskId INTEGER PRIMARY KEY,
                              blockDim INTEGER, mixBlockDim INTEGER, taskType INTEGER, opType INTEGER,
                              inputFormats INTEGER, inputDataTypes INTEGER, inputShapes INTEGER,
                              outputFormats INTEGER, outputDataTypes INTEGER, outputShapes INTEGER,
                              attrInfo INTEGER, opState INTEGER, hf32Eligible INTEGER, waitNs INTEGER);
            CREATE TABLE PYTORCH_API(id INTEGER, name INTEGER, globalTid INTEGER,
                              startNs INTEGER, endNs INTEGER, connectionId INTEGER);
            CREATE TABLE CANN_API(id INTEGER, name INTEGER, globalTid INTEGER,
                              startNs INTEGER, endNs INTEGER, connectionId INTEGER);
            CREATE TABLE CONNECTION_IDS(id INTEGER, connectionId INTEGER);
            CREATE TABLE MSTX_EVENTS(id INTEGER, name INTEGER, globalTid INTEGER,
                              startNs INTEGER, endNs INTEGER);
            """
        )
        con.execute("INSERT INTO META_DATA VALUES ('SCHEMA_VERSION', '1.2.0')")

        def host(table, rowid, name, start, end, connection=None):
            placeholders = "?" if table == "MSTX_EVENTS" else "?,?"
            values = (rowid, _sid(con, cache, name), 100, start, end)
            if table != "MSTX_EVENTS":
                values = values + (connection,)
            con.execute(
                f"INSERT INTO {table} VALUES (?,?,?,?,{placeholders})",
                values,
            )

        host("PYTORCH_API", 1, "model.forward", 0, 90_000)
        host("PYTORCH_API", 2, "layer.mla", 1_000, 80_000)
        host("PYTORCH_API", 3, "ProfilerStep#1", 0, 90_000)
        host("CANN_API", 10, "aclrtLaunchKernel", 2_000, 2_500)
        host("MSTX_EVENTS", 20, "prefill", 0, 50_000)
        host("MSTX_EVENTS", 21, "decode", 50_000, 90_000)
        con.execute("INSERT INTO CONNECTION_IDS VALUES (10, 500)")

        tasks = [
            (1, 3_000, 6_000, 500, 1, 1),
            (2, 7_000, 8_000, 501, 2, 2),
            (1, 20_000, 22_000, 500, 1, 1),
            (3, 30_000, 35_000, 502, 3, 3),
        ]
        for global_id, start, end, connection, task_id, stream in tasks:
            con.execute(
                "INSERT INTO TASK VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (start, end, 0, connection, global_id, 100, 3, 4294967295, stream, task_id, 4294967295, 0),
            )
        for global_id, task_id, name, op_type in (
            (1, 1, "aclnnFlashAttention", "FlashAttention"),
            (2, 2, "aclnnRmsNorm", "RmsNorm"),
            (3, 3, "aclnnTransData", "TransData"),
        ):
            con.execute(
                "INSERT INTO COMPUTE_TASK_INFO VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    _sid(con, cache, name), global_id, 20, 0, 3, _sid(con, cache, op_type),
                    0, 0, _sid(con, cache, '"8,16"'), 0, 0, None, 0, _sid(con, cache, "dynamic"),
                    _sid(con, cache, "NO"), 0,
                ),
            )
        con.commit()
    finally:
        con.close()
    return path


def test_inventory_lists_tables_native_windows_and_capture_span(tmp_path):
    db_path = build_correlate_db(tmp_path / "rank_0" / "ASCEND_PROFILER_OUTPUT" / "ascend_pytorch_profiler_0.db")
    result = inventory_profile(db_path.parent.parent)
    db = result["ranks"][0]["files"][0]
    assert db["core_tables"]["PYTORCH_API"] is True
    assert db["core_tables"]["CANN_API"] is True
    assert db["core_tables"]["CONNECTION_IDS"] is True
    assert db["core_tables"]["MSTX_EVENTS"] is True
    assert [row["step"] for row in db["native_step_windows"]] == [1]
    assert {row["phase"] for row in db["mstx_windows"]} == {"prefill", "decode"}
    assert db["capture_span"]["start_ns"] == 0
    assert db["capture_span"]["end_ns"] == 90_000


def test_correlate_builds_host_stack_dispatch_gap_and_unmatched_ratio(tmp_path):
    db_path = build_correlate_db(tmp_path / "rank_0" / "ASCEND_PROFILER_OUTPUT" / "ascend_pytorch_profiler_0.db")
    result = correlate_profile(
        db_path.parent.parent,
        window=Window(0, 60_000),
        kernel_name="FlashAttention",
        min_gap_ms=0.01,
    )
    assert result["host"]["matched_task_count"] == 2
    assert result["host"]["unmatched_task_count"] == 0
    call = result["correlated_calls"][0]
    assert [item["name"] for item in call["host_stack"]] == ["model.forward", "layer.mla"]
    assert call["cann_api"]["name"] == "aclrtLaunchKernel"
    assert call["dispatch"]["host_dispatch_gap_ms"] == 0.0005
    assert call["association_basis"]["cann"] == "connection_ids"
    assert call["association_basis"]["pytorch"] == "same_global_tid_containment"
    assert result["aclgraph"]["ambiguous_global_task_ids"] == [1]
    assert all(call["ambiguous_global_task_id"] for call in result["correlated_calls"])
    assert len(result["device_gaps"]) == 2

    all_tasks = correlate_profile(db_path.parent.parent, window=Window(0, 60_000))
    assert all_tasks["host"]["matched_task_count"] == 2
    assert all_tasks["host"]["unmatched_task_count"] == 2
    assert all_tasks["host"]["unmatched_ratio"] == 0.5
    assert any(gap["device_uncovered_ms"] == 0.012 for gap in all_tasks["device_gaps"])


def test_inventory_correlate_and_sql_cli_subcommands(tmp_path, capsys):
    db_path = build_correlate_db(tmp_path / "rank_0" / "ASCEND_PROFILER_OUTPUT" / "ascend_pytorch_profiler_0.db")
    root = db_path.parent.parent
    assert main(["inventory", "--root", str(root)]) == 0
    assert '"schema_version": "mindie-profiler-inventory/1"' in capsys.readouterr().out

    assert main(["correlate", "--root", str(root), "--window", "0:60"]) == 0
    assert '"schema_version": "mindie-profiler-correlation/1"' in capsys.readouterr().out

    assert main(["sql", "--db", str(db_path), "--query", "SELECT id, value FROM STRING_IDS LIMIT 2"]) == 0
    assert '"row_count": 2' in capsys.readouterr().out


def test_correlate_requires_explicit_window(tmp_path):
    db_path = build_correlate_db(tmp_path / "rank_0" / "ASCEND_PROFILER_OUTPUT" / "ascend_pytorch_profiler_0.db")
    with pytest.raises(ProfilerUsageError, match="explicit"):
        correlate_profile(db_path.parent.parent)
