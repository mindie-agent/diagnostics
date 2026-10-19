"""Meaningful tests for the compact profiler core.

The DB fixture uses the real torch_npu 1.2.0 column set
(``blockDim``/``mixBlockDim``/``waitNs``) rather than the legacy
``blockNum`` spelling.  The CSV fixture is a real 46-column export header with
the documented trailing tab in ``Start Time(us)``.

Covered semantics: interval union vs event sum, exposed communication
overlap, host/idle gap, seven semantic blocks, explicit step/rank selection,
cross-rank slowest attribution, sha256/schema/query-timing/evidence locators,
unknown-schema failure, row budgets, and missing-data limitations.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from mindie_profiler import (
    PARSER_VERSION,
    ProfilerQueryError,
    ProfilerSchemaError,
    ProfilerUsageError,
    Window,
    analyze_profile,
)
from mindie_profiler.__main__ import main
from mindie_profiler.segment import load_steps_file, parse_step_windows

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "profiler_core"
CSV_ROOT = FIXTURE_ROOT / "rank_0"

RANK0_ID = "dp0_pp0_tp0_dcp0_ep0_rank0"
RANK1_ID = "dp0_pp0_tp0_dcp0_ep0_rank1"
STEP4 = Window(start_ns=0, end_ns=60_000, step=4, label="step_4")
STEP5 = Window(start_ns=100_000, end_ns=160_000, step=5, label="step_5")


def _us(value: float) -> int:
    return int(round(value * 1000.0))


def _event(start, end, name, op_type, core, stream, task, *, shapes='"4,8"', emit_comm=False):
    return {
        "start": _us(start),
        "end": _us(end),
        "name": name,
        "op_type": op_type,
        "core": core,
        "task_type": core,
        "stream": stream,
        "task": task,
        "shapes": shapes,
        "emit_comm": emit_comm,
    }


def rank0_events() -> list[dict]:
    return [
        _event(0, 10, "aclnnFlashAttention", "FlashAttention", "AI_CORE", 1, 7),
        _event(5, 15, "GroupedMatmul", "GroupedMatmul", "AI_CORE", 2, 8),
        _event(20, 30, "AivKernel", "hcom_allReduce_", "COMMUNICATION", 3, 9, shapes="", emit_comm=True),
        _event(25, 28, "AivKernel", "hcom_allGather_", "COMMUNICATION", 4, 10, shapes="", emit_comm=True),
        _event(30, 35, "aclnnRmsNorm", "RmsNorm", "AI_VECTOR_CORE", 1, 11),
        _event(35, 40, "aclnnTransData", "TransData", "AI_CORE", 1, 12),
        _event(40, 45, "aclnnUnknown", "UnknownOp", "AI_CORE", 1, 13, shapes=""),
        _event(100, 110, "aclnnFlashAttention", "FlashAttention", "AI_CORE", 1, 21),
        _event(110, 120, "AivKernel", "hcom_allReduce_", "COMMUNICATION", 3, 22, shapes="", emit_comm=True),
        _event(120, 125, "aclnnRmsNorm", "RmsNorm", "AI_VECTOR_CORE", 1, 23),
    ]


def rank1_events() -> list[dict]:
    return [
        _event(0, 10, "aclnnFlashAttention", "FlashAttention", "AI_CORE", 1, 7),
        _event(5, 15, "GroupedMatmul", "GroupedMatmul", "AI_CORE", 2, 8),
        _event(20, 45, "AivKernel", "hcom_allReduce_", "COMMUNICATION", 3, 9, shapes="", emit_comm=True),
        _event(25, 28, "AivKernel", "hcom_allGather_", "COMMUNICATION", 4, 10, shapes="", emit_comm=True),
        _event(45, 55, "aclnnUnknown", "UnknownOp", "AI_CORE", 1, 13, shapes=""),
        _event(100, 110, "aclnnFlashAttention", "FlashAttention", "AI_CORE", 1, 21),
        _event(110, 135, "AivKernel", "hcom_allReduce_", "COMMUNICATION", 3, 22, shapes="", emit_comm=True),
        _event(135, 145, "aclnnUnknown", "UnknownOp", "AI_CORE", 1, 23, shapes=""),
    ]


def _create_schema(con: sqlite3.Connection) -> None:
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
        CREATE TABLE COMMUNICATION_OP(opName INTEGER, startNs INTEGER, endNs INTEGER,
                          connectionId INTEGER, groupName INTEGER, opId INTEGER, relay INTEGER,
                          retry INTEGER, dataType INTEGER, algType INTEGER, count NUMERIC,
                          opType INTEGER, deviceId INTEGER, waitNs INTEGER, opConnectionId INTEGER);
        CREATE TABLE TASK_PMU_INFO(globalTaskId INTEGER, name INTEGER, value NUMERIC);
        """
    )


def build_db(path: Path, events: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        _create_schema(con)
        con.execute("INSERT INTO META_DATA VALUES ('SCHEMA_VERSION', '1.2.0')")
        cache: dict[str, int] = {}

        def sid(value: str) -> int:
            if value not in cache:
                cache[value] = len(cache) + 1
                con.execute("INSERT INTO STRING_IDS VALUES (?, ?)", (cache[value], value))
            return cache[value]

        for index, event in enumerate(events, start=1):
            task_type = sid(event["task_type"])
            con.execute(
                "INSERT INTO TASK VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event["start"], event["end"], 0, index, index, 100, task_type,
                    4294967295, event["stream"], event["task"], 4294967295, 0,
                ),
            )
            con.execute(
                "INSERT INTO COMPUTE_TASK_INFO VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sid(event["name"]), index, 20, 0, task_type, sid(event["op_type"]),
                    0, 0, sid(event["shapes"]) if event["shapes"] else None,
                    0, 0, None, 0, sid("dynamic"), sid("NO"), 0,
                ),
            )
            if event["emit_comm"]:
                con.execute(
                    "INSERT INTO COMMUNICATION_OP VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        sid("hcom_" + event["op_type"].strip("_") + "__dup"),
                        event["start"], event["end"], index, 0, 0, 0, 0, 0, 0, 1,
                        sid(event["op_type"]), 0, 0, 0,
                    ),
                )
        con.commit()
    finally:
        con.close()
    return path


def build_run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    build_db(
        root / RANK0_ID / "ASCEND_PROFILER_OUTPUT" / "ascend_pytorch_profiler_0.db",
        rank0_events(),
    )
    build_db(
        root / RANK1_ID / "ASCEND_PROFILER_OUTPUT" / "ascend_pytorch_profiler_1.db",
        rank1_events(),
    )
    return root


def _window_row(report: dict, step: int) -> dict:
    rows = [row for row in report["window_accounting"] if row["step"] == step]
    assert rows, f"no window accounting for step {step}"
    return rows[0]


def test_db_and_csv_accounting_match_for_real_schema(tmp_path):
    db_root = build_run(tmp_path)
    db_report = analyze_profile(
        db_root,
        ranks=[RANK0_ID],
        windows=[STEP4, STEP5],
    )
    csv_report = analyze_profile(
        CSV_ROOT,
        windows=[STEP4, STEP5],
    )
    for step in (4, 5):
        db_row = _window_row(db_report, step)
        csv_row = _window_row(csv_report, step)
        for key in (
            "wall_ms", "busy_union_ms", "kernel_sum_ms", "exposed_comm_ms",
            "host_gap_ms", "total_cost_ms", "event_count",
        ):
            assert db_row[key] == csv_row[key], (step, key, db_row[key], csv_row[key])
    assert db_report["source_identity"]["exporter_schema_version"] == "torch_npu_db_v1.2.0"
    assert csv_report["source_identity"]["exporter_schema_version"] == "kernel_details_csv_46col"
    assert csv_report["source_identity"]["sources"][0]["header_sha256"]
    assert csv_report["source_identity"]["sources"][0]["schema_variant"] == "csv_blockDim_wait_pmu"


def test_overlap_union_exposed_comm_and_host_gap(tmp_path):
    report = analyze_profile(build_run(tmp_path), ranks=[RANK0_ID], windows=[STEP4])
    row = _window_row(report, 4)
    assert row["wall_ms"] == 0.045
    assert row["busy_union_ms"] == 0.040
    assert row["kernel_sum_ms"] == 0.061
    assert row["exposed_comm_ms"] == 0.010
    assert row["host_gap_ms"] == 0.005
    assert row["total_cost_ms"] == 0.066
    assert row["comm_event_count"] == 4
    assert row["underfeed_bubble_ratio"] == pytest.approx(0.005 / 0.045, abs=1e-6)


def test_seven_semantic_blocks_and_normalised_shares(tmp_path):
    report = analyze_profile(build_run(tmp_path), ranks=[RANK0_ID], windows=[STEP4])
    blocks = {row["block"]: row for row in report["semantic_block_breakdown"]}
    assert set(blocks) == {
        "Attention", "MoE", "Comm", "Norm", "Copy_Layout", "Host_Gap", "Unclassified",
    }
    assert blocks["Attention"]["busy_union_ms_p50"] == 0.010
    assert blocks["MoE"]["busy_union_ms_p50"] == 0.010
    assert blocks["Comm"]["busy_union_ms_p50"] == 0.010
    assert blocks["Norm"]["busy_union_ms_p50"] == 0.005
    assert blocks["Copy_Layout"]["busy_union_ms_p50"] == 0.005
    assert blocks["Unclassified"]["busy_union_ms_p50"] == 0.005
    assert blocks["Host_Gap"]["busy_union_ms_p50"] == 0.005
    assert blocks["Host_Gap"]["share_of_busy_pct"] is None
    shares = [blocks[name]["share_of_busy_pct"] for name in blocks if name != "Host_Gap"]
    assert sum(shares) == pytest.approx(100.0, abs=1e-4)
    assert blocks["Comm"]["mapping_rule"] == "hccl_communication_op"


def test_cross_rank_slowest_and_dominant_excess_block(tmp_path):
    report = analyze_profile(build_run(tmp_path), windows=[STEP4, STEP5])
    slowest = report["step_accounting"]["slowest_rank"]
    assert slowest["rank_index"] == 1
    assert slowest["rank_id"] == RANK1_ID
    assert slowest["wall_ms_p50"] == 0.050
    assert slowest["dominant_excess_block"] == "Comm"
    assert slowest["dominant_excess_ms"] == pytest.approx(0.015)
    assert report["step_accounting"]["cross_rank"]["rank_count_with_data"] == 2


def test_explicit_steps_produce_p50_and_by_step(tmp_path):
    report = analyze_profile(build_run(tmp_path), windows=[STEP4, STEP5])
    accounting = report["step_accounting"]
    assert accounting["steady_steps"] == [4, 5]
    assert accounting["step_status"] == "explicit"
    assert accounting["wall_ms_p50"] == 0.045
    assert [row["step"] for row in accounting["by_step"]] == [4, 5]
    assert len(accounting["by_rank"]) == 2
    assert report["window_accounting"][0]["wall_ms"] is not None


def test_unlabelled_window_does_not_invent_a_step(tmp_path):
    report = analyze_profile(
        build_run(tmp_path),
        ranks=[RANK0_ID],
        windows=[Window(start_ns=0, end_ns=60_000, step=None, label="request_window")],
    )
    assert report["step_accounting"]["steady_steps"] == []
    assert report["step_accounting"]["step_status"] == "not_provided"
    assert report["step_accounting"]["wall_ms_p50"] is None
    assert _window_row(report, None)["wall_ms"] == 0.045
    assert any(item["code"] == "step_not_provided" for item in report["limitations"])


def test_named_rank_keeps_null_index_in_hotspot_modeling_scope(tmp_path):
    source = build_db(
        tmp_path / "glm45" / "ASCEND_PROFILER_OUTPUT" / "ascend_pytorch_profiler_0.db",
        rank0_events(),
    )
    report = analyze_profile(source.parent.parent, windows=[STEP4])

    assert report["window_accounting"][0]["rank_id"] == "glm45"
    assert report["window_accounting"][0]["rank_index"] is None
    for hotspot in report["top_hotspot_kernels"]:
        assert hotspot["rank"] == "glm45"
        assert hotspot["rank_index"] is None
        assert hotspot["modeling_alignment"]["scope"]["rank_index"] is None


@pytest.mark.parametrize("split_windows", [False, True])
def test_hotspot_locator_sample_is_explicit_and_totals_use_all_events(tmp_path, split_windows):
    events = [
        _event(index * 10, index * 10 + 5, "aclnnRmsNorm", "RmsNorm", "AI_VECTOR_CORE", 1, index)
        for index in range(40)
    ]
    source = build_db(tmp_path / "rank_0" / "ascend_pytorch_profiler_0.db", events)
    windows = (
        [Window(0, 100_000), Window(100_000, 200_000), Window(200_000, 400_000)]
        if split_windows else [Window(0, 400_000)]
    )
    report = analyze_profile(source.parent, windows=windows)
    hotspot = report["top_hotspot_kernels"][0]
    evidence = hotspot["evidence_locator"]

    assert hotspot["call_count"] == 40
    assert hotspot["measured_device_ms_total"] == pytest.approx(0.2)
    assert hotspot["measured_device_ms_p50"] == pytest.approx(0.005)
    with sqlite3.connect(source) as database:
        selected_count = database.execute(f"SELECT COUNT(*) FROM {evidence['filter_sql']}").fetchone()[0]
    assert selected_count == evidence["retained_event_locators"] == 16
    assert evidence["total_contributing_events"] == hotspot["call_count"]
    assert evidence["ids_truncated"] is True
    assert evidence["filter_scope"] == "locator_sample"
    assert evidence["window_bounds_ns"] == [[window.start_ns, window.end_ns] for window in windows]
    assert hotspot["modeling_alignment"]["scope"]["ids_truncated"] is True


def test_step_accounting_excludes_unlabelled_windows(tmp_path):
    report = analyze_profile(
        build_run(tmp_path),
        ranks=[RANK0_ID],
        windows=[STEP4, Window(start_ns=100_000, end_ns=160_000, step=None, label="untagged")],
    )
    assert report["step_accounting"]["wall_ms_p50"] == 0.045
    assert [row["step"] for row in report["step_accounting"]["by_step"]] == [4]
    assert len(report["window_accounting"]) == 2


def test_full_range_is_explicit_and_unlabelled(tmp_path):
    report = analyze_profile(build_run(tmp_path), ranks=[RANK0_ID], full_range=True)
    row = report["window_accounting"][0]
    assert row["step"] is None
    assert row["label"] == "full_range"
    assert row["wall_ms"] == 0.125
    assert report["step_accounting"]["steady_steps"] == []
    assert any(item["code"] == "step_not_provided" for item in report["limitations"])
    csv_report = analyze_profile(CSV_ROOT, full_range=True)
    assert csv_report["window_accounting"][0]["wall_ms"] == 0.125


def test_no_explicit_selection_is_a_usage_error(tmp_path):
    with pytest.raises(ProfilerUsageError, match="explicit"):
        analyze_profile(build_run(tmp_path), ranks=[RANK0_ID])


def test_empty_window_is_null_not_zero(tmp_path):
    report = analyze_profile(
        build_run(tmp_path),
        ranks=[RANK0_ID],
        windows=[Window(start_ns=_us(1000), end_ns=_us(1060), step=99, label="step_99")],
    )
    row = _window_row(report, 99)
    assert row["event_count"] == 0
    assert row["wall_ms"] is None
    assert row["busy_union_ms"] is None
    assert row["kernel_sum_ms"] is None
    assert row["host_gap_ms"] is None
    assert all(value is None for value in row["block_union_ms"].values())
    assert any(item["code"] == "empty_window" for item in report["limitations"])


def test_legacy_block_num_schema_and_unversioned_metadata(tmp_path):
    db_path = tmp_path / "legacy" / "rank_0" / "ASCEND_PROFILER_OUTPUT" / "ascend_pytorch_profiler_0.db"
    db_path.parent.mkdir(parents=True)
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE TASK(startNs INTEGER, endNs INTEGER, deviceId INTEGER, connectionId INTEGER,
                          globalTaskId INTEGER, globalPid INTEGER, taskType INTEGER, contextId INTEGER,
                          streamId INTEGER, taskId INTEGER, modelId INTEGER);
        CREATE TABLE COMPUTE_TASK_INFO(name INTEGER, globalTaskId INTEGER PRIMARY KEY,
                          blockNum INTEGER, mixBlockNum INTEGER, taskType INTEGER, opType INTEGER);
        CREATE TABLE STRING_IDS(id INTEGER PRIMARY KEY, value TEXT);
        """
    )
    for name_id, value in enumerate(
        ("aclnnFlashAttention", "FlashAttention", "AI_CORE", "dynamic", "NO", '"4,8"', "DT_BF16", "ND"),
        start=1,
    ):
        con.execute("INSERT INTO STRING_IDS VALUES (?,?)", (name_id, value))
    con.execute(
        "INSERT INTO TASK VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (0, 10_000, 0, 1, 1, 100, 3, 4294967295, 1, 7, 4294967295),
    )
    con.execute(
        "INSERT INTO COMPUTE_TASK_INFO VALUES (?,?,?,?,?,?)",
        (1, 1, 20, 0, 3, 2),
    )
    con.commit()
    con.close()
    report = analyze_profile(db_path.parent.parent, windows=[Window(start_ns=0, end_ns=20_000, step=1, label="step_1")])
    schema = report["source_identity"]["schema"][0]
    assert schema["schema_version"] == "unversioned"
    assert schema["block_column"] == "blockNum"
    assert schema["schema_variant"] == "vunversioned_blockNum"
    assert any(item["code"] == "schema_unversioned" for item in report["limitations"])
    assert any(item["code"] == "shape_columns_absent" for item in report["limitations"])
    assert _window_row(report, 1)["wall_ms"] == 0.010
    assert report["top_hotspot_kernels"][0]["modeling_alignment"]["unmodeled_reason"].startswith("missing input shape")


def test_unknown_db_schema_fails_explicitly(tmp_path):
    broken = tmp_path / "broken" / "rank_0" / "ASCEND_PROFILER_OUTPUT" / "ascend_pytorch_profiler_0.db"
    broken.parent.mkdir(parents=True)
    con = sqlite3.connect(broken)
    con.executescript(
        """
        CREATE TABLE TASK(startNs INTEGER, deviceId INTEGER, globalTaskId INTEGER);
        CREATE TABLE COMPUTE_TASK_INFO(name INTEGER, globalTaskId INTEGER);
        CREATE TABLE STRING_IDS(id INTEGER, value TEXT);
        """
    )
    con.close()
    with pytest.raises(ProfilerSchemaError, match="endNs"):
        analyze_profile(broken.parent.parent, windows=[STEP4])


def test_unsupported_export_version_fails_explicitly(tmp_path):
    root = build_run(tmp_path)
    db_path = root / RANK0_ID / "ASCEND_PROFILER_OUTPUT" / "ascend_pytorch_profiler_0.db"
    con = sqlite3.connect(db_path)
    con.execute("UPDATE META_DATA SET value='9.9.9' WHERE name='SCHEMA_VERSION'")
    con.commit()
    con.close()
    with pytest.raises(ProfilerSchemaError, match="unsupported exporter SCHEMA_VERSION"):
        analyze_profile(root, ranks=[RANK0_ID], windows=[STEP4])


def test_csv_missing_required_header_fails_explicitly(tmp_path):
    csv_path = tmp_path / "rank_0" / "kernel_details.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("Device_id,Name,Accelerator Core\n0,Foo,AI_CORE\n", encoding="utf-8")
    with pytest.raises(ProfilerSchemaError, match="start_us"):
        analyze_profile(csv_path, windows=[STEP4])


def test_csv_invalid_rows_are_reported_not_counted_as_zero(tmp_path):
    csv_path = tmp_path / "rank_0" / "kernel_details.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text(
        "Name,Start Time(us),Duration(us),Accelerator Core\n"
        "aclnnFlashAttention,0.000,10.000,AI_CORE\n"
        "aclnnBroken,20.000,not-a-number,AI_CORE\n",
        encoding="utf-8",
    )
    report = analyze_profile(csv_path, windows=[STEP4])
    row = _window_row(report, 4)
    assert row["event_count"] == 1
    assert row["wall_ms"] == 0.010
    assert any(item["code"] == "bad_csv_rows" for item in report["limitations"])


def test_row_budget_is_enforced_not_truncated(tmp_path):
    with pytest.raises(ProfilerQueryError, match="row budget"):
        analyze_profile(
            build_run(tmp_path),
            ranks=[RANK0_ID],
            windows=[STEP4],
            max_rows=2,
        )


def test_hash_schema_timing_and_event_locator_present(tmp_path):
    root = build_run(tmp_path)
    report = analyze_profile(root, windows=[STEP4, STEP5])
    identity = report["source_identity"]
    db_path = root / RANK0_ID / "ASCEND_PROFILER_OUTPUT" / "ascend_pytorch_profiler_0.db"
    assert identity["db_files_sha256"][RANK0_ID] == hashlib.sha256(db_path.read_bytes()).hexdigest()
    assert identity["parser_version"] == PARSER_VERSION
    assert report["schema_version"] == "mindie-profile-summary/1"
    assert identity["cache_state"] == "cold"
    assert identity["query_elapsed_ms"] >= 0
    assert any(item["phase"] == "sha256" for item in identity["query_timings"])
    assert any(item["phase"] == "read" for item in identity["query_timings"])
    assert identity["schema"][0]["block_column"] == "blockDim"
    hotspot = report["top_hotspot_kernels"][0]
    locator = hotspot["evidence_locator"]
    assert locator["db_file"].endswith(".db")
    assert locator["table"] == "COMPUTE_TASK_INFO"
    assert locator["task_global_ids"]
    assert "globalTaskId" in locator["filter_sql"]
    assert locator["window_bounds_ns"]
    assert locator["source_sha256"] == identity["db_files_sha256"][hotspot["rank_id"]]
    assert hotspot["modeling_alignment"]["status"] == "unmodeled"
    assert hotspot["modeling_alignment"]["model_predicted_ms"] is None
    assert hotspot["modeling_alignment"]["unmodeled_reason"]


def test_partial_coverage_and_optional_table_limitations(tmp_path):
    root = build_run(tmp_path)
    # Rank 0 has no COMMUNICATION_OP table after deletion: comm compute rows
    # still work, but the absence is reported instead of being hidden.
    db_path = root / RANK0_ID / "ASCEND_PROFILER_OUTPUT" / "ascend_pytorch_profiler_0.db"
    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE COMMUNICATION_OP")
    con.commit()
    con.close()
    report = analyze_profile(root, ranks=[RANK0_ID], windows=[STEP4])
    codes = {item["code"] for item in report["limitations"]}
    assert "optional_table_absent" in codes
    assert "partial_window_coverage" in codes
    assert _window_row(report, 4)["exposed_comm_ms"] == 0.010


def test_rank_resolution_by_index_and_unknown_rank(tmp_path):
    root = build_run(tmp_path)
    report = analyze_profile(root, ranks=["1"], windows=[STEP4])
    assert report["step_accounting"]["rank_scope"] == [RANK1_ID]
    assert any(item["code"] == "single_rank" for item in report["limitations"])
    with pytest.raises(ProfilerUsageError, match="not found"):
        analyze_profile(root, ranks=["rank_99"], windows=[STEP4])


def test_rank_0_token_resolves_numeric_index(tmp_path):
    report = analyze_profile(build_run(tmp_path), ranks=["rank_0"], windows=[STEP4])
    assert report["step_accounting"]["rank_scope"] == [RANK0_ID]


def test_window_units_and_steps_file_are_converted_once(tmp_path):
    assert parse_step_windows(["4=0:60"])[0].end_ns == 60_000
    assert parse_step_windows(["4=0:60"], unit="ms")[0].end_ns == 60_000_000
    steps_file = tmp_path / "steps.json"
    steps_file.write_text(
        json.dumps({"steps": [{"step": 4, "start_us": 0, "end_us": 60}]}),
        encoding="utf-8",
    )
    windows = load_steps_file(steps_file, unit="ms")
    assert len(windows) == 1
    assert windows[0].start_ns == 0
    assert windows[0].end_ns == 60_000


def test_c7_contract_surface(tmp_path):
    report = analyze_profile(build_run(tmp_path), windows=[STEP4])
    assert set(report) >= {
        "schema_version", "source_identity", "scene", "step_accounting",
        "semantic_block_breakdown", "top_hotspot_kernels", "findings",
        "limitations", "artifacts",
    }
    assert report["schema_version"] == "mindie-profile-summary/1"
    identity = report["source_identity"]
    assert {"source_kind", "db_files_sha256", "exporter_schema_version", "parser_version",
            "query_elapsed_ms", "cache_state", "query_timings"} <= set(identity)
    accounting = report["step_accounting"]
    assert {"steady_steps", "wall_ms_p50", "busy_union_ms_p50", "kernel_sum_ms_p50",
            "exposed_comm_ms_p50", "host_gap_ms_p50", "total_cost_ms_p50",
            "underfeed_bubble_ratio", "aicpu_exposed_ratio", "slowest_rank"} <= set(accounting)
    block = report["semantic_block_breakdown"][0]
    assert {"block", "busy_union_ms_p50", "share_of_busy_pct", "mapping_rule"} <= set(block)
    hotspot = report["top_hotspot_kernels"][0]
    assert {"rank", "semantic_block", "kernel_name", "measured_device_ms_p50",
            "modeling_alignment", "evidence_locator"} <= set(hotspot)


def test_cli_writes_atomic_c7_json(tmp_path, capsys):
    root = build_run(tmp_path)
    output = tmp_path / "out" / "c7.json"
    code = main(
        [
            "analyze",
            "--root", str(root),
            "--rank", RANK0_ID,
            "--step-window", "4=0:60",
            "--step-window", "5=100:160",
            "--output", str(output),
        ]
    )
    assert code == 0
    assert output.is_file()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "mindie-profile-summary/1"
    assert report["step_accounting"]["steady_steps"] == [4, 5]
    assert not list(output.parent.glob("*.tmp"))
    summary = json.loads(capsys.readouterr().out)
    assert summary["output"] == str(output)


def test_cli_usage_error_returns_two(tmp_path, capsys):
    code = main(["--root", str(build_run(tmp_path)), "--rank", RANK0_ID])
    assert code == 2
    assert "explicit" in capsys.readouterr().err
