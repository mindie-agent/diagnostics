"""Tier 0/1 tests for the mindie_sim four-layer simulation package.

These tests use real A2/A3/A5 CANN platform captures and explicit K3/A5 Golden
graph definitions.  They do not connect to a live NPU and do not claim that a
board measurement happened on this machine.
"""
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from mindie_profiler import theoretical_model as tm
from mindie_sim import hardware as hw
from mindie_sim import net
from mindie_sim import pipeline as pl
from mindie_sim import subgraph as sg

FIXTURES = Path(__file__).parent / "fixtures" / "mindie_sim"
MIB = 1024 * 1024
KIB = 1024
A5_MICROSTAGES = [
    "kv_gm_l1", "qk_b0", "qk_m0", "qk_b1", "qk_m1", "qk_br", "qk_mr",
    "score_fix", "v1_main", "v1_sum", "p_ub_l1", "pv_a", "pv_b0", "pv_m0",
    "pv_b1", "pv_m1", "pv_fix", "v2",
]
M_STAGES = {"qk_m0", "qk_m1", "qk_mr", "pv_m0", "pv_m1"}


def _a5_config() -> hw.PlatformConfig:
    return hw.parse_platform_ini(FIXTURES / "Ascend950DT_9581.ini")


def _a2_snapshot() -> hw.NpuSmiSnapshot:
    return hw.parse_npu_smi_snapshot(
        (FIXTURES / "npu-smi-a2-info.txt").read_text(encoding="utf-8"),
        common_text=(FIXTURES / "npu-smi-a2-common.txt").read_text(encoding="utf-8"),
        board_text=(FIXTURES / "npu-smi-a2-board.txt").read_text(encoding="utf-8"),
    )


def _a3_snapshot() -> hw.NpuSmiSnapshot:
    return hw.parse_npu_smi_snapshot(
        (FIXTURES / "npu-smi-a3-info.txt").read_text(encoding="utf-8"),
        common_text=(FIXTURES / "npu-smi-a3-common.txt").read_text(encoding="utf-8"),
        board_text=(FIXTURES / "npu-smi-a3-board.txt").read_text(encoding="utf-8"),
    )


def _c8_spec() -> dict:
    stages = [
        {"name": "kv_gm_l1", "pipe": "MTE2"},
        {"name": "qk_b0", "pipe": "MTE1", "transaction_index": 0},
        {"name": "qk_m0", "pipe": "M", "transaction_index": 0},
        {"name": "qk_b1", "pipe": "MTE1", "transaction_index": 1},
        {"name": "qk_m1", "pipe": "M", "transaction_index": 1},
        {"name": "qk_br", "pipe": "MTE1", "transaction_index": 2},
        {"name": "qk_mr", "pipe": "M", "transaction_index": 2},
        {"name": "score_fix", "pipe": "FIX"},
        {"name": "v1_main", "pipe": "V"},
        {"name": "v1_sum", "pipe": "V"},
        {"name": "p_ub_l1", "pipe": "MTE3"},
        {"name": "pv_a", "pipe": "MTE1"},
        {"name": "pv_b0", "pipe": "MTE1", "transaction_index": 3},
        {"name": "pv_m0", "pipe": "M", "transaction_index": 3},
        {"name": "pv_b1", "pipe": "MTE1", "transaction_index": 4},
        {"name": "pv_m1", "pipe": "M", "transaction_index": 4},
        {"name": "pv_fix", "pipe": "FIX"},
        {"name": "v2", "pipe": "V"},
    ]
    edges: list[dict] = []

    def edge(src: str, dst: str, distance: int, kind: str = "semaphore", same_slot: str | None = None) -> None:
        item = {"src": src, "dst": dst, "distance": distance, "kind": kind}
        if same_slot:
            item["same_slot"] = same_slot
        edges.append(item)

    for src, dst in (
        ("kv_gm_l1", "qk_b0"), ("qk_b0", "qk_m0"), ("qk_b1", "qk_m1"), ("qk_br", "qk_mr"),
        ("qk_m0", "score_fix"), ("qk_m1", "score_fix"), ("qk_mr", "score_fix"),
        ("score_fix", "v1_main"), ("v1_main", "v1_sum"), ("v1_sum", "v2"),
        ("v1_main", "p_ub_l1"), ("p_ub_l1", "pv_a"),
        ("kv_gm_l1", "pv_b0"), ("kv_gm_l1", "pv_b1"),
        ("pv_a", "pv_m0"), ("pv_b0", "pv_m0"), ("pv_b1", "pv_m1"),
        ("pv_m0", "pv_fix"), ("pv_m1", "pv_fix"), ("pv_fix", "v2"),
    ):
        edge(src, dst, 0)
    edge("pv_m0", "qk_b0", 3, "reuse", "L0B")
    edge("pv_m1", "qk_b1", 3, "reuse", "L0B")
    edge("qk_m0", "qk_br", 0, "reuse", "L0B")
    edge("qk_m1", "pv_b0", -2, "reuse", "L0B")
    edge("qk_mr", "pv_b1", -2, "reuse", "L0B")
    edge("score_fix", "qk_m0", 1, "l0c_score")
    edge("pv_fix", "pv_m0", 1, "l0c_pv")
    edge("v1_main", "score_fix", 2, "score_ub")
    edge("v2", "pv_fix", 1, "pv_ub")
    edge("pv_b1", "kv_gm_l1", 4, "kv4")
    edge("pv_m1", "pv_a", 2, "l0a_p")
    edge("p_ub_l1", "v1_main", 2, "vecout_ub")
    edge("p_ub_l1", "v2", -1, "mte3_token")
    edge("v2", "v1_main", 3, "p3")
    return {
        "name": "c8-flashmla-pv1",
        "phases": 2,
        "tiles_per_period": 2,
        "stages": stages,
        "edges": edges,
        "transactions_per_loop": 5,
        "pipe_fifo": {
            "MTE2": [("kv_gm_l1", 0)],
            "MTE1": [("qk_b0", 0), ("qk_b1", 0), ("qk_br", 0), ("pv_a", -2), ("pv_b0", -2), ("pv_b1", -2)],
            "M": [("qk_m0", 0), ("qk_m1", 0), ("qk_mr", 0), ("pv_m0", -2), ("pv_m1", -2)],
            "FIX": [("score_fix", 0), ("pv_fix", -2)],
            "V": [("v1_main", 0), ("v1_sum", 0), ("v2", -2)],
            "MTE3": [("p_ub_l1", 0)],
        },
        "relaxation_checks": [
            {"name": "pv_ub", "base_distance": 1, "relaxed_distance": 2},
            {"name": "score_l0c", "base_distance": 1, "relaxed_distance": 2},
        ],
    }


def _c8_durations() -> dict:
    return {
        "durations_ns": {
            stage: (180.224 if stage in M_STAGES else 0.0)
            for stage in A5_MICROSTAGES
        },
        "source": "reference-cube-cycles-442918.5024-at-1280mhz",
    }


def _write_json(tmp_path: Path, name: str, payload: object) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Layer 1: hardware, buffers and thermal bounds
# ---------------------------------------------------------------------------


def test_a5_platform_ini_capacity_and_allocatable_ub():
    config = _a5_config()
    assert config.soc_version == "Ascend950DT_9581"
    assert config.short_soc_version == "Ascend950"
    assert config.npu_arch == "3510"
    assert config.ai_core_cnt == 32
    assert config.cube_core_cnt == 32
    assert config.vector_core_cnt == 64
    assert config.platform_config_hbm_mib == 147456.0
    assert config.l2_size_bytes == 128 * MIB
    assert config.l0_a_kib == 64.0
    assert config.l0_b_kib == 64.0
    assert config.l0_c_kib == 256.0
    assert config.l1_kib == 512.0
    assert config.platform_allocatable_ub_kib == 248.0
    assert config.cube_freq_mhz == 1650.0
    assert hw.parse_platform_ini(FIXTURES / "Ascend950DT_9582.ini").platform_config_hbm_mib == 98304.0


def test_a2_and_a3_platform_ini_real_captures():
    a2 = hw.parse_platform_ini(FIXTURES / "Ascend910B4.ini")
    assert a2.short_soc_version == "Ascend910B"
    assert a2.ai_core_cnt == 20
    assert a2.platform_config_hbm_mib == 32768.0
    assert a2.l2_size_bytes == 96 * MIB
    assert a2.l0_c_kib == 128.0
    assert a2.platform_allocatable_ub_kib == 192.0
    assert a2.cube_freq_mhz == 1500.0

    a3 = hw.parse_platform_ini(FIXTURES / "Ascend910_9391.ini")
    assert a3.short_soc_version == "Ascend910_93"
    assert a3.npu_arch == "2201"
    assert a3.ai_core_cnt == 24
    assert a3.platform_config_hbm_mib == 65536.0
    assert a3.l2_size_bytes == 192 * MIB
    assert a3.l0_c_kib == 128.0
    assert a3.platform_allocatable_ub_kib == 192.0
    assert a3.cube_freq_mhz == 1850.0


def test_npu_smi_parsers_real_a2_a3_output():
    a2 = _a2_snapshot()
    assert len(a2.devices) == 8
    assert a2.chip_count == 1
    assert a2.dual_die is False
    assert a2.phy_ids == ()
    assert a2.hbm_totals_mib == (32768,)
    assert [chip.aicore_count for chip in a2.chips] == [20]
    assert a2.chips[0].aicore_freq_mhz == 1650.0
    assert a2.chips[0].aicore_current_freq_mhz == 800.0
    assert a2.board["Product Name"] == "IT21HMDC_Bin6x"

    a3 = _a3_snapshot()
    assert len(a3.devices) == 16
    assert a3.chip_count == 2
    assert a3.dual_die is True
    assert a3.phy_ids == tuple(range(16))
    assert a3.hbm_totals_mib == (65536,)
    assert [chip.aicore_count for chip in a3.chips] == [20, 20]
    assert a3.board["Product Name"] == "IT22HMDA_4_S"


def test_cross_check_never_overwrites_metadata_with_runtime_capacity():
    config = _a5_config()
    result = hw.cross_check_capacities(
        config,
        verified_runtime_hbm_mib=98304,
        simulator_ub_kib=256,
        sustained_mhz_range=(1250, 1300),
    )
    assert result["platform_config_hbm_mib"] == 147456.0
    assert result["verified_runtime_hbm_mib"] == 98304.0
    assert result["hbm_discrepancy_mib"] == 49152.0
    assert result["platform_allocatable_ub_kib"] == 248.0
    assert result["simulator_ub_kib"] == 256.0
    assert result["allocatable_ub_kib"] == 248.0
    assert result["ub_discrepancy_kib"] == 8.0
    assert "platform_config_hbm_mib_differs_from_runtime" in result["flags"]
    assert "simulator_ub_kib_differs_from_allocatable" in result["flags"]
    assert result["capacity_authority"] == "verified_runtime"

    a2_snapshot = _a2_snapshot()
    a2_result = hw.cross_check_capacities(config, a2_snapshot, simulator_ub_kib=None)
    assert a2_result["platform_config_hbm_mib"] == 147456.0
    assert a2_result["verified_runtime_hbm_mib"] == 32768.0
    assert "runtime_aicore_count_differs_from_platform_config" in a2_result["flags"]


def test_buffer_alignment_ping_pong_overflow_and_overlap():
    capacities = {"L1": 512, "L0A": 64, "L0B": 64, "L0C": 256, "UB": 248}
    plan = hw.validate_buffer_allocations(
        [
            {"name": "L1_A", "buffer": "L1", "offset_kib": 0, "size_kib": 128},
            {"name": "L1_B", "buffer": "L1", "offset_kib": 128, "size_kib": 128},
            {"name": "L1_P", "buffer": "L1", "offset_kib": 256, "size_kib": 8},
            {"name": "UB_fixed", "buffer": "UB", "offset_kib": 0, "size_kib": 192},
        ],
        capacities,
        alignment_bytes=256,
        ping_pong=[{"buffer": "L1", "slot_size_kib": 128, "slots": 2}],
    )
    assert plan["ok"] is True
    assert plan["ping_pong"][0]["fits"] is True
    assert plan["ping_pong"][0]["total_bytes"] == 256 * KIB

    misaligned = hw.validate_buffer_allocations(
        [{"name": "odd", "buffer": "UB", "offset_bytes": 100, "size_bytes": 1000}],
        capacities,
        alignment_bytes=256,
    )
    record = misaligned["allocations"][0]
    assert record["aligned_offset_bytes"] == 256
    assert record["aligned_size_bytes"] == 1024
    assert record["waste_bytes"] == 180
    assert any(item["code"] == "buffer_offset_misaligned" for item in misaligned["diagnostics"])

    overflow = hw.validate_buffer_allocations(
        [{"name": "UB_too_large", "buffer": "UB", "size_kib": 250}],
        capacities,
    )
    assert overflow["ok"] is False
    assert overflow["allocations"][0]["overflow_bytes"] == 2 * KIB

    overlap = hw.validate_buffer_allocations(
        [
            {"name": "A", "buffer": "UB", "offset_kib": 0, "size_kib": 100},
            {"name": "B", "buffer": "UB", "offset_kib": 64, "size_kib": 100},
        ],
        capacities,
    )
    assert overlap["ok"] is False
    assert overlap["overlaps"][0]["overlap_bytes"] == 36 * KIB


def test_mlapo_l1b_mx_scale_offset_collision_and_relocation():
    base = hw.check_mx_scale_layout(128, scale_offset_kib=64)
    assert base["status"] == "ok"
    assert base["overlap_bytes"] == 0
    assert base["weight_tile_kib"] == 64.0

    collision = hw.check_mx_scale_layout(192, scale_offset_kib=64)
    assert collision["status"] == "violation"
    assert collision["overlap_bytes"] == 32 * KIB
    assert collision["recommended_scale_offset_kib"] == 96.0
    assert collision["diagnostic"] == "mx_scale_overwritten_by_weight_tile"

    relocated = hw.check_mx_scale_layout(192, scale_offset_kib=96)
    assert relocated["status"] == "ok"
    assert relocated["overlap_bytes"] == 0
    assert relocated["weight_tile_kib"] == 96.0
    assert relocated["allocation_check"]["ok"] is True


def test_cube_reference_cycles_and_sustained_reference_lower_bounds():
    cycles = hw.cube_reference_cycles(206158430208, 12884901888)
    assert cycles == pytest.approx(442918.5024, rel=1e-12)
    expected = {
        1250.0: 354.33480192,
        1280.0: 346.03008,
        1300.0: 340.7065403076923,
        1650.0: 268.435456,
    }
    for mhz, value in expected.items():
        assert hw.reference_time_us(cycles, mhz) == pytest.approx(value, rel=1e-12)

    thermal = hw.thermal_bounds()
    assert thermal["sustained_mhz_range"] == [1250.0, 1300.0]
    assert thermal["short_burst_mhz_range"] == [1640.0, 1650.0]
    required = thermal["required_mhz_for_target"]
    assert required["counted_mac_mhz"] == pytest.approx(1362.80375, rel=1e-12)
    assert required["vector_mhz"] == pytest.approx(1302.934017142857, rel=1e-12)
    assert required["wall_mhz"] == pytest.approx(1618.1260714285713, rel=1e-12)
    assert thermal["sustained_reference_us"][0] > 350.0
    assert thermal["short_burst_reference_us"][0] < 300.0


def test_gap_elimination_sensitivity_matrix_is_reproducible():
    result = hw.sensitivity_matrix(566344.125, 476981.3125)
    assert result["gap_cycles"] == pytest.approx(89362.8125)
    assert result["r_values"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    by_r = {row["r"]: row["time_us"] for row in result["table"]}
    assert by_r[0.0]["1250"] == pytest.approx(453.0753, rel=1e-12)
    assert by_r[0.0]["1280"] == pytest.approx(442.45634765625, rel=1e-12)
    assert by_r[0.25]["1280"] == pytest.approx(425.00267333984374, rel=1e-12)
    assert by_r[0.5]["1280"] == pytest.approx(407.5489990234375, rel=1e-12)
    assert by_r[0.75]["1280"] == pytest.approx(390.09532470703124, rel=1e-12)
    assert by_r[1.0]["1280"] == pytest.approx(372.641650390625, rel=1e-12)
    assert by_r[0.75]["1300"] == pytest.approx(384.0938581730769, rel=1e-12)


def test_link_reference_bounds_use_per_aic_traffic_and_do_not_sum_links():
    traffic = {
        "gm_l2_to_l1": 30 * MIB,
        "l1_to_l0b": 54 * MIB,
        "l1_to_l0a": 3 * MIB + 120 * KIB,
        "l0c_to_ub0": 30 * MIB,
        "l0c_to_ub1": 30 * MIB,
        "l0c_to_fixp": 60 * MIB,
    }
    result = hw.link_reference_bounds(traffic, mhz=1280.0)
    by_link: dict[str, list[float]] = {}
    for item in result["links"]:
        by_link.setdefault(item["link"], []).append(item["minimum_us"])
    assert by_link["gm_l2_to_l1"] == [pytest.approx(192.0)]
    assert by_link["l1_to_l0b"] == [pytest.approx(172.8)]
    assert by_link["l1_to_l0a"] == [pytest.approx(9.975)]
    assert by_link["l0c_to_ub"] == [pytest.approx(192.0), pytest.approx(192.0)]
    assert by_link["l0c_to_fixp"] == [pytest.approx(96.0)]
    assert result["max_us"] == pytest.approx(192.0)
    assert result["sum_us_not_additive"] is True


def test_cannsim_link_constant_parser_keeps_simulator_provenance():
    parsed = hw.parse_cannsim_link_ceilings(FIXTURES / "cannsim_link_constants.py")
    assert parsed["ceilings_bpc"]["gm_l2_to_l1"] == 128
    assert parsed["ceilings_bpc"]["l1_to_l0a"] == 256
    assert parsed["ceilings_bpc"]["l0c_to_fixp"] == 512
    assert parsed["ceilings_bpc"]["cube_to_l0c"] == 1024
    assert parsed["verified_bandwidth"] is False
    assert "not measured bandwidth" in parsed["note"]


# ---------------------------------------------------------------------------
# Layer 2: periodic microstage graph and Karp solver
# ---------------------------------------------------------------------------


def test_c8_graph_expands_18_microstages_to_36_phase_nodes():
    graph = pl.parse_pipeline_graph(_c8_spec())
    expanded = pl.expand_pipeline_graph(graph)
    assert len(graph.stages) == 18
    assert expanded.node_count == 36
    assert graph.phases == 2
    assert graph.tiles_per_period == 2
    assert {stage.name for stage in graph.stages} == set(A5_MICROSTAGES)
    check = pl.pipeline_self_check(graph)
    assert check["status"] == "ok"
    assert check["checks"]["distance_expansion_ok"] is True
    assert check["checks"]["slot_identity"]["checks"] == 10
    assert check["checks"]["slot_identity"]["failures"] == []
    assert check["checks"]["finite_schedule"] is True
    relaxation = check["checks"]["dimensionless_relaxation"]["checks"]
    assert [item["name"] for item in relaxation] == ["pv_ub", "score_l0c"]
    assert all(item["strictly_relaxes"] for item in relaxation)


def test_c8_karp_solves_346us_and_identifies_m_pipe_critical_cycle():
    result = pl.solve_periodic_pipeline(
        _c8_spec(),
        _c8_durations(),
        target_us=350.0,
        total_tiles=384,
        reference_mhz=1280.0,
    )
    assert result["status"] == "modeled"
    assert result["minimum_period_ns"] == pytest.approx(1802.24, rel=1e-12)
    assert result["minimum_tile_ii_ns"] == pytest.approx(901.12, rel=1e-12)
    assert result["total_time_us"] == pytest.approx(346.03008, rel=1e-12)
    assert result["bottleneck_pipes"] == ["M"]
    assert result["critical_cycles"][0]["distance"] == 1
    assert set(result["critical_cycles"][0]["unique_stages"]) == M_STAGES
    assert result["critical_cycles"][0]["cycle_mean_ns"] == pytest.approx(1802.24, rel=1e-12)
    target = result["target"]
    assert target["target_tile_ii_ns"] == pytest.approx(350000.0 / 384.0, rel=1e-12)
    assert target["allowed_service_cycles_per_tile_at_mhz"] == pytest.approx(1166.6666666666667, rel=1e-12)
    assert target["feasible"] is True
    assert target["headroom_ns_per_tile"] == pytest.approx(10.338333333333594, rel=1e-9)


def test_c8_pv1_pv2_score2_share_the_free_vector_m_pipe_bound():
    durations = _c8_durations()
    periods = {}
    for name, kind, distance in (
        ("pv1", "pv_ub", 1),
        ("pv2", "pv_ub", 2),
        ("score2", "l0c_score", 2),
    ):
        spec = copy.deepcopy(_c8_spec())
        if name != "pv1":
            for edge in spec["edges"]:
                if edge["kind"] == kind:
                    edge["distance"] = distance
        graph = pl.parse_pipeline_graph(spec)
        assert pl.expand_pipeline_graph(graph).node_count == 36
        result = pl.solve_periodic_pipeline(graph, durations)
        periods[name] = result["minimum_period_ns"]
        assert result["bottleneck_pipes"] == ["M"]
    assert periods == pytest.approx({name: 1802.24 for name in periods}, rel=1e-12)


def test_phase_qualified_duration_keys_are_accepted_and_override_base():
    phase_durations = {}
    for phase in (0, 1):
        for stage in A5_MICROSTAGES:
            phase_durations[f"p{phase}.{stage}"] = 180.224 if stage in M_STAGES else 0.0
    result = pl.solve_periodic_pipeline(_c8_spec(), {"durations_ns": phase_durations})
    assert result["status"] == "modeled"
    assert result["minimum_period_ns"] == pytest.approx(1802.24, rel=1e-12)

    mixed = _c8_durations()
    mixed["durations_ns"]["p1.qk_m0"] = 0.0
    mixed_result = pl.solve_periodic_pipeline(_c8_spec(), mixed)
    assert mixed_result["status"] == "modeled"
    assert mixed_result["minimum_period_ns"] < 1802.24


def test_uncalibrated_durations_return_only_topology_and_symbolic_cycles():
    spec = _c8_spec()
    graph = pl.parse_pipeline_graph(spec)
    nulls = {stage: None for stage in A5_MICROSTAGES}
    for payload in (None, nulls):
        result = pl.solve_periodic_pipeline(graph, payload)
        assert result["status"] == "uncalibrated_durations"
        assert result["minimum_period_ns"] is None
        assert result["minimum_tile_ii_ns"] is None
        assert result["total_time_us"] is None
        assert result["fabricated_prediction"] is False
        assert result["self_check"]["status"] == "ok"
        assert result["symbolic_cycles"]
        m_cycle = next(item for item in result["symbolic_cycles"] if item["kind"] == "pipe_fifo:M")
        assert m_cycle["expression"].endswith("/1")
        assert "qk_m0" in m_cycle["expression"]
        assert result["critical_cycles"] == []

    missing = dict(nulls)
    missing.pop("score_fix")
    missing_result = pl.solve_periodic_pipeline(graph, missing)
    assert missing_result["status"] == "uncalibrated_durations"
    assert "score_fix" in missing_result["durations"]["uncalibrated_stages"]
    assert missing_result["minimum_period_ns"] is None

    percentage = {
        "durations_ns": {stage: 37.6 for stage in A5_MICROSTAGES},
        "unit": "PipeUtilization %",
        "source": "aggregate PMU",
    }
    percentage_result = pl.solve_periodic_pipeline(graph, percentage)
    assert percentage_result["status"] == "uncalibrated_durations"
    assert "raw_pipe_utilization_is_not_service_duration" in percentage_result["durations"]["reasons"]
    assert percentage_result["minimum_period_ns"] is None

    invalid = {stage: 1.0 for stage in A5_MICROSTAGES}
    invalid["qk_m0"] = math.nan
    invalid_result = pl.solve_periodic_pipeline(graph, invalid)
    assert invalid_result["status"] == "uncalibrated_durations"
    assert "qk_m0" in invalid_result["durations"]["uncalibrated_stages"]
    assert invalid_result["minimum_period_ns"] is None

    zero = {stage: 0.0 for stage in A5_MICROSTAGES}
    zero_result = pl.solve_periodic_pipeline(graph, zero)
    assert zero_result["status"] == "modeled"
    assert zero_result["minimum_period_ns"] == 0.0


def test_pipeline_self_check_rejects_nonpositive_fifo_closure():
    spec = _c8_spec()
    spec["pipe_fifo"]["MTE2"] = [("kv_gm_l1", 0), ("qk_b0", 1)]
    check = pl.pipeline_self_check(spec)
    assert check["status"] == "violations"
    assert "fifo_closure_nonpositive:MTE2" in check["failures"]


def test_pipeline_compare_variant_keeps_variant_result_separate():
    base = _c8_spec()
    variant = copy.deepcopy(base)
    for edge in variant["edges"]:
        if edge["kind"] == "pv_ub":
            edge["distance"] = 2
    comparison = pl.compare_pipeline_variants(base, variant, _c8_durations())
    assert comparison["status"] == "modeled"
    assert comparison["base"]["minimum_period_ns"] == pytest.approx(1802.24, rel=1e-12)
    assert comparison["variant"]["minimum_period_ns"] == pytest.approx(1802.24, rel=1e-12)
    assert comparison["comparison"]["delta_ns"] == pytest.approx(0.0, abs=1e-9)
    assert comparison["comparison"]["relaxed"] is True


# ---------------------------------------------------------------------------
# Layer 3: subgraph fusion and critical-path overlap
# ---------------------------------------------------------------------------


def _fusion_dag() -> dict:
    return {
        "name": "mlapo-dcp8",
        "nodes": [
            {"id": "mm3", "kind": "matmul", "duration_us": 10.0},
            {"id": "cast", "kind": "aiv_cast", "duration_us": 5.0},
            {"id": "pack", "kind": "pack", "duration_us": 22.2},
            {"id": "comm", "kind": "device_comm", "duration_us": 34.0},
            {"id": "current", "kind": "attention", "duration_us": 390.0},
            {"id": "gate", "kind": "matmul", "duration_us": 19.4},
        ],
        "edges": [
            {
                "src": "mm3",
                "dst": "cast",
                "tensor": {"bytes": 4.5 * MIB, "dtype": "float32", "layout": "ND", "scale": "per_tensor"},
            },
            {"src": "cast", "dst": "pack"},
            {"src": "pack", "dst": "current"},
        ],
        "capacities_kib": {"L1": 512, "L0A": 64, "L0B": 64, "L0C": 256, "UB": 248},
        "l2_cache_mib": 128,
    }


def _fusion_candidate() -> dict:
    return {
        "name": "MM3 FP32 -> GM -> AIV Cast BF16",
        "producer": "mm3",
        "consumer": "cast",
        "remove_nodes": ["cast"],
        "remove_edges": [{"src": "mm3", "dst": "cast"}],
        "live_buffers": [{"name": "UB scratch", "buffer": "UB", "size_kib": 192}],
        "overlap": {"pack": "pack", "comm": "comm", "current": "current", "gate": "gate"},
    }


def test_subgraph_removable_gm_bytes_and_l2_residency_caveat():
    result = sg.analyze_subgraph_fusion(_fusion_dag(), _fusion_candidate())
    assert result["status"] == "modeled"
    assert result["removable_gm_bytes"] == pytest.approx(9 * MIB)
    assert result["removable_gm_write_bytes"] == pytest.approx(4.5 * MIB)
    assert result["removable_gm_read_bytes"] == pytest.approx(4.5 * MIB)
    assert result["removed_kernel_launches"] == 1
    caveat = result["l2_cache_caveat"]
    assert caveat["hbm_bytes_lower_bound"] is None
    assert caveat["hbm_time_lower_bound_us"] is None
    assert "not HBM traffic" in caveat["note"]
    assert result["latency"]["critical_path_saving_us"] == pytest.approx(5.0)
    assert result["latency"]["min_saving_us"] == 0.0
    assert result["latency"]["max_saving_us"] == pytest.approx(5.0)


def test_subgraph_hard_fusion_gates():
    collective = _fusion_dag()
    collective["nodes"][0]["collective"] = True
    result = sg.analyze_subgraph_fusion(collective, _fusion_candidate())
    assert result["status"] == "infeasible"
    assert "collective_barrier_requires_global_sync" in result["feasibility"]["failures"]

    overflow_candidate = _fusion_candidate()
    overflow_candidate["live_buffers"] = [{"name": "UB scratch", "buffer": "UB", "size_kib": 250}]
    overflow = sg.analyze_subgraph_fusion(_fusion_dag(), overflow_candidate)
    assert overflow["status"] == "infeasible"
    assert "live_buffer_limit" in overflow["feasibility"]["failures"]

    dcp_dag = _fusion_dag()
    dcp_dag["nodes"][1] = {
        "id": "cast",
        "op": "dcp_merge",
        "normalization": "full_reduction_lse_fp32",
        "duration_us": 5.0,
    }
    dcp_candidate = _fusion_candidate()
    dcp_candidate["elide_normalization"] = True
    dcp = sg.analyze_subgraph_fusion(dcp_dag, dcp_candidate)
    assert dcp["status"] == "infeasible"
    assert "non_elidable_full_reduction_normalization" in dcp["feasibility"]["failures"]

    route_dag = _fusion_dag()
    route_dag["nodes"][1]["route_fallback"] = "ZeroExpertRouter"
    route = sg.analyze_subgraph_fusion(route_dag, _fusion_candidate())
    assert "route_fallback_contract" in route["feasibility"]["failures"]

    contract_candidate = _fusion_candidate()
    contract_candidate["consumer_input_contract"] = {"dtype": "bfloat16"}
    contract = sg.analyze_subgraph_fusion(_fusion_dag(), contract_candidate)
    assert "layout_scale_contract_mismatch" in contract["feasibility"]["failures"]


def test_subgraph_overlap_uses_max_p_plus_c_q_plus_g():
    candidate = _fusion_candidate()
    candidate["overlap"] = dict(candidate["overlap"], pack_us=400.0)
    result = sg.analyze_subgraph_fusion(_fusion_dag(), candidate)
    overlap = result["overlap"]
    assert overlap["status"] == "modeled"
    assert overlap["pack_plus_comm_us"] == pytest.approx(434.0)
    assert overlap["current_plus_gate_us"] == pytest.approx(409.4)
    assert overlap["tail_before_us"] == pytest.approx(434.0)
    assert overlap["tail_after_eliding_pack_us"] == pytest.approx(409.4)
    assert overlap["min_saving_us"] == 0.0
    assert overlap["max_saving_us"] == pytest.approx(24.6)
    assert overlap["critical_branch_before"] == "pack_plus_comm"

    hidden = sg.analyze_subgraph_fusion(_fusion_dag(), _fusion_candidate())
    assert hidden["overlap"]["tail_before_us"] == pytest.approx(409.4)
    assert hidden["overlap"]["max_saving_us"] == 0.0
    assert hidden["overlap"]["critical_branch_before"] == "current_plus_gate"


def test_subgraph_rejects_aicpu_hccl_launch_as_device_communication():
    dag = _fusion_dag()
    dag["nodes"][3]["duration_kind"] = "aicpu_hccl_launch"
    result = sg.analyze_subgraph_fusion(dag, _fusion_candidate())
    assert result["status"] == "invalid_communication_duration"
    assert result["overlap"]["status"] == "invalid_communication_duration"
    assert result["overlap"]["reason"] == "aicpu_hccl_launch_is_not_device_communication"
    assert result["overlap"]["min_saving_us"] is None
    assert result["overlap"]["max_saving_us"] is None


def test_subgraph_missing_durations_keeps_latency_unmodeled():
    dag = _fusion_dag()
    dag["nodes"][1]["duration_us"] = None
    result = sg.analyze_subgraph_fusion(dag, _fusion_candidate())
    assert result["status"] == "unmodeled_latency"
    assert result["latency"]["critical_path_saving_us"] is None
    assert result["latency"]["max_saving_us"] is None
    assert result["removable_gm_bytes"] == pytest.approx(9 * MIB)


# ---------------------------------------------------------------------------
# Layer 4: three-value bridge and parallel decomposition
# ---------------------------------------------------------------------------


class _FixtureProvider:
    def __init__(self) -> None:
        self.identity = tm.ProviderIdentity("fixture-provider", "9.9.9")

    def available(self) -> bool:
        return True

    def predict(self, request: tm.ModelingRequest) -> tm.ProviderPrediction:
        return tm.ProviderPrediction(
            predicted_ms=0.002,
            model_assumptions="fixture assumptions for the bridge boundary",
            input_signature=request.input_signature,
            evidence=("source:fixture",),
        )


def _net_spec() -> dict:
    return {
        "name": "k3-dp4-tp8-ep64-pp2-dcp8",
        "kernel_inputs": {
            "shape": [2, 4, 8],
            "dtype": "bf16",
            "layout": "row_major",
            "hardware": {
                "name": "fixture-device",
                "peak_flops_per_s": 1e12,
                "memory_bandwidth_bytes_per_s": 1e9,
            },
            "op_kind": "matmul",
        },
        "parallel": {"dp": 4, "tp": 8, "ep": 64, "pp": 2, "dcp": 8},
        "step_components_ms": {
            "compute": 8.0,
            "attention": 4.0,
            "moe": 6.0,
            "communication": 2.0,
            "overhead": 1.0,
        },
        "pp_bubble_ms": 0.5,
        "dspark": {"acceptance_length": 3.0},
        "prefix_cache": {"hit_rate": 0.8, "reusable_fraction": 0.9, "prefill_ms": 20.0},
    }


def test_net_preserves_three_values_and_decomposes_parallel_step():
    result = net.net_report(_net_spec(), provider=_FixtureProvider(), measured_device_ms=2.578125)
    assert result["status"] == "modeled"
    assert result["ideal_hardware_lower_bound_ms"] == pytest.approx(0.000112)
    assert result["model_predicted_ms"] == pytest.approx(0.002)
    assert result["measured_device_ms"] == pytest.approx(2.578125)
    assert result["model_provider"] == "fixture-provider@9.9.9"
    step = result["step_decomposition"]
    assert step["status"] == "modeled"
    assert step["parallel"] == {"dp": 4, "tp": 8, "ep": 64, "pp": 2, "dcp": 8}
    assert step["step_ms"] == pytest.approx(2.578125)
    assert step["dspark"]["tokens_per_target_step"] == 4.0
    assert step["dspark"]["tpot_ms"] == pytest.approx(0.64453125)
    assert step["prefix_cache"]["saved_prefill_ms"] == pytest.approx(14.4)
    assert step["prefix_cache"]["ttft_ms"] == pytest.approx(5.6)
    separation = result["three_value_separation"]
    assert separation["values_are_distinct_fields"] is True
    assert "measured" in separation["measured_source"]


def test_net_keeps_prediction_when_only_the_ideal_workload_is_missing():
    spec = _net_spec()
    spec["kernel_inputs"].pop("op_kind")
    result = net.net_report(spec, provider=_FixtureProvider())
    assert result["status"] == "partial"
    assert result["ideal_hardware_lower_bound_ms"] is None
    assert result["model_predicted_ms"] == pytest.approx(0.002)
    assert "missing_workload" in result["ideal_unmodeled_reason"]
    assert result["prediction_unmodeled_reason"] is None
    assert result["step_decomposition"]["status"] == "modeled"


def test_net_without_provider_keeps_prediction_none_and_measured_separate():
    result = net.net_report(_net_spec(), measured_device_ms=2.0)
    assert result["status"] == "partial"
    assert result["ideal_hardware_lower_bound_ms"] == pytest.approx(0.000112)
    assert result["model_predicted_ms"] is None
    assert result["measured_device_ms"] == pytest.approx(2.0)
    assert result["unmodeled_reason"] == "msmodeling_provider_not_configured"
    assert result["step_decomposition"]["status"] == "modeled"


def test_net_missing_step_components_and_unknown_shard_divisor_are_unmodeled():
    missing = _net_spec()
    missing.pop("step_components_ms")
    result = net.net_report(missing, provider=_FixtureProvider())
    assert result["step_decomposition"]["status"] == "unmodeled"
    assert result["step_decomposition"]["unmodeled_reason"] == "missing_step_components_ms"
    assert result["step_decomposition"]["step_ms"] is None

    unknown = _net_spec()
    unknown["shard_divisors"] = {"compute": "mp"}
    unknown_result = net.net_report(unknown, provider=_FixtureProvider())
    assert unknown_result["step_decomposition"]["status"] == "unmodeled"
    assert unknown_result["step_decomposition"]["unmodeled_reason"] == "shard_divisor_unknown:mp"


def test_tensor_cast_provider_bridge_without_source_is_explicitly_unavailable():
    provider, status = net.provider_bridge(None, None)
    assert provider is None
    assert status["status"] == "unavailable"
    assert status["reason"] == "msmodeling_not_installed"
    missing_provider, missing_status = net.provider_bridge("/does/not/exist", "device-x")
    assert missing_provider is None
    assert missing_status["status"] == "unavailable"
    assert missing_status["reason"] in {"source_missing", "source_git_commit_failed", "source_version_unreadable"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_pipeline_self_check_and_hardware(tmp_path: Path, capsys):
    from mindie_sim.__main__ import main

    graph_path = _write_json(tmp_path, "graph.json", _c8_spec())
    assert main(["pipeline", "--graph", str(graph_path), "--self-check"]) == 0
    uncalibrated = json.loads(capsys.readouterr().out)
    assert uncalibrated["status"] == "uncalibrated_durations"
    assert uncalibrated["self_check"]["status"] == "ok"
    assert uncalibrated["minimum_period_ns"] is None

    durations_path = _write_json(tmp_path, "durations.json", _c8_durations())
    assert main([
        "pipeline", "--graph", str(graph_path), "--durations", str(durations_path),
        "--target-us", "350", "--total-tiles", "384", "--reference-mhz", "1280",
    ]) == 0
    modeled = json.loads(capsys.readouterr().out)
    assert modeled["status"] == "modeled"
    assert modeled["minimum_tile_ii_ns"] == pytest.approx(901.12, rel=1e-12)

    assert main([
        "hardware",
        "--ini", str(FIXTURES / "Ascend950DT_9581.ini"),
        "--npu-smi-file", str(FIXTURES / "npu-smi-a3-info.txt"),
        "--verified-runtime-hbm-mib", "98304",
        "--simulator-ub-kib", "256",
    ]) == 0
    hardware = json.loads(capsys.readouterr().out)
    assert hardware["capacity_cross_check"]["platform_config_hbm_mib"] == 147456.0
    assert hardware["capacity_cross_check"]["verified_runtime_hbm_mib"] == 65536.0


def test_cli_subgraph_and_net_dispatch(tmp_path: Path, capsys):
    from mindie_sim.__main__ import main

    dag_path = _write_json(tmp_path, "dag.json", _fusion_dag())
    candidate_path = _write_json(tmp_path, "candidate.json", _fusion_candidate())
    assert main(["subgraph", "--dag", str(dag_path), "--candidate-fusion", str(candidate_path)]) == 0
    subgraph_result = json.loads(capsys.readouterr().out)
    assert subgraph_result["status"] == "modeled"
    assert subgraph_result["removable_gm_bytes"] == pytest.approx(9 * MIB)

    model_path = _write_json(tmp_path, "model.json", _net_spec())
    assert main(["net", "--model-spec", str(model_path)]) == 0
    net_result = json.loads(capsys.readouterr().out)
    assert net_result["status"] == "partial"
    assert net_result["model_predicted_ms"] is None
    assert net_result["step_decomposition"]["step_ms"] == pytest.approx(2.578125)


def test_module_entrypoint_starts_without_live_npu():
    completed = subprocess.run(
        [sys.executable, "-m", "mindie_sim", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0
    assert "hardware" in completed.stdout
    assert "pipeline" in completed.stdout
    assert "subgraph" in completed.stdout
    assert "net" in completed.stdout


def test_multi_sku_device_capability_profile_and_ambiguous_guard(tmp_path: Path, capsys):
    from mindie_sim.__main__ import main

    # 1. Ambiguous family-only guard: passing "A2" / "A3" / "A5" without exact SKU, INI, or probe
    ambiguous = hw.hardware_report(
        None,
        soc="A2",
        buffer_plan={"allocations": [{"name": "ub_ok", "buffer": "UB", "size_kib": 160}]},
        workload={"flops_bf16": 1e12, "target_us": 100.0},
    )
    assert ambiguous["status"] == "ambiguous_sku"
    assert ambiguous["unmodeled_reason"] == hw.AMBIGUOUS_SKU_REASON
    profile = ambiguous["device_capability_profile"]
    assert profile["cube_spec"]["aic_count_per_die"] is None
    assert profile["memory_hierarchy_spec"]["platform_config_hbm_mib"] is None
    assert profile["vector_spec"]["allocatable_ub_kib"] == 192.0
    assert ambiguous["buffers"]["ok"] is True
    assert ambiguous["thermal"]["status"] == "unmodeled"
    assert ambiguous["thermal"]["ideal_hardware_lower_bound_us"] is None

    # 2. Multi-SKU differences across A2 (910B1 vs 910B3 vs 910B4), A3 dual-die, and A5 (9581 vs 9582 vs PR)
    b1 = hw.resolve_device_capability_profile(soc="Ascend910B1")
    b3 = hw.resolve_device_capability_profile(soc="Ascend910B3")
    b4 = hw.resolve_device_capability_profile(config=hw.parse_platform_ini(FIXTURES / "Ascend910B4.ini"), snapshot=_a2_snapshot())
    assert (b1.cube_spec.aic_count_per_die, b1.memory_hierarchy_spec.platform_config_hbm_mib) == (24, 65536.0)
    assert (b3.cube_spec.aic_count_per_die, b3.memory_hierarchy_spec.platform_config_hbm_mib) == (20, 65536.0)
    assert (b4.cube_spec.aic_count_per_die, b4.memory_hierarchy_spec.verified_runtime_hbm_mib) == (20, 32768.0)

    # A3 INI (9391: 24 AIC/die @ 1850 MHz) overridden by live A3 snapshot (20 AIC/die @ 1650 MHz, dual-die)
    a3_prof = hw.resolve_device_capability_profile(
        config=hw.parse_platform_ini(FIXTURES / "Ascend910_9391.ini"),
        snapshot=_a3_snapshot(),
    )
    assert a3_prof.clock_and_topology_spec.arch_family == "A3"
    assert a3_prof.clock_and_topology_spec.dies_per_package == 2
    assert a3_prof.cube_spec.aic_count_per_die == 20
    assert a3_prof.vector_spec.aiv_count_per_die == 40
    assert any(d["field"] == "aic_count_per_die" and d["static_value"] == 24 and d["runtime_value"] == 20 for d in a3_prof.discrepancies)

    # A5 9581 vs 9582 vs PR and custom JSON SKU override
    a5_81 = hw.resolve_device_capability_profile(config=hw.parse_platform_ini(FIXTURES / "Ascend950DT_9581.ini"), simulator_ub_kib=256.0)
    a5_pr = hw.resolve_device_capability_profile(soc="Ascend950PR_9579")
    assert a5_81.vector_spec.physical_ub_kib == 256.0
    assert a5_81.vector_spec.allocatable_ub_kib == 248.0
    assert a5_81.memory_hierarchy_spec.hbm_peak_bw_gbps == 4000.0
    assert a5_pr.memory_hierarchy_spec.hbm_peak_bw_gbps == 1600.0

    custom_sku_path = _write_json(tmp_path, "custom_sku.json", {
        "soc_version": "Ascend950DT_9599",
        "arch_family": "A5",
        "cube_spec": {"aic_count_per_die": 36},
        "memory_hierarchy_spec": {"platform_config_hbm_mib": 196608.0, "hbm_peak_bw_gbps": 4800.0},
    })
    assert main(["hardware", "--sku-profile", str(custom_sku_path), "--allocatable-ub-kib", "240"]) == 0
    cli_custom = json.loads(capsys.readouterr().out)
    assert cli_custom["device_capability_profile"]["cube_spec"]["aic_count_per_die"] == 36
    assert cli_custom["device_capability_profile"]["vector_spec"]["allocatable_ub_kib"] == 240.0
    assert cli_custom["device_capability_profile"]["memory_hierarchy_spec"]["platform_config_hbm_mib"] == 196608.0


def test_interconnect_topo_pcie_downgrade_and_d2d_h2d_rh2d_rd2h_bounds(tmp_path: Path, capsys):
    from mindie_sim.__main__ import main

    # 1. Real A2 HCCS topology + NUMA affinity + PCIe Gen5 (32GT/s) -> Gen4 (16GT/s) downgrade
    a2_topo = hw.parse_npu_smi_topo(FIXTURES / "npu-smi-a2-topo.txt")
    assert a2_topo.has_sio_intra_package is False
    assert a2_topo.intra_node_fabric == "HCCS"
    assert a2_topo.link_type_counts["HCCS"] == 56
    assert len(a2_topo.numa_groups) == 4
    assert a2_topo.numa_groups[0] == ("NPU0", "NPU1")

    a2_pcie = hw.parse_pcie_link_info(FIXTURES / "lspci-a2-npu0.txt")
    assert a2_pcie.cap_speed_gtps == 32.0
    assert a2_pcie.sta_speed_gtps == 16.0
    assert a2_pcie.downgraded is True
    assert a2_pcie.effective_pinned_uni_bw_gbps == pytest.approx(16.0 * 1.9692 * 0.78, rel=1e-4)

    a2_prof = hw.resolve_device_capability_profile(
        config=hw.parse_platform_ini(FIXTURES / "Ascend910B4.ini"),
        snapshot=_a2_snapshot(),
        topo_snapshot=a2_topo,
        pcie_snapshot=a2_pcie,
    )
    assert any(d["field"] == "pcie_link_speed" and d["sta_speed_gtps"] == 16.0 for d in a2_prof.discrepancies)
    ch = a2_prof.memory_hierarchy_spec.transfer_channels
    assert set(ch) == set(hw.TRANSFER_CHANNELS)
    assert ch["d2d_intra_package"].effective_uni_bw_gbps is None
    assert ch["h2d"].calibrated_uni_bw_gbps == pytest.approx(24.576, rel=1e-3)

    # 2. H2D / D2H pinned vs pageable & Root Complex contention, plus RH2D / RD2H direct_rdma vs host_staged
    h2d_pinned = hw.solve_transfer_bound("h2d", 64 * MIB, ch, memory_type="pinned")
    h2d_pageable = hw.solve_transfer_bound("h2d", 64 * MIB, ch, memory_type="pageable")
    h2d_contended = hw.solve_transfer_bound("h2d", 64 * MIB, ch, memory_type="pinned", shared_root_complex_npus=2)
    assert h2d_pageable["lower_bound_us"] > h2d_pinned["lower_bound_us"]
    assert "pageable_host_memory_ceiling_applied" in h2d_pageable["flags"]
    assert h2d_contended["lower_bound_us"] > h2d_pinned["lower_bound_us"]

    rh2d_auto = hw.solve_transfer_bound("rh2d", 128 * MIB, ch)
    assert rh2d_auto["lower_bound_us"] is None
    assert "transport_mode_unverified" in rh2d_auto["flags"]
    assert rh2d_auto["host_staged_lower_bound_us"] > rh2d_auto["direct_rdma_lower_bound_us"]

    rd2h_staged = hw.solve_transfer_bound("rd2h", 128 * MIB, ch, transport_mode="host_staged", pipeline_chunks=4)
    assert rd2h_staged["lower_bound_us"] == pytest.approx(rh2d_auto["host_staged_lower_bound_us"])

    # 3. Real A3 16-Die SIO + HCCS_SW 3-tier collective communication bound
    a3_topo = hw.parse_npu_smi_topo(FIXTURES / "npu-smi-a3-topo.txt")
    assert a3_topo.has_sio_intra_package is True
    assert a3_topo.intra_node_fabric == "HCCS_SW"
    assert a3_topo.link_type_counts == {"SIO": 16, "HCCS_SW": 224}
    a3_prof = hw.resolve_device_capability_profile(
        config=hw.parse_platform_ini(FIXTURES / "Ascend910_9391.ini"),
        snapshot=_a3_snapshot(),
        topo_snapshot=a3_topo,
    )
    coll = hw.solve_collective_bound(
        "all_reduce",
        64 * MIB,
        a3_prof.memory_hierarchy_spec.transfer_channels,
        total_ranks=32,
        ranks_per_node=16,
        dies_per_package=2,
    )
    assert [t["tier"] for t in coll["tiers"]] == ["d2d_intra_package", "d2d_intra_node", "d2d_inter_node"]
    assert coll["lower_bound_us"] > 0

    # 4. Layer 4 net_report step decomposition with D2D + RH2D transfer overlap
    spec = _net_spec()
    spec["transfer_plan"] = {
        "overlapped_compute_ms": 1.0,
        "transfers": [{"channel": "rh2d", "size_bytes": 64 * MIB, "transport_mode": "direct_rdma"}],
        "collectives": [{"collective": "all_to_all", "size_bytes": 32 * MIB, "total_ranks": 64, "ranks_per_node": 8}],
    }
    net_out = net.net_report(spec)
    comm_out = net_out["step_decomposition"]["communication_and_transfers"]
    assert comm_out["total_transfer_lower_bound_ms"] > 1.0
    assert comm_out["exposed_transfer_ms"] == pytest.approx(comm_out["total_transfer_lower_bound_ms"] - 1.0)

    # 5. CLI hardware --topo-file --pcie-link-info --transfers
    transfers_path = _write_json(tmp_path, "transfers.json", spec["transfer_plan"])
    assert main([
        "hardware",
        "--ini", str(FIXTURES / "Ascend910B4.ini"),
        "--topo-file", str(FIXTURES / "npu-smi-a2-topo.txt"),
        "--pcie-link-info", str(FIXTURES / "lspci-a2-npu0.txt"),
        "--transfers", str(transfers_path),
    ]) == 0
    cli_hw = json.loads(capsys.readouterr().out)
    assert cli_hw["topology"]["intra_node_fabric"] == "HCCS"
    assert cli_hw["pcie_link"]["downgraded"] is True
    assert len(cli_hw["transfers"]["transfers"]) == 1


