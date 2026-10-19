"""Business tests for C7 comparison against the real analyze_profile producer."""

from __future__ import annotations

import copy

from mindie_profiler import Window, analyze_profile
from mindie_profiler.compare import COMPARISON_SCHEMA, compare_profiles
from tests.test_profiler_core import (
    RANK0_ID,
    STEP4,
    STEP5,
    build_run,
)

FULL_SCENE = {
    "framework": "vllm-ascend",
    "phase": "decode",
    "tp": 4,
    "dp": 1,
    "graph_mode": "piecewise_graph",
    "clock_domain": "syscnt_aligned",
    "runtime": {"soc": "Ascend910B4", "framework": {"torch_npu": "2.10.0.post4"}},
    "device": {"card": 0, "host": "a3-165"},
    "model": "tiny-llm",
    "shape": [8, 1024],
    "dtype": "bf16",
}


def _report(root, *, ranks=(RANK0_ID,), windows=(STEP4, STEP5), scene=FULL_SCENE):
    return analyze_profile(
        root,
        ranks=list(ranks) if ranks is not None else None,
        windows=list(windows),
        scene=dict(scene) if scene is not None else None,
    )


def _codes(result):
    return {item["code"] for item in result["limitations"]}


def _reasons(result):
    return {item["code"] for item in result["reasons"]}


def _block(result, name):
    return {item["block"]: item for item in result["semantic_blocks"]["blocks"]}[name]


def _attach_structure(report, *, layer_names=("layer_0", "layer_1"), shape=None, operator_sequence=None):
    report = copy.deepcopy(report)
    shape = shape or ["layer_0:attention", "layer_0:moe", "layer_1:norm"]
    report["structure"] = {
        "event_sequence_shape": list(shape),
        "layers": [
            {
                "layer_id": f"stable_{index}",
                "ordinal": index,
                "name": name,
                "busy_union_ms_p50": 0.010 * (index + 1),
                "wall_ms_p50": 0.020 * (index + 1),
                "host_gap_ms_p50": 0.001 * (index + 1),
            }
            for index, name in enumerate(layer_names)
        ],
    }
    if operator_sequence is not None:
        report["structure"]["operator_sequence"] = list(operator_sequence)
    return report


def test_real_analyze_output_is_consumed_by_compare_profiles(tmp_path):
    root = build_run(tmp_path)
    baseline = _report(root)
    candidate = _report(root)

    result = compare_profiles(baseline, candidate)

    assert result["schema_version"] == COMPARISON_SCHEMA
    assert result["comparable"] is True
    assert result["status"] == "comparable_with_limitations"
    assert result["step_accounting"]["status"] == "comparable"
    assert result["matching"]["clock_domain"]["status"] == "match"
    assert result["matching"]["rank_scope"]["baseline"] == [RANK0_ID]
    assert result["matching"]["steps"]["status"] == "same_steps"
    assert result["matching"]["steps"]["basis"] == "declared_p50"
    assert result["applicability"]["aligned_observation_count"] == 2
    assert result["applicability"]["missing_observation_count"] == 0

    wall = result["step_accounting"]["metrics"]["wall_ms_p50"]
    assert wall["baseline_p50"] == baseline["step_accounting"]["wall_ms_p50"]
    assert wall["candidate_p50"] == candidate["step_accounting"]["wall_ms_p50"]
    assert wall["delta"] == 0.0
    assert wall["confidence_interval"] is None
    assert wall["confidence_interval_reason"] == "fewer_than_6_per_sample_observations"
    assert result["step_accounting"]["metrics"]["aicpu_exposed_ratio"]["delta_pp"] == 0.0

    assert result["semantic_blocks"]["status"] == "comparable"
    assert _block(result, "Attention")["delta_ms"] == 0.0
    assert _block(result, "Host_Gap")["share_delta_pp"] is None
    assert result["structural_reuse"]["status"] == "not_available"
    assert result["hierarchy"]["status"] == "not_available"
    assert result["operator_sequence"]["status"] == "not_available"
    assert result["operator_sequence"]["classification"] == "unknown"
    assert result["operator_sequence"]["reuse_recommendation"] == "not_available"
    assert "input_identity_unavailable" in _codes(result)
    assert "structure_evidence_unavailable" in _codes(result)
    assert "operator_sequence_unavailable" in _codes(result)


def test_partial_step_window_recomputes_step_and_blocks_from_common_rows(tmp_path):
    root = build_run(tmp_path)
    baseline = _report(root, windows=(STEP4, STEP5))
    candidate = _report(root, windows=(STEP5,))

    # The declared baseline p50 spans steps 4 and 5; a naive p50 reuse would
    # report 0.035 vs 0.025 and invent a change.
    assert baseline["step_accounting"]["wall_ms_p50"] != candidate["step_accounting"]["wall_ms_p50"]
    result = compare_profiles(baseline, candidate)

    assert result["status"] == "comparable_with_limitations"
    assert result["matching"]["steps"]["status"] == "partial_overlap"
    assert result["matching"]["steps"]["common"] == [5]
    assert result["matching"]["steps"]["basis"] == "recomputed_from_window_accounting"
    assert result["applicability"]["compared_steps"] == [5]
    assert result["applicability"]["aligned_observation_count"] == 1

    wall = result["step_accounting"]["metrics"]["wall_ms_p50"]
    assert wall["baseline_p50"] == 0.025
    assert wall["candidate_p50"] == 0.025
    assert wall["delta"] == 0.0
    attention = _block(result, "Attention")
    assert attention["baseline_busy_union_ms_p50"] == 0.010
    assert attention["candidate_busy_union_ms_p50"] == 0.010
    assert attention["delta_ms"] == 0.0
    assert _block(result, "MoE")["baseline_busy_union_ms_p50"] == 0.0
    assert "partial_step_overlap" in _codes(result)


def test_missing_common_observation_is_reported_and_excluded(tmp_path):
    root = build_run(tmp_path)
    baseline = _report(root)
    candidate = _report(root)
    candidate["window_accounting"] = [
        row for row in candidate["window_accounting"] if row["step"] != 5
    ]

    result = compare_profiles(baseline, candidate)

    assert result["comparable"] is True
    assert result["matching"]["steps"]["alignment"] == "incomplete"
    assert result["matching"]["steps"]["missing"] == [{"step": 5, "rank_id": RANK0_ID}]
    assert result["applicability"]["aligned_observation_count"] == 1
    assert result["applicability"]["missing_observation_count"] == 1
    assert "incomplete_common_observations" in _codes(result)
    wall = result["step_accounting"]["metrics"]["wall_ms_p50"]
    assert wall["baseline_p50"] == 0.045
    assert wall["candidate_p50"] == 0.045
    assert wall["delta"] == 0.0


def test_partial_rank_scope_aligns_only_common_rank_ids(tmp_path):
    root = build_run(tmp_path)
    baseline = _report(root, ranks=None)
    candidate = _report(root, ranks=(RANK0_ID,))

    result = compare_profiles(baseline, candidate)

    assert result["comparable"] is True
    assert result["matching"]["rank_scope"]["status"] == "partial_overlap"
    assert result["matching"]["rank_scope"]["common"] == [RANK0_ID]
    assert result["applicability"]["compared_ranks"] == [RANK0_ID]
    assert result["applicability"]["aligned_observation_count"] == 2
    assert result["matching"]["slowest_rank"]["status"] == "changed"
    assert "rank_scope_partial_overlap" in _codes(result)
    assert result["step_accounting"]["status"] == "comparable"


def test_unknown_clock_is_not_comparable_even_when_both_say_unknown(tmp_path):
    root = build_run(tmp_path)
    scene = {key: value for key, value in FULL_SCENE.items() if key != "clock_domain"}
    baseline = _report(root, scene=scene)
    candidate = _report(root, scene=scene)
    assert baseline["scene"]["clock_domain"] == "unknown"
    assert candidate["scene"]["clock_domain"] == "unknown"

    result = compare_profiles(baseline, candidate)

    assert result["status"] == "not_comparable"
    assert result["comparable"] is False
    assert "clock_domain_unknown" in _reasons(result)
    assert result["step_accounting"] is None


def test_missing_scene_facts_are_not_declared_a_match(tmp_path):
    root = build_run(tmp_path)
    baseline = _report(root, scene=None)
    candidate = _report(root, scene=None)
    assert baseline["scene"]["scene_source"] == "not_provided"
    assert baseline["scene"]["framework"] is None
    assert baseline["scene"]["clock_domain"] == "unknown"

    result = compare_profiles(baseline, candidate)

    assert result["status"] == "not_comparable"
    assert "scene_not_provided" in _reasons(result)
    assert "clock_domain_unknown" in _reasons(result)
    assert result["matching"]["scene"]["framework"]["status"] == "unavailable"


def test_scene_and_clock_mismatch_are_not_comparable(tmp_path):
    root = build_run(tmp_path)
    baseline = _report(root)
    phase = _report(root, scene={**FULL_SCENE, "phase": "prefill"})
    clock = _report(root, scene={**FULL_SCENE, "clock_domain": "host_wall_clock"})

    phase_result = compare_profiles(baseline, phase)
    assert phase_result["status"] == "not_comparable"
    assert "scene_phase_mismatch" in _reasons(phase_result)

    clock_result = compare_profiles(baseline, clock)
    assert clock_result["status"] == "not_comparable"
    assert "clock_domain_mismatch" in _reasons(clock_result)


def test_input_identity_is_required_when_declared_and_unknown_when_absent(tmp_path):
    root = build_run(tmp_path)
    baseline = _report(root)
    candidate = _report(root)
    baseline["scene"]["input_signature"] = {"batch": 8, "seq": 1024}
    candidate["scene"]["input_signature"] = {"batch": 16, "seq": 1024}

    mismatch = compare_profiles(baseline, candidate)
    assert mismatch["status"] == "not_comparable"
    assert "input_identity_mismatch" in _reasons(mismatch)

    matching = compare_profiles(baseline, copy.deepcopy(baseline))
    assert matching["matching"]["input"]["status"] == "match"


def test_structure_alignment_uses_shape_and_layer_identity_not_db_hash(tmp_path):
    root = build_run(tmp_path)
    baseline = _attach_structure(_report(root))
    candidate = _attach_structure(_report(root))
    candidate["source_identity"]["db_files_sha256"][RANK0_ID] = "f" * 64
    candidate["source_identity"]["query_elapsed_ms"] = 999.0
    candidate["source_identity"]["cache_state"] = "warm"
    assert baseline["source_identity"]["db_files_sha256"] != candidate["source_identity"]["db_files_sha256"]

    result = compare_profiles(baseline, candidate)

    assert result["structural_reuse"]["status"] == "verified"
    assert result["structural_reuse"]["allowed"] is True
    assert result["structural_reuse"]["baseline_source_content"]["db_files_sha256"][RANK0_ID] != \
        result["structural_reuse"]["candidate_source_content"]["db_files_sha256"][RANK0_ID]
    assert "query_elapsed_ms" not in result["structural_reuse"]["baseline_source_content"]
    assert "cache_state" not in result["structural_reuse"]["baseline_source_content"]
    assert result["hierarchy"]["status"] == "comparable"
    assert result["hierarchy"]["layers"][0]["match_basis"] == "event_sequence_shape+layer_id+ordinal"
    assert result["step_accounting"]["status"] == "comparable"


def test_name_only_layer_is_not_reused_but_step_summary_survives(tmp_path):
    root = build_run(tmp_path)
    baseline = _attach_structure(_report(root))
    candidate = _attach_structure(_report(root))
    candidate["structure"]["layers"][1]["layer_id"] = "wrong_layer_id"

    result = compare_profiles(baseline, candidate)

    assert result["structural_reuse"]["status"] == "rejected"
    assert result["structural_reuse"]["reason_codes"] == ["layer_identity_mismatch"]
    assert result["hierarchy"]["status"] == "not_comparable"
    assert result["hierarchy"]["layers"] == []
    assert "structure_reuse_not_established" in _codes(result)
    assert result["comparable"] is True
    assert result["step_accounting"]["status"] == "comparable"


def test_event_shape_mismatch_rejects_structure_without_blocking_steps(tmp_path):
    root = build_run(tmp_path)
    baseline = _attach_structure(_report(root))
    candidate = _attach_structure(_report(root), shape=["layer_0:attention"])

    result = compare_profiles(baseline, candidate)

    assert result["structural_reuse"]["status"] == "rejected"
    assert "event_sequence_shape_mismatch" in result["structural_reuse"]["reason_codes"]
    assert result["hierarchy"]["status"] == "not_comparable"
    assert result["comparable"] is True
    assert result["step_accounting"]["status"] == "comparable"


def test_confidence_interval_uses_six_real_aligned_observations(tmp_path):
    root = build_run(tmp_path)
    step6 = Window(start_ns=0, end_ns=60_000, step=6, label="step_6")
    baseline = _report(root, ranks=None, windows=(STEP4, STEP5, step6))
    candidate = _report(root, ranks=None, windows=(STEP4, STEP5, step6))

    result = compare_profiles(baseline, candidate)

    wall = result["step_accounting"]["metrics"]["wall_ms_p50"]
    assert wall["baseline_sample_count"] == 6
    assert wall["candidate_sample_count"] == 6
    assert wall["confidence_interval"] is not None
    assert wall["confidence_interval"]["method"] == "distribution_free_median_order_statistics"
    assert result["applicability"]["confidence_intervals"] == (
        "computed_from_aligned_window_observations"
    )
    attention = _block(result, "Attention")
    assert attention["confidence_interval"] is not None


def test_same_step_but_missing_metric_value_is_not_zero(tmp_path):
    root = build_run(tmp_path)
    baseline = _report(root)
    candidate = _report(root)
    candidate["window_accounting"][0]["wall_ms"] = None

    result = compare_profiles(baseline, candidate)

    wall = result["step_accounting"]["metrics"]["wall_ms_p50"]
    assert wall["status"] == "not_comparable"
    assert wall["candidate_p50"] is None
    assert wall["delta"] is None
    assert "metric_incomplete_observation_coverage" in _codes(result)
    assert result["comparable"] is True


def test_invalid_or_incompatible_input_is_returned_not_raised(tmp_path):
    root = build_run(tmp_path)
    wrong = _report(root)
    wrong["schema_version"] = "mindie-profile-summary/2"

    result = compare_profiles(_report(root), wrong)
    assert result["status"] == "not_comparable"
    assert "unsupported_schema_version" in _reasons(result)

    non_mapping = compare_profiles(None, _report(root))
    assert non_mapping["status"] == "not_comparable"
    assert "summary_not_an_object" in _reasons(non_mapping)


def test_comparison_does_not_mutate_real_reports(tmp_path):
    root = build_run(tmp_path)
    baseline = _attach_structure(_report(root))
    candidate = _attach_structure(_report(root))
    before_baseline = copy.deepcopy(baseline)
    before_candidate = copy.deepcopy(candidate)

    compare_profiles(baseline, candidate)

    assert baseline == before_baseline
    assert candidate == before_candidate


def test_required_run_binding_accepts_explicit_runtime_device_model_shape_dtype(tmp_path):
    root = build_run(tmp_path)
    baseline = _report(root)
    candidate = _report(root)

    result = compare_profiles(baseline, candidate, require_binding=True)

    assert result["comparable"] is True
    assert result["matching"]["run_binding"]["status"] == "match"
    assert result["matching"]["run_binding"]["fields"]["runtime"]["status"] == "match"
    assert result["matching"]["run_binding"]["fields"]["device"]["status"] == "match"
    assert result["matching"]["run_binding"]["fields"]["model"]["status"] == "match"
    assert result["matching"]["run_binding"]["fields"]["shape"]["status"] == "match"
    assert result["matching"]["run_binding"]["fields"]["dtype"]["status"] == "match"


def test_required_run_binding_refuses_missing_or_mismatched_facts(tmp_path):
    root = build_run(tmp_path)
    baseline = _report(root)
    missing = _report(root)
    for field in ("runtime", "device", "model", "shape", "dtype"):
        missing["scene"].pop(field, None)

    missing_result = compare_profiles(baseline, missing, require_binding=True)
    assert missing_result["status"] == "not_comparable"
    assert "run_binding_field_unavailable" in _reasons(missing_result)

    mismatch = _report(root)
    mismatch["scene"]["device"] = {"card": 1, "host": "a3-165"}
    mismatch["scene"]["shape"] = [16, 1024]
    mismatch_result = compare_profiles(baseline, mismatch, require_binding=True)
    assert mismatch_result["status"] == "not_comparable"
    assert "run_binding_field_mismatch" in _reasons(mismatch_result)


def test_operator_sequence_small_medium_large_classification(tmp_path):
    root = build_run(tmp_path)
    sequence = [f"op_{index}" for index in range(10)]
    baseline = _attach_structure(_report(root), operator_sequence=sequence)

    small = _attach_structure(_report(root), operator_sequence=[*sequence[:8], "op_changed", sequence[9]])
    medium = _attach_structure(_report(root), operator_sequence=[*sequence[:7], "op_a", "op_b", "op_c"])
    large = _attach_structure(_report(root), operator_sequence=[f"other_{index}" for index in range(10)])

    small_result = compare_profiles(baseline, small)
    medium_result = compare_profiles(baseline, medium)
    large_result = compare_profiles(baseline, large)

    assert small_result["operator_sequence"]["classification"] == "small"
    assert small_result["operator_sequence"]["reuse_recommendation"] == "layer_identity_reuse_with_event_shape"
    assert medium_result["operator_sequence"]["classification"] == "medium"
    assert medium_result["operator_sequence"]["reuse_recommendation"] == "block_level_only"
    assert large_result["operator_sequence"]["classification"] == "large"
    assert large_result["operator_sequence"]["reuse_recommendation"] == "none"
    assert small_result["operator_sequence"]["basis"] == "structure.operator_sequence"


def test_median_tables_include_layers_seven_blocks_and_bubbles(tmp_path):
    root = build_run(tmp_path)
    baseline = _attach_structure(_report(root), operator_sequence=["op_a", "op_b"])
    candidate = _attach_structure(_report(root), operator_sequence=["op_a", "op_b"])

    result = compare_profiles(baseline, candidate)

    assert len(result["tables"]["layers"]) == 2
    assert result["tables"]["layers"][0]["match_basis"] == "event_sequence_shape+layer_id+ordinal"
    assert len(result["tables"]["semantic_blocks"]) == 7
    assert [row["block"] for row in result["tables"]["semantic_blocks"]] == [
        "Attention", "MoE", "Comm", "Norm", "Copy_Layout", "Unclassified", "Host_Gap",
    ]
    assert {row["name"] for row in result["tables"]["bubble"]} == {
        "host_gap_ms_p50", "underfeed_bubble_ratio",
        "aicpu_exposed_ratio", "exposed_comm_ms_p50",
    }


def test_required_binding_rejects_empty_or_unknown_values(tmp_path):
    root = build_run(tmp_path)
    baseline = _report(root)
    empty = _report(root)
    empty["scene"]["runtime"] = {}
    empty["scene"]["device"] = {}
    empty["scene"]["model"] = ""
    empty["scene"]["shape"] = []
    empty["scene"]["dtype"] = "unknown"

    result = compare_profiles(baseline, empty, require_binding=True)

    assert result["status"] == "not_comparable"
    assert result["matching"]["run_binding"]["status"] == "not_comparable"
    assert "run_binding_field_unavailable" in _reasons(result)
