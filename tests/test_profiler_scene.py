"""Pure-logic tests for scene/steady detection and interval anomaly discovery.

Fixtures are known arithmetic windows and intervals. They are not presented as
real device results and do not claim that a particular NPU bottleneck occurred.
"""
from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

import pytest

from mindie_profiler import Window, analyze_profile
from mindie_profiler import anomaly_discovery as ad
from mindie_profiler import scene_detector as sd
from mindie_profiler.theoretical_model import (
    HardwareSpec,
    KernelInputs,
    ProviderIdentity,
    ProviderPrediction,
)

CORE_CSV = Path(__file__).resolve().parent / "fixtures" / "profiler_core" / "rank_0" / "kernel_details.csv"


def _samples(values, *, start=1, rank=None):
    return [sd.StepSample(step=start + index, wall_ms=value, rank=rank) for index, value in enumerate(values)]


def _event(
    name,
    step,
    start,
    duration,
    *,
    wait=None,
    task=None,
    stream=None,
    rank=None,
    event_id=None,
    related=(),
    counters=(),
):
    return ad.KernelEvent(
        name=name,
        step=step,
        start_us=start,
        duration_us=duration,
        wait_us=wait,
        task_type=task,
        stream_id=stream,
        rank=rank,
        event_id=event_id,
        related_event_ids=related,
        counters=counters,
    )


def _shifted_core_csv(tmp_path, offsets, *, extra_rows=()):
    """Copy the real core CSV header/rows, shifting only explicit times."""
    with CORE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [row for row in reader if row]
    columns = {name: index for index, name in enumerate(header)}
    start_index = columns["Start Time(us)"]
    destination = tmp_path / "run" / "rank_0" / "kernel_details.csv"
    destination.parent.mkdir(parents=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for offset in offsets:
            for row in rows:
                shifted = list(row)
                value = Decimal(shifted[start_index].strip()) + Decimal(offset)
                shifted[start_index] = f"{value:.3f}\t"
                writer.writerow(shifted)
        for values in extra_rows:
            row = [""] * len(header)
            for name, value in values.items():
                row[columns[name]] = value
            writer.writerow(row)
    return destination.parent.parent


class _FixtureProvider:
    identity = ProviderIdentity("fixture-only-provider", "9.9.9")

    def available(self) -> bool:
        return True

    def predict(self, request):
        measured = request.measured_device_ms or 0.01
        return ProviderPrediction(
            predicted_ms=measured * 0.8,
            model_assumptions="fixture-only assumptions",
            input_signature=request.input_signature,
            evidence=("fixture:analyze-integration",),
        )


def _windows(steps=(1, 2, 3)):
    return [
        Window(start_ns=index * 100_000, end_ns=index * 100_000 + 60_000, step=step, label=f"step_{step}")
        for index, step in enumerate(steps)
    ]


@pytest.mark.parametrize(
    ("annotation", "mode"),
    [
        ("inference", "inference"),
        ("infer", "inference"),
        ("training", "training"),
        ("train", "training"),
        ("rl", "rl"),
        ("reinforcement-learning", "rl"),
    ],
)
def test_explicit_scene_annotation_is_honoured(annotation, mode):
    result = sd.detect_scene({"mode": annotation, "phase": "decode", "framework": "vllm-ascend"})

    assert result.status == "detected"
    assert result.mode == mode
    assert result.phase == "decode"
    assert result.as_c7_scene()["framework"] == "vllm-ascend"


def test_scene_is_not_inferred_from_framework_or_missing_annotation():
    missing = sd.detect_scene({"framework": "vllm-ascend"})
    unknown = sd.detect_scene({"mode": "benchmark-harness"})

    assert missing.status == "uncertain"
    assert missing.mode is None
    assert missing.uncertainty_reason == "missing_explicit_scene_annotation"
    assert unknown.status == "uncertain"
    assert "unsupported_scene_annotation" in unknown.uncertainty_reason


def test_scene_annotation_from_metadata_annotation_mapping():
    metadata = sd.SceneMetadata(
        framework="mindie",
        annotations={"scene": "training", "phase": "train_backward"},
    )
    result = sd.detect_scene(metadata)

    assert result.status == "detected"
    assert result.mode == "training"
    assert result.phase == "train_backward"
    assert result.annotation_source == "annotations.scene"


def test_steady_window_keeps_known_stable_suffix_and_excludes_warmup():
    samples = _samples([50.0, 40.0, 10.0, 10.0, 11.0, 10.0, 9.0, 10.0])

    window = sd.detect_steady_window(samples)

    assert window.status == "steady"
    assert window.steady_steps == (3, 4, 5, 6, 7, 8)
    assert window.window_start_step == 3
    assert window.window_end_step == 8
    assert window.excluded_steps == (1, 2)
    assert window.baseline_ms == pytest.approx(10.0)
    assert window.as_c7_dict()["steady_steps"] == [3, 4, 5, 6, 7, 8]


def test_insufficient_or_unstable_samples_are_uncertain_not_guessed():
    too_few = sd.detect_steady_window(_samples([10.0, 10.0]))
    unstable = sd.detect_steady_window(_samples([1.0, 100.0, 2.0, 200.0, 3.0, 300.0]))

    assert too_few.status == "uncertain"
    assert too_few.uncertainty_reason == "insufficient_step_samples"
    assert too_few.steady_steps == ()
    assert unstable.status == "uncertain"
    assert unstable.steady_steps == ()
    assert unstable.uncertainty_reason == "no_steady_suffix_within_tolerance"


def test_missing_metric_and_duplicate_step_are_uncertain():
    missing_metric = sd.detect_steady_window([
        sd.StepSample(step=1, wall_ms=10.0),
        sd.StepSample(step=2, wall_ms=None),
        sd.StepSample(step=3, wall_ms=10.0),
    ])
    duplicate = sd.detect_steady_window(_samples([10.0, 10.0, 10.0], start=7) + _samples([10.0], start=7))

    assert missing_metric.status == "uncertain"
    assert missing_metric.uncertainty_reason == "missing_metric:wall_ms"
    assert duplicate.status == "uncertain"
    assert duplicate.uncertainty_reason == "duplicate_step_identity"


def test_conflicting_step_and_step_id_is_uncertain():
    samples = [
        sd.StepSample(step=1, step_id=1, wall_ms=10.0),
        sd.StepSample(step=2, step_id=99, wall_ms=10.0),
        sd.StepSample(step=3, step_id=3, wall_ms=10.0),
    ]

    window = sd.detect_steady_window(samples)

    assert window.status == "uncertain"
    assert window.uncertainty_reason == "conflicting_step_identity"


def test_metadata_hints_only_filter_existing_measured_steps():
    samples = _samples([99.0, 10.0, 10.0, 10.0, 10.0])
    metadata = sd.SceneMetadata(annotations={"warmup_steps": "1", "steady_start_step": "2"})

    window = sd.detect_steady_window(samples, metadata)

    assert window.status == "steady"
    assert window.steady_steps == (2, 3, 4, 5)
    assert window.metadata_used == ("warmup_steps", "steady_start_step")


def test_step_id_alias_and_mapping_samples_are_accepted():
    samples = [
        {"step_id": 1, "wall_ms": 50.0},
        {"step_id": 2, "wall_ms": 10.0},
        {"step_id": 3, "wall_ms": 10.0},
        {"step_id": 4, "wall_ms": 10.0},
    ]

    window = sd.detect_steady_window(samples)

    assert window.status == "steady"
    assert window.steady_steps == (2, 3, 4)


def test_scene_and_steady_combined_fragment_is_plain_dict():
    result = sd.detect_scene_and_steady(
        {"mode": "inference", "phase": "decode"},
        _samples([50.0, 10.0, 10.0, 10.0]),
    )
    payload = result.as_c7_dict()

    assert payload["scene"]["phase"] == "decode"
    assert payload["steady_window"]["steady_steps"] == [2, 3, 4]


def test_interval_helpers_use_union_not_sum_and_report_overlap():
    merged = ad.merge_intervals([(0, 10), (5, 20), (30, 40), (40, 50)])
    assert merged == [(0, 20), (30, 50)]
    assert ad.union_ms([(0, 10), (5, 20)]) == pytest.approx(0.02)
    assert ad.intersection_ms((10, 20), [(5, 15), (12, 25)]) == pytest.approx(0.01)
    assert ad.overlap_ratio((10, 20), [(5, 15)]) == pytest.approx(0.5)
    assert ad.overlap_ratio((10, 10), [(5, 15)]) is None


def test_wait_anchor_requires_overlapping_correlated_event_not_name():
    anchor = _event(
        "ordinary_compute_name", 1, 100, 2, wait=1000, task="AI_CORE",
        stream=0, event_id="anchor",
    )
    overlapping = _event(
        "ordinary_upstream_name", 1, 0, 1000, wait=0, task="AI_CORE",
        stream=0, event_id="upstream",
    )

    report = ad.assess_wait_anchor([anchor, overlapping])
    finding = report.findings[0]

    assert finding["is_false_hotspot_risk"] is True
    assert finding["wait_anchor_false_hotspot"] is True
    assert finding["evidence_status"] == "confirmed"
    assert "ordinary_compute_name" in finding["op_name"]


def test_wait_anchor_without_overlap_or_association_is_only_insufficient():
    anchor = _event("anchor", 1, 100, 2, wait=1000, task="AI_CORE", stream=0, event_id="a")
    other_stream = _event("other", 1, 0, 1000, wait=0, task="AI_CORE", stream=7, event_id="b")

    report = ad.assess_wait_anchor([anchor, other_stream])
    finding = report.findings[0]

    assert finding["is_false_hotspot_risk"] is False
    assert finding["evidence_status"] == "insufficient"
    assert finding["reason"] == "missing_correlated_event_overlap_for_wait_interval"


def test_aicpu_exposure_uses_explicit_type_and_core_overlap():
    events = [
        _event("opaque_a", 4, 0, 100, task="AI_CPU", event_id="a"),
        _event("opaque_b", 4, 0, 80, task="AI_CORE", event_id="b"),
        _event("opaque_c", 4, 100, 100, task="AI_CPU", event_id="c"),
    ]

    report = ad.assess_aicpu_exposure(events)

    assert report.status == "assessed"
    assert report.aicpu_duration_ms == pytest.approx(0.2)
    assert report.masked_ms == pytest.approx(0.08)
    assert report.exposed_ms == pytest.approx(0.12)
    assert report.masked_ratio == pytest.approx(0.4)
    assert report.exposed_ratio == pytest.approx(0.6)
    assert report.classification == "AICPU_PARTIALLY_EXPOSED"
    assert report.counter_status == "missing"
    assert "pmu_counters_missing" in report.limitations
    assert report.as_c7_dict()["attribution"] == "exposure_only"


@pytest.mark.parametrize(
    ("core_duration", "classification"),
    [
        (100, "AICPU_MASKED_BUT_UNDESIRABLE"),
        (50, "AICPU_PARTIALLY_EXPOSED"),
        (10, "AICPU_EXPOSED_NOT_ALLOWED"),
    ],
)
def test_aicpu_classification_uses_overlap_thresholds(core_duration, classification):
    events = [
        _event("opaque_aicpu", 1, 0, 100, task="AI_CPU", event_id="a"),
        _event("opaque_core", 1, 0, core_duration, task="AI_CORE", event_id="b"),
    ]

    report = ad.assess_aicpu_exposure(events)

    assert report.status == "assessed"
    assert report.classification == classification


def test_aicpu_mask_all_non_aicpu_supports_optional_step_evidence():
    events = [
        ad.KernelEvent("aicpu", None, 0, 100, task_type="AI_CPU", event_id="a"),
        ad.KernelEvent("other", None, 0, 60, task_type="mystery", event_id="b"),
    ]

    report = ad.assess_aicpu_exposure(events, mask_all_non_aicpu=True)

    assert report.status == "assessed"
    assert report.masked_ratio == pytest.approx(0.6)
    assert report.exposed_ratio == pytest.approx(0.4)


def test_aicpu_name_alone_is_not_exposure_evidence():
    events = [
        _event("AICPU_fallback", 1, 0, 100),
        _event("AI_CORE_kernel", 1, 0, 80),
    ]

    report = ad.assess_aicpu_exposure(events)

    assert report.status == "uncertain"
    assert "missing_task_type" in report.limitations
    assert report.classification is None


def test_aicpu_missing_type_or_core_reference_is_uncertain():
    missing_type = [
        _event("opaque", 1, 0, 100),
        _event("other", 1, 0, 80),
    ]
    missing_core = [_event("opaque", 1, 0, 100, task="AI_CPU")]

    type_report = ad.assess_aicpu_exposure(missing_type)
    core_report = ad.assess_aicpu_exposure(missing_core)

    assert type_report.status == "uncertain"
    assert "missing_task_type" in type_report.limitations
    assert core_report.status == "uncertain"
    assert any(item.startswith("missing_ai_core_events") for item in core_report.limitations)


def test_missing_wait_data_is_uncertain_and_sample_only_fragment_has_no_counter_claim():
    events = [_event("opaque", 1, 0, 100, task="AI_CORE", event_id="a")]
    wait_report = ad.assess_wait_anchor(events)

    payload = ad.build_c7_fragment(samples=_samples([10.0, 10.0, 10.0]), events=[])

    assert wait_report.status == "uncertain"
    assert "missing_wait_time_for_all_events" in wait_report.limitations
    assert payload["findings"] == []
    assert payload["anomaly_discovery"]["aicpu_exposure"]["status"] == "not_applicable"


def test_step_timing_keeps_union_sum_and_exposed_comm_distinct():
    events = [
        _event("compute", 2, 0, 1000, wait=0, task="AI_CORE", event_id="c"),
        _event("hccl", 2, 200, 300, wait=0, task="HCCL", event_id="h"),
    ]
    sample = sd.StepSample(step=2, wall_ms=1.5, host_gap_ms=0.2)

    timing = ad.compute_step_timing(2, events, sample)

    assert timing.wall_ms == pytest.approx(1.5)
    assert timing.busy_union_ms == pytest.approx(1.0)
    assert timing.kernel_sum_ms == pytest.approx(1.3)
    assert timing.exposed_comm_ms == pytest.approx(0.0)
    assert timing.host_gap_ms == pytest.approx(0.2)
    assert timing.total_cost_ms == pytest.approx(1.3)
    assert timing.underfeed_bubble_ratio == pytest.approx((1.5 - 1.0) / 1.5)
    payload = timing.as_c7_dict()
    assert payload["busy_union_ms"] != payload["kernel_sum_ms"]


def test_missing_task_type_leaves_exposed_comm_unknown():
    events = [_event("opaque", 3, 0, 100, wait=0)]
    timing = ad.compute_step_timing(3, events, sd.StepSample(step=3, wall_ms=0.1))

    assert timing.exposed_comm_ms is None
    assert "missing_task_type_exposed_comm" in timing.limitations
    assert timing.total_cost_ms == pytest.approx(0.1)


def test_step_accounting_uses_sample_aggregates_when_events_are_absent():
    samples = [
        sd.StepSample(step=1, wall_ms=10.0, busy_union_ms=8.0, kernel_sum_ms=9.0,
                      exposed_comm_ms=1.0, host_gap_ms=2.0, total_cost_ms=9.5, rank=0),
        sd.StepSample(step=2, wall_ms=12.0, busy_union_ms=9.0, kernel_sum_ms=10.0,
                      exposed_comm_ms=1.5, host_gap_ms=3.0, total_cost_ms=10.5, rank=1),
    ]

    accounting = ad.summarize_step_accounting(samples)

    assert accounting["wall_ms_p50"] == pytest.approx(11.0)
    assert accounting["busy_union_ms_p50"] == pytest.approx(8.5)
    assert accounting["kernel_sum_ms_p50"] == pytest.approx(9.5)
    assert accounting["exposed_comm_ms_p50"] == pytest.approx(1.25)
    assert accounting["host_gap_ms_p50"] == pytest.approx(2.5)
    assert accounting["total_cost_ms_p50"] == pytest.approx(10.0)
    assert accounting["slowest_rank"]["rank"] == 1


def test_c7_fragment_is_plain_dict_and_limits_missing_counters():
    samples = _samples([10.0, 10.0, 10.0], start=1)
    events = [
        _event("anchor", 1, 100, 2, wait=1000, task="AI_CORE", stream=0, event_id="a"),
        _event("upstream", 1, 0, 1000, wait=0, task="AI_CORE", stream=0, event_id="b"),
        _event("aicpu", 2, 0, 100, task="AI_CPU", event_id="c"),
        _event("core", 2, 0, 80, task="AI_CORE", event_id="d"),
    ]

    payload = ad.build_c7_fragment(samples=samples, events=events)

    assert payload["schema_version"] == "mindie-profile-summary/1"
    assert set(payload["step_accounting"]).issuperset({
        "steady_steps", "wall_ms_p50", "busy_union_ms_p50", "kernel_sum_ms_p50",
        "exposed_comm_ms_p50", "host_gap_ms_p50", "total_cost_ms_p50",
        "underfeed_bubble_ratio", "aicpu_exposed_ratio", "slowest_rank",
    })
    finding_types = {item["finding_type"] for item in payload["findings"]}
    assert "wait_anchor_masked" in finding_types
    assert "aicpu_fallback_exposed" in finding_types
    assert "measurement_limitation" in finding_types
    assert payload["anomaly_discovery"]["aicpu_exposure"]["counter_status"] == "missing"


def test_analyze_profile_c7_integration_preserves_accounting_and_uses_measured_wait(tmp_path):
    root = _shifted_core_csv(tmp_path, offsets=(0, 100, 200))
    report = analyze_profile(
        root,
        windows=_windows(),
        scene={"mode": "inference", "phase": "decode"},
    )

    assert report["step_accounting"]["steady_steps"] == [1, 2, 3]
    assert report["step_accounting"]["wall_ms_p50"] == 0.045
    assert report["window_accounting"][0]["wall_ms"] == 0.045
    assert report["window_accounting"][0]["busy_union_ms"] == 0.040

    assert report["scene_detection"]["status"] == "detected"
    assert report["scene_detection"]["mode"] == "inference"
    steady = report["steady_window_suggestion"]
    assert steady["status"] == "steady"
    assert steady["steady_steps"] == [1, 2, 3]
    assert steady["filters_user_windows"] is False

    assert report["top_hotspot_kernels"]
    assert all(item["wait_anchor_false_hotspot"] is None for item in report["top_hotspot_kernels"])
    wait_scope = report["enrichment"]["anomaly_discovery"]["wait_anchor"]["measurement_scope"]
    assert wait_scope["wait_bound_event_count"] > 0
    assert wait_scope["wait_usable_event_count"] > 0
    codes = {item["code"] for item in report["limitations"]}
    assert "wait_anchor_unverified" not in codes
    assert "pmu_values_absent" in codes
    assert report["enrichment"]["pmu_observations"]["status"] == "not_available"
    assert any(
        item["finding_type"] == "measurement_limitation"
        and item.get("limitation_code") == "pmu_values_absent"
        for item in report["findings"]
    )
    assert all(
        item["modeling_alignment"]["status"] == "unmodeled"
        and item["modeling_alignment"]["model_predicted_ms"] is None
        for item in report["top_hotspot_kernels"]
    )

    missing_scene = analyze_profile(root, windows=_windows(), scene={"framework": "vllm-ascend"})
    assert missing_scene["scene_detection"]["status"] == "uncertain"
    assert missing_scene["scene_detection"]["uncertainty_reason"] == "missing_explicit_scene_annotation"


def test_analyze_profile_optional_modeling_shows_measured_ideal_prediction_and_scope(tmp_path):
    root = _shifted_core_csv(tmp_path, offsets=(0, 100, 200))
    baseline = analyze_profile(root, windows=_windows())
    inputs = KernelInputs(shape=(1, 4, 8), dtype="bf16", layout="ND", op_kind="matmul")
    report = analyze_profile(
        root,
        windows=_windows(),
        scene={"mode": "inference", "phase": "decode"},
        model_provider=_FixtureProvider(),
        model_inputs={"aclnnFlashAttention": inputs},
        model_hardware=HardwareSpec("fixture-only-npu", 1e12, 1e9),
    )

    for key in (
        "wall_ms_p50", "busy_union_ms_p50", "kernel_sum_ms_p50", "exposed_comm_ms_p50",
        "host_gap_ms_p50", "total_cost_ms_p50", "underfeed_bubble_ratio", "aicpu_exposed_ratio",
    ):
        assert report["step_accounting"][key] == baseline["step_accounting"][key]

    hotspot = next(item for item in report["top_hotspot_kernels"] if item["kernel_name"] == "aclnnFlashAttention")
    alignment = hotspot["modeling_alignment"]
    assert alignment["status"] == "modeled"
    assert alignment["measured_device_ms"] == pytest.approx(0.01)
    assert alignment["ideal_hardware_lower_bound_ms"] is not None
    assert alignment["ideal_hardware_lower_bound_ms"] > 0
    assert alignment["model_predicted_ms"] == pytest.approx(0.008)
    assert alignment["model_provider"] == "fixture-only-provider@9.9.9"
    assert alignment["model_assumptions"] == "fixture-only assumptions"
    assert alignment["ideal_assumptions"]
    assert alignment["scope"]["rank_id"] == "rank_0"
    assert alignment["scope"]["window_bounds_ns"]
    assert alignment["model_input_scope"]["shape"] == [1, 4, 8]
    assert alignment["model_input_scope"]["hardware"] == "fixture-only-npu"
    assert report["enrichment"]["modeling"]["hotspots_modeled"] == 1
    assert any(
        item["modeling_alignment"]["unmodeled_reason"] == "no model inputs supplied for this hotspot"
        for item in report["top_hotspot_kernels"]
        if item["kernel_name"] != "aclnnFlashAttention"
    )


def test_analyze_profile_aicpu_finding_uses_overlap_and_preserves_evidence(tmp_path):
    root = _shifted_core_csv(
        tmp_path,
        offsets=(0,),
        extra_rows=(
            {
                "Device_id": "0",
                "Task ID": "30",
                "Stream ID": "5",
                "Name": "aclnnAicpuKernel",
                "Type": "AicpuTestOp",
                "Accelerator Core": "AI_CPU",
                "Start Time(us)": "50.000\t",
                "Duration(us)": "10.000",
                "Input Shapes": '"4,8"',
                "Input Data Types": "DT_BF16",
                "Input Formats": "ND",
            },
        ),
    )
    report = analyze_profile(root, windows=[Window(0, 60_000, step=4, label="step_4")])

    row = report["window_accounting"][0]
    assert row["aicpu_exposed_ratio"] > 0
    finding = next(item for item in report["findings"] if item["finding_type"] == "aicpu_fallback_exposed")
    scope = finding["evidence_scope"]
    assert scope["rank_id"] == "rank_0"
    assert scope["step"] == 4
    assert "AI_CPU" in scope["task_types"]
    assert "5" in scope["stream_ids"]
    assert "30" in scope["task_ids"]
    assessment = report["enrichment"]["anomaly_discovery"]["aicpu_exposure"]["by_window"][0]["interval_assessment"]
    assert assessment["classification"] == "AICPU_EXPOSED_NOT_ALLOWED"
    assert assessment["counter_status"] == "missing"
    assert "aicpu_pmu_counters_missing" in {item["code"] for item in report["limitations"]}
    assert report["enrichment"]["pmu_observations"]["aicpu_counters"]["counter_status"] == "missing"


def test_analyze_profile_epoch_scale_ns_uses_local_origin_for_float_conversion(tmp_path):
    origin_ns = 1_000_000_000_000_000_000
    root = _shifted_core_csv(tmp_path, offsets=(origin_ns // 1000,))
    report = analyze_profile(
        root,
        windows=[Window(origin_ns, origin_ns + 60_000, step=4, label="step_4")],
    )

    row = report["window_accounting"][0]
    assert row["window_origin_ns"] == origin_ns
    assert row["window_start_ns"] == origin_ns
    assert row["window_end_ns"] == origin_ns + 60_000
    assert row["window_start_us"] == 0.0
    assert row["window_end_us"] == 60.0
    scope = report["enrichment"]["anomaly_discovery"]["wait_anchor"]["evidence_scope"][0]
    assert scope["window_start_ns"] == origin_ns
    assert scope["window_end_ns"] == origin_ns + 60_000
    assert report["top_hotspot_kernels"][0]["modeling_alignment"]["scope"]["window_bounds_ns"][0][0] == origin_ns
    assert report["top_hotspot_kernels"][0]["modeling_alignment"]["scope"]["window_bounds_ns"][0][1] == origin_ns + 60_000
    assert "time_origin_policy" in report


def test_analyze_profile_provider_without_inputs_remains_unmodeled(tmp_path):
    root = _shifted_core_csv(tmp_path, offsets=(0,))
    report = analyze_profile(
        root,
        windows=[Window(0, 60_000, step=4, label="step_4")],
        model_provider=_FixtureProvider(),
    )

    assert report["top_hotspot_kernels"]
    assert all(
        item["modeling_alignment"]["status"] == "unmodeled"
        and item["modeling_alignment"]["model_predicted_ms"] is None
        and "no model inputs supplied" in item["modeling_alignment"]["unmodeled_reason"]
        for item in report["top_hotspot_kernels"]
    )
    assert "model_inputs_missing" in {item["code"] for item in report["limitations"]}


def test_aicpu_actual_counters_are_preserved_and_used_as_limitation_evidence():
    events = [
        ad.KernelEvent(
            "opaque_aicpu", 4, 0, 100, task_type="AI_CPU", event_id="a",
            counters=(("aicpu_cycles", 1234), ("aicpu_time_us", 88.0)),
        ),
        ad.KernelEvent("opaque_core", 4, 0, 80, task_type="AI_CORE", event_id="b"),
    ]

    report = ad.assess_aicpu_exposure(events)

    assert report.status == "assessed"
    assert report.counter_status == "present"
    assert "pmu_counters_missing" not in report.limitations
    assert report.classification == "AICPU_PARTIALLY_EXPOSED"
    assert report.exposed_ratio == pytest.approx(0.2)
