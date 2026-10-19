"""Analyze-to-C7 enrichment orchestration.

[Role]
  Attach explicit scene/steady suggestions, measured Wait-Anchor/PMU/AICPU
  fragments, and optional hotspot modeling to the C7 report.  Core accounting
  values remain read-only inputs; this module never recomputes them.

[Invariants]
  * Steady suggestions use only observations carrying an explicit step id;
    user-selected windows are reported, not replaced.
  * Wait-Anchor/PMU binding rules live in ``enrichment_anomaly``: ambiguous
    full-capture task occurrences and clipped events stay unknown/group-scoped.
  * Modeling predictions stay separate from measured and ideal values.
  * Every emitted scope preserves rank/step/window/stream/table/task evidence.

[Timeouts & Locks]
  Pure in-memory work over already-bounded observations.  No I/O, locks,
  subprocesses, network, or source re-reads.

[Side-Effect Policy]
  Mutates only the caller-owned report/rows passed to ``enrich_report``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .enrichment_anomaly import (
    PMU_REASON,
    PMU_SCHEMA,
    UNDERFEED_THRESHOLD,
    WAIT_ANCHOR_REASON,
    add_limitation,
    aicpu_summary,
    attach_wait_anchor,
    pmu_summary,
    underfeed_summary,
    wait_anchor_summary,
)
from .enrichment_model import _coerce_hardware, _normalize_registry, apply_modeling
from .events import Observation
from .scene_detector import (
    StepSample,
    SteadyStateConfig,
    detect_scene,
    detect_steady_window,
)

ENRICHMENT_SCHEMA = "mindie-profile-enrichment/1"


def _duration_ms(value_ns: int | None) -> float | None:
    return None if value_ns is None else round(float(value_ns) / 1_000_000.0, 6)


def _scene_detection(scene: Mapping[str, Any] | None) -> dict[str, Any]:
    return detect_scene(scene or {}).as_dict()


def _steady_suggestion(
    observations: Sequence[Observation],
    requested_steps: Sequence[int],
    config: Any = None,
) -> dict[str, Any]:
    if isinstance(config, Mapping):
        try:
            steady_config = SteadyStateConfig(**dict(config))
        except (TypeError, ValueError) as exc:
            return {
                "basis": "explicit_step_samples_from_observation_accounting",
                "suggested_only": True,
                "filters_user_windows": False,
                "status": "uncertain",
                "steady_steps": [],
                "reason": f"invalid_config:{type(exc).__name__}",
                "requested_steps": list(requested_steps),
                "sample_metric": "wall_ms",
                "by_rank": [],
                "limitations": [],
            }
    elif isinstance(config, SteadyStateConfig):
        steady_config = config
    else:
        steady_config = SteadyStateConfig()
    by_rank: list[dict[str, Any]] = []
    grouped: dict[str, list[Observation]] = {}
    for observation in observations:
        if observation.window.step is None or not observation.accounting.available:
            continue
        grouped.setdefault(observation.rank_id, []).append(observation)
    limitations: list[dict[str, Any]] = []
    for rank_id in sorted(grouped):
        samples: list[StepSample] = []
        rank_index: int | None = None
        for observation in grouped[rank_id]:
            if observation.rank_index is not None:
                rank_index = observation.rank_index
            accounting = observation.accounting
            samples.append(
                StepSample(
                    step=observation.window.step,
                    wall_ms=_duration_ms(accounting.wall_ns),
                    busy_union_ms=_duration_ms(accounting.busy_union_ns),
                    kernel_sum_ms=_duration_ms(accounting.kernel_sum_ns),
                    exposed_comm_ms=_duration_ms(accounting.exposed_comm_ns),
                    host_gap_ms=_duration_ms(accounting.host_gap_ns),
                    total_cost_ms=_duration_ms(accounting.total_cost_ns),
                    rank=rank_index,
                )
            )
        window = detect_steady_window(samples, config=steady_config)
        by_rank.append(
            {
                "rank_id": rank_id,
                "rank_index": rank_index,
                "sample_count": len(samples),
                "window": window.as_c7_dict(),
            }
        )
        if window.status != "steady":
            add_limitation(
                limitations, "steady_suggestion_uncertain", rank_id,
                "explicit step samples did not produce a steady suffix: " + str(window.uncertainty_reason),
                "medium",
            )
    if not by_rank:
        return {
            "basis": "explicit_step_samples_from_observation_accounting",
            "suggested_only": True,
            "filters_user_windows": False,
            "status": "not_applicable",
            "steady_steps": [],
            "reason": "no_explicit_step_samples",
            "requested_steps": list(requested_steps),
            "sample_metric": "wall_ms",
            "by_rank": [],
            "limitations": limitations,
        }
    rank_windows = [row["window"] for row in by_rank]
    if all(item["status"] == "steady" for item in rank_windows):
        common = set(rank_windows[0]["steady_steps"])
        for item in rank_windows[1:]:
            common.intersection_update(item["steady_steps"])
        steps = sorted(int(step) for step in common)
        status = "steady" if steps else "uncertain"
        reason = None if steps else "no_common_steady_steps_across_ranks"
    else:
        steps = []
        status = "uncertain"
        reason = "rank_window_uncertain"
    return {
        "basis": "explicit_step_samples_from_observation_accounting",
        "suggested_only": True,
        "filters_user_windows": False,
        "status": status,
        "steady_steps": steps,
        "reason": reason,
        "requested_steps": list(requested_steps),
        "sample_metric": "wall_ms",
        "by_rank": by_rank,
        "limitations": limitations,
    }


def _merge_findings(report: dict[str, Any], additions: Sequence[Mapping[str, Any]]) -> None:
    findings = report.setdefault("findings", [])
    seen = {
        (
            item.get("finding_type"),
            item.get("evidence_ref"),
            item.get("limitation_code"),
            item.get("summary"),
        )
        for item in findings
    }
    for item in additions:
        key = (
            item.get("finding_type"),
            item.get("evidence_ref"),
            item.get("limitation_code"),
            item.get("summary"),
        )
        if key not in seen:
            findings.append(dict(item))
            seen.add(key)


def _merge_limitations_and_findings(
    report: dict[str, Any],
    additions: Sequence[dict[str, Any]],
) -> None:
    limitations = report.setdefault("limitations", [])
    findings: list[dict[str, Any]] = []
    for item in additions:
        code = str(item.get("code"))
        scope = str(item.get("scope"))
        if any(existing.get("code") == code and existing.get("scope") == scope for existing in limitations):
            continue
        limitations.append(dict(item))
        findings.append(
            {
                "finding_type": "measurement_limitation",
                "severity": str(item.get("severity", "medium")),
                "confidence": "high",
                "summary": str(item.get("detail", "")),
                "evidence_ref": code,
                "evidence_scope": {"scope": scope},
                "limitation_code": code,
            }
        )
    _merge_findings(report, findings)


def enrich_report(
    report: dict[str, Any],
    *,
    observations: Sequence[Observation],
    requested_steps: Sequence[int],
    scene: Mapping[str, Any] | None = None,
    model_provider: Any = None,
    model_inputs: Any = None,
    model_hardware: Any = None,
    steady_config: Any = None,
) -> dict[str, Any]:
    """Attach scene, steady, measured anomaly/PMU and optional modeling fragments."""

    hardware, hardware_error = _coerce_hardware(model_hardware)
    registry, registry_limitations = _normalize_registry(model_inputs, hardware)
    scene_detection = _scene_detection(scene)
    steady = _steady_suggestion(observations, requested_steps, steady_config)
    wait_anchor = wait_anchor_summary(observations)
    aicpu = aicpu_summary(observations)
    underfeed = underfeed_summary(observations)
    pmu = pmu_summary(observations)
    modeling = apply_modeling(
        report.get("top_hotspot_kernels", []),
        provider=model_provider,
        registry=registry,
        hardware=hardware,
        hardware_error=hardware_error,
    )
    attach_wait_anchor(report.get("top_hotspot_kernels", []), wait_anchor)

    anomaly_limitations: list[dict[str, Any]] = []
    for item in wait_anchor.get("limitations", []):
        if isinstance(item, dict):
            anomaly_limitations.append(item)
        else:
            add_limitation(anomaly_limitations, str(item), "global", "Wait-Anchor limitation", "medium")
    anomaly_limitations.extend(aicpu.get("limitations", []))
    anomaly_limitations.extend(pmu.get("limitations", []))
    anomaly_limitations.extend(steady.get("limitations", []))
    anomaly_limitations.extend(registry_limitations)
    anomaly_limitations.extend(modeling.get("limitations", []))
    _merge_limitations_and_findings(report, anomaly_limitations)
    _merge_findings(
        report,
        [*wait_anchor.get("findings", []), *aicpu.get("findings", []), *underfeed.get("findings", [])],
    )
    report["scene_detection"] = scene_detection
    report["steady_window_suggestion"] = steady
    report["enrichment"] = {
        "schema_version": ENRICHMENT_SCHEMA,
        "time_precision": {
            "event_time_unit": "ns",
            "wait_time_unit": "ns (DB waitNs); CSV Wait Time(us) converted to ns once",
            "pmu_time_unit": "ns full-event values (CSV us time columns converted to ns once)",
            "pmu_ratio_unit": "fraction 0..1",
            "absolute_values": "kept as integer ns in scopes/window bounds",
            "float_conversion": "per-observation local window origin subtracted before ns->us; durations converted independently",
        },
        "window_filtering": "none; user-selected windows are reported unchanged",
        "scene_detection": scene_detection,
        "steady_window_suggestion": steady,
        "anomaly_discovery": {
            "wait_anchor": wait_anchor,
            "aicpu_exposure": aicpu,
            "underfeed": underfeed,
            "limitations": anomaly_limitations,
        },
        "pmu_observations": pmu,
        "modeling": modeling,
    }
    return report


__all__ = [
    "ENRICHMENT_SCHEMA",
    "PMU_REASON",
    "PMU_SCHEMA",
    "UNDERFEED_THRESHOLD",
    "WAIT_ANCHOR_REASON",
    "enrich_report",
]
