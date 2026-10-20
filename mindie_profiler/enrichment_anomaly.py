"""Interval anomalies, measured wait, and PMU observation scope.

[Role]
  Map core ``Event`` rows into the anomaly-discovery records, attach measured
  wait and unambiguous PMU counters, and emit conservative C7 fragments for
  Wait-Anchor, AICPU exposure, underfeed, and generic PMU observations.

[Invariants]
  * Wait-Anchor uses only a complete event whose wait interval is fully inside
    the observed window.  A clipped long event never contributes its full wait
    to a tiny clipped hotspot.
  * A waitNs/PMU group is bound only when the full-capture TASK occurrence
    count is exactly one and the metric rows are unambiguous.  Repeated
    globalTaskIds stay group-scoped with raw counts; values are never summed,
    averaged, picked, or assigned to a selected occurrence.
  * PMU values are full-event counters.  They are never relabeled as
    clipped-window counts, and AI-core PMU is not treated as AICPU evidence.
  * Missing/ambiguous measurements stay ``None``/``unknown``.

[Timeouts & Locks]
  Pure in-memory passes over already-bounded observations; no I/O or locks.

[Side-Effect Policy]
  Returns new dicts and mutates only the explicitly passed hotspot rows in
  ``attach_wait_anchor``.  No source artifact is read or changed.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .anomaly_discovery import (
    KernelEvent,
    assess_aicpu_exposure,
    assess_wait_anchor,
)
from .events import Event, Observation
from .pmu_derivation import derive_pmu_evidence, metric_units
from .sources_db import load_mapping
from .wait_provenance import attach_wait_anchor

WAIT_ANCHOR_REASON = (
    "no complete unambiguous waitNs/Wait Time event with a fully observed wait "
    "interval was available for Wait-Anchor verification"
)
PMU_REASON = (
    "no AICPU-specific PMU counters were attached to the selected AICPU events; "
    "AI-core PMU is not used as AICPU counter evidence"
)
UNDERFEED_THRESHOLD = 0.30
PMU_SCHEMA = "mindie-pmu-observation/1"


def _duration_ms(value_ns: int | None) -> float | None:
    return None if value_ns is None else round(float(value_ns) / 1_000_000.0, 6)


def _relative_us(value_ns: int, origin_ns: int) -> float:
    return (int(value_ns) - int(origin_ns)) / 1000.0


def add_limitation(
    limitations: list[dict[str, Any]],
    code: str,
    scope: str,
    detail: str,
    severity: str = "medium",
) -> None:
    for item in limitations:
        if item.get("code") == code and item.get("scope") == scope:
            return
    limitations.append({"code": code, "scope": scope, "detail": detail, "severity": severity})


def _finding(
    finding_type: str,
    severity: str,
    confidence: str,
    summary: str,
    evidence_ref: str,
    *,
    scope: Mapping[str, Any] | None = None,
    limitation_code: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "finding_type": finding_type,
        "severity": severity,
        "confidence": confidence,
        "summary": summary,
        "evidence_ref": evidence_ref,
        "evidence_scope": dict(scope or {}),
    }
    if limitation_code is not None:
        payload["limitation_code"] = limitation_code
    return payload


def _explicit_task_type(event: Event) -> str | None:
    engine = str(getattr(event, "execution_engine", "") or "")
    if event.is_aicpu or engine == "aicpu":
        return "AI_CPU"
    if event.is_comm or engine == "hccl":
        return "COMMUNICATION"
    if engine in {"aic", "aiv", "mix"}:
        return "AI_CORE"
    return event.task_type or event.core or None


def _stream_id(event: Event) -> int | None:
    match = re.search(r"\d+", str(event.stream_id or ""))
    return int(match.group(0)) if match else None




def _wait_interval(
    event: Event,
    window,
) -> tuple[float | None, float | None, str | None]:
    """Return (wait_us, wait_start_us, exclusion_reason) for assessment.

    The returned interval is relative to the observation window origin.  Only
    a complete event with a fully observed wait interval is usable.
    """

    wait_ns = event.wait_ns
    if event.wait_status != "bound" or wait_ns is None:
        return None, None, f"wait_status_{event.wait_status}"
    if not event.complete_in(window.start_ns, window.end_ns):
        return None, None, "event_clipped_by_window"
    wait_us = float(wait_ns) / 1000.0
    wait_end_us = _relative_us(event.start_ns, window.start_ns)
    wait_start_us = wait_end_us - wait_us
    if wait_start_us < 0:
        return None, None, "wait_interval_outside_observed_window"
    return wait_us, wait_start_us, None


def _mapped_events(observation: Observation) -> tuple[list[KernelEvent], int, list[Event]]:
    origin_ns = int(observation.window.start_ns)
    observation_id = str(observation.identity()["observation_id"])
    mapped: list[KernelEvent] = []
    core_events: list[Event] = []
    for event in observation.events:
        clipped = event.clipped(observation.window.start_ns, observation.window.end_ns)
        if clipped is None:
            continue
        left, right = clipped
        wait_us, wait_start_us, _reason = _wait_interval(event, observation.window)
        counters: tuple[tuple[str, float], ...] = ()
        if (
            event.complete_in(observation.window.start_ns, observation.window.end_ns)
            and event.pmu_status == "bound"
            and event.pmu
            and event.pmu_kind == "aicore"
            and not event.is_aicpu
            and not event.is_comm
        ):
            counters = event.pmu
        identity = event.identity()
        mapped.append(
            KernelEvent(
                name=event.name or "<missing>",
                step=observation.window.step,
                start_us=_relative_us(left, origin_ns),
                duration_us=_relative_us(right, left),
                wait_us=wait_us,
                task_type=_explicit_task_type(event),
                accelerator_core=event.core or None,
                stream_id=_stream_id(event),
                rank=event.rank_index if event.rank_index is not None else event.rank_id,
                event_id=str(identity["event_id"]),
                wait_start_us=wait_start_us,
                wait_end_us=(None if wait_start_us is None else wait_start_us + (wait_us or 0.0)),
                counters=counters,
                observation_id=observation_id,
                window_start_ns=observation.window.start_ns,
                window_end_ns=observation.window.end_ns,
                window_label=observation.window.label or str(observation.window.step),
                source_kind=observation.source.kind,
                source_sha256=observation.source.sha256,
                source_table=event.table,
                device_id=event.device_id,
                global_task_id=event.locator.get("task_global_id"),
                task_id=event.task_id,
                semantic_block=event.block,
                event_start_ns=event.start_ns,
                event_end_ns=event.end_ns,
                wait_ns=(None if wait_us is None else event.wait_ns),
                wait_start_ns=(
                    None if wait_us is None or event.wait_ns is None else event.start_ns - int(event.wait_ns)
                ),
                wait_end_ns=(None if wait_us is None else event.start_ns),
                event_identity=identity,
            )
        )
        core_events.append(event)
    return mapped, origin_ns, core_events


def _measurement_counts(observations: Sequence[Observation]) -> dict[str, int]:
    counts = {
        "event_count": 0,
        "wait_bound_event_count": 0,
        "wait_usable_event_count": 0,
        "wait_clipped_event_count": 0,
        "wait_outside_window_event_count": 0,
        "wait_ambiguous_event_count": 0,
        "wait_absent_event_count": 0,
        "wait_not_available_event_count": 0,
        "wait_not_applicable_event_count": 0,
        "pmu_bound_event_count": 0,
        "pmu_ambiguous_event_count": 0,
        "pmu_clipped_bound_event_count": 0,
        "aicpu_event_count": 0,
    }
    for observation in observations:
        for event in observation.events:
            counts["event_count"] += 1
            status = event.wait_status
            if status == "bound":
                counts["wait_bound_event_count"] += 1
                wait_us, _start, reason = _wait_interval(event, observation.window)
                if wait_us is not None:
                    counts["wait_usable_event_count"] += 1
                elif reason == "event_clipped_by_window":
                    counts["wait_clipped_event_count"] += 1
                elif reason == "wait_interval_outside_observed_window":
                    counts["wait_outside_window_event_count"] += 1
            elif status == "ambiguous_task_occurrence":
                counts["wait_ambiguous_event_count"] += 1
            elif status in {"absent", "invalid_value"}:
                counts["wait_absent_event_count"] += 1
            elif status == "not_available":
                counts["wait_not_available_event_count"] += 1
            elif status == "not_applicable":
                counts["wait_not_applicable_event_count"] += 1
            if event.pmu_status == "bound":
                counts["pmu_bound_event_count"] += 1
                if not event.complete_in(observation.window.start_ns, observation.window.end_ns):
                    counts["pmu_clipped_bound_event_count"] += 1
            elif event.pmu_status in {
                "ambiguous_task_occurrence",
                "duplicate_metric_rows",
                "incomplete_metric_value",
                "cardinality_unknown",
            }:
                counts["pmu_ambiguous_event_count"] += 1
            if event.is_aicpu:
                counts["aicpu_event_count"] += 1
    return counts


def _evidence_scope(observation: Observation, core_events: Sequence[Event] | None = None) -> dict[str, Any]:
    events = list(core_events if core_events is not None else observation.events)
    streams = sorted({str(event.stream_id) for event in events if event.stream_id})
    task_types = sorted(
        {str(_explicit_task_type(event)) for event in events if _explicit_task_type(event) is not None}
    )
    task_ids = sorted({str(event.task_id) for event in events if event.task_id})
    tables = sorted({str(event.table) for event in events if event.table})
    names = sorted({str(event.name) for event in events if event.name})
    devices = sorted({str(event.device_id) for event in events if event.device_id})
    bound_families = sorted({str(event.bound_family) for event in events if event.bound_family})
    execution_engines = sorted(
        {str(getattr(event, "execution_engine", "unknown") or "unknown") for event in events}
    )
    return {
        "observation_id": observation.identity()["observation_id"],
        "rank_id": observation.rank_id,
        "rank_index": observation.rank_index,
        "step": observation.window.step,
        "window_label": observation.window.label or str(observation.window.step),
        "time_origin_ns": observation.window.start_ns,
        "window_start_ns": observation.window.start_ns,
        "source_kind": observation.source.kind,
        "source_sha256": observation.source.sha256,
        "window_end_ns": observation.window.end_ns,
        "stream_ids": streams,
        "task_types": task_types,
        "reported_task_types": sorted({str(event.task_type) for event in events if event.task_type}),
        "reported_cores": sorted({str(event.core) for event in events if event.core}),
        "task_ids": task_ids[:32],
        "event_tables": tables,
        "kernel_names": names[:16],
        "device_ids": devices[:16],
        "bound_families": bound_families,
        "execution_engines": execution_engines,
        "event_count": len(events),
        "aicpu_event_count": sum(1 for event in events if event.is_aicpu),
        "comm_event_count": sum(1 for event in events if event.is_comm),
        "wait_bound_event_count": sum(1 for event in events if event.wait_status == "bound"),
        "wait_ambiguous_event_count": sum(
            1 for event in events if event.wait_status == "ambiguous_task_occurrence"
        ),
        "hardware_scope": {
            "status": "not_available",
            "reason": "no hardware identity envelope was attached to the observation",
        },
        "pmu_bound_event_count": sum(1 for event in events if event.pmu_status == "bound"),
        "pmu_ambiguous_event_count": sum(
            1
            for event in events
            if event.pmu_status
            in {
                "ambiguous_task_occurrence",
                "duplicate_metric_rows",
                "incomplete_metric_value",
                "cardinality_unknown",
            }
        ),
    }


def wait_anchor_summary(observations: Sequence[Observation]) -> dict[str, Any]:
    scopes = [_evidence_scope(observation) for observation in observations if observation.events]
    counts = _measurement_counts(observations)
    limitations: list[dict[str, Any]] = []
    if counts["wait_ambiguous_event_count"]:
        add_limitation(
            limitations,
            "wait_time_ambiguous_task_occurrence",
            "global",
            "One or more selected events share a globalTaskId with other full-capture TASK "
            "occurrences; their CTI waitNs is group-scoped and was not assigned.",
            "high",
        )
    if counts["wait_clipped_event_count"]:
        add_limitation(
            limitations,
            "wait_time_clipped_event_not_used_for_wait_anchor",
            "global",
            "One or more events carrying waitNs were clipped by the observation window; "
            "their full wait was not used for Wait-Anchor.",
            "high",
        )
    if counts["wait_outside_window_event_count"]:
        add_limitation(
            limitations,
            "wait_interval_outside_observed_window",
            "global",
            "One or more complete events have a wait interval starting before the observation "
            "window; only fully observed wait intervals are used.",
            "medium",
        )
    if not counts["wait_usable_event_count"]:
        add_limitation(limitations, "wait_anchor_unverified", "global", WAIT_ANCHOR_REASON, "medium")
        if counts["wait_absent_event_count"] or counts["wait_not_available_event_count"]:
            add_limitation(
                limitations,
                "wait_time_absent_or_unavailable",
                "global",
                "No explicit waitNs/Wait Time value was present for the selected events.",
                "medium",
            )
        return {
            "status": "unverified",
            "wait_anchor_false_hotspot": None,
            "reason": WAIT_ANCHOR_REASON,
            "assessment": "not_run",
            "findings": [],
            "limitations": limitations,
            "measurement_scope": counts,
            "evidence_scope": scopes,
            "hardware_scope": {
                "status": "not_available",
                "reason": "no hardware identity envelope was attached to the observations",
            },
        }
    mapped: list[KernelEvent] = []
    for observation in observations:
        events, _, _ = _mapped_events(observation)
        mapped.extend(events)
    report = assess_wait_anchor(mapped)
    overlap_missing = 0
    no_individual = 0
    for item in report.limitations:
        text = str(item)
        if text.startswith("missing_wait_time"):
            continue
        if text.startswith("wait_anchor_overlap_missing:"):
            overlap_missing += 1
            continue
        if text.startswith("wait_anchor_no_individual_candidate:"):
            no_individual += 1
            continue
        add_limitation(limitations, text, "global", "Wait-Anchor limitation", "medium")
    if overlap_missing:
        add_limitation(
            limitations,
            "wait_anchor_overlap_missing",
            "global",
            f"{overlap_missing} Wait-Anchor candidate group(s) lacked a same-scope overlapping event; no false-hotspot flag was emitted for them.",
            "medium",
        )
    if no_individual:
        add_limitation(
            limitations,
            "wait_anchor_no_individual_candidate",
            "global",
            f"{no_individual} Wait-Anchor group candidate(s) had no individual event meeting the strict wait-ratio and tiny-duration thresholds; no group result was attached to an individual hotspot.",
            "high",
        )
    raw_findings = report.as_c7_list()
    findings: list[dict[str, Any]] = []
    for item in raw_findings:
        confirmed = bool(item.get("is_false_hotspot_risk"))
        candidate_ids = list(item.get("candidate_event_ids") or [])
        event_ref = candidate_ids[0] if candidate_ids else f"wait_anchor:{item.get('location')}:{item.get('op_name')}"
        findings.append(
            {
                **item,
                "finding_type": "wait_anchor_masked" if confirmed else "wait_anchor_insufficient",
                "severity": "high" if confirmed else "medium",
                "confidence": "high" if confirmed else "low",
                "summary": (
                    f"Wait-Anchor candidate {item.get('op_name')} has wait ratio "
                    f"{item.get('wait_ratio')} and insufficient correlated overlap."
                    if not confirmed
                    else f"Wait-Anchor candidate {item.get('op_name')} overlaps another "
                    "same-scope event while carrying high wait time."
                ),
                "evidence_ref": f"wait_anchor:{event_ref}",
            }
        )
    confirmed = any(item.get("is_false_hotspot_risk") for item in raw_findings)
    return {
        "status": report.status,
        "wait_anchor_false_hotspot": True if confirmed else None,
        "reason": None,
        "assessment": report.status,
        "findings": findings,
        "limitations": limitations,
        "measurement_scope": counts,
        "evidence_scope": scopes,
        "hardware_scope": {
            "status": "not_available",
            "reason": "no hardware identity envelope was attached to the observations",
        },
    }


def aicpu_summary(observations: Sequence[Observation]) -> dict[str, Any]:
    by_window: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    limitations: list[dict[str, Any]] = []
    for observation in observations:
        accounting = observation.accounting
        if not accounting.available or accounting.aicpu_event_count <= 0:
            continue
        mapped, origin_ns, core_events = _mapped_events(observation)
        assessment = assess_aicpu_exposure(mapped, mask_all_non_aicpu=True)
        assessment_payload = assessment.as_c7_dict()
        assessment_payload["counter_scope"] = "aicpu_events_only"
        assessment_payload["counter_source"] = (
            "AICPU-specific event counters" if assessment.counter_status == "present" else None
        )
        if assessment.counter_status != "present":
            assessment_payload["counter_reason"] = PMU_REASON
            add_limitation(
                limitations,
                "aicpu_pmu_counters_missing",
                observation.rank_id,
                PMU_REASON,
                "medium",
            )
        core_exposed_ms = _duration_ms(accounting.exposed_aicpu_ns)
        core_ratio = accounting.aicpu_exposed_ratio
        non_aicpu_count = max(0, accounting.event_count - accounting.aicpu_event_count)
        scope = _evidence_scope(observation, core_events)
        scope["time_origin_ns"] = origin_ns
        scope["float_conversion"] = "relative_ns_to_us"
        supported = core_exposed_ms is not None and core_exposed_ms > 0
        if supported and non_aicpu_count > 0:
            severity = "high" if (core_ratio or 0.0) >= 0.20 else "medium"
            findings.append(
                _finding(
                    "aicpu_fallback_exposed",
                    severity,
                    "high",
                    (
                        f"Core accounting measured {core_exposed_ms:.6f} ms AICPU exposure "
                        f"(ratio {core_ratio}); interval assessment classification is "
                        f"{assessment.classification or 'unavailable'}, counter status is "
                        f"{assessment.counter_status}. This is interval exposure evidence, "
                        "not a PMU-counter hardware diagnosis."
                    ),
                    f"aicpu:{observation.rank_id}:{observation.window.step}:{observation.window.label}",
                    scope=scope,
                )
            )
        elif supported:
            add_limitation(
                limitations,
                "aicpu_overlap_reference_missing",
                observation.rank_id,
                "AICPU exposure was measured but no non-AICPU event exists for overlap attribution",
                "medium",
            )
        for item in assessment_payload.get("limitations", []):
            if item == "pmu_counters_missing":
                continue
            add_limitation(
                limitations,
                str(item),
                observation.rank_id,
                "AICPU interval assessment limitation",
                "medium",
            )
        by_window.append(
            {
                "scope": scope,
                "core_accounting": {
                    "aicpu_event_count": accounting.aicpu_event_count,
                    "exposed_aicpu_ms": core_exposed_ms,
                    "aicpu_exposed_ratio": core_ratio,
                },
                "interval_assessment": assessment_payload,
                "supported_exposure": bool(supported and non_aicpu_count > 0),
            }
        )
    return {
        "status": "assessed" if by_window else "not_applicable",
        "by_window": by_window,
        "findings": findings,
        "limitations": limitations,
    }


def underfeed_summary(observations: Sequence[Observation]) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    for observation in observations:
        accounting = observation.accounting
        ratio = accounting.underfeed_ratio
        if not accounting.available or ratio is None or ratio < UNDERFEED_THRESHOLD:
            continue
        scope = _evidence_scope(observation)
        windows.append(
            {
                "scope": scope,
                "underfeed_bubble_ratio": ratio,
                "wall_ms": _duration_ms(accounting.wall_ns),
                "busy_union_ms": _duration_ms(accounting.busy_union_ns),
                "host_gap_ms": _duration_ms(accounting.host_gap_ns),
            }
        )
        findings.append(
            _finding(
                "underfeed_bubble",
                "high" if ratio >= 0.5 else "medium",
                "high",
                (
                    f"Core accounting measured host_gap/wall underfeed ratio {ratio:.6f}; "
                    "this is a device-idle proxy, not a host-dispatch attribution."
                ),
                f"underfeed:{observation.rank_id}:{observation.window.step}:{observation.window.label}",
                scope=scope,
            )
        )
    return {"threshold": UNDERFEED_THRESHOLD, "windows": windows, "findings": findings}


def _pmu_event_payload(observation: Observation, event: Event) -> dict[str, Any]:
    complete = event.complete_in(observation.window.start_ns, observation.window.end_ns)
    clipped = event.clipped(observation.window.start_ns, observation.window.end_ns)
    coverage = None
    if clipped is not None and event.duration_ns > 0:
        coverage = round((clipped[1] - clipped[0]) / event.duration_ns, 6)
    provenance = {
        key: value
        for key, value in event.pmu_provenance.items()
        if key != "per_metric"
    }
    metrics = event.pmu_map()
    return {
        "rank_id": observation.rank_id,
        "step": observation.window.step,
        "window_label": observation.window.label or str(observation.window.step),
        "observation_id": observation.identity()["observation_id"],
        "kernel_name": event.name,
        "task_global_id": event.locator.get("task_global_id"),
        "event_id": event.event_id,
        "row_index": event.locator.get("row_index"),
        "source_table": event.table,
        "source": event.pmu_source,
        "complete_in_window": complete,
        "window_coverage": coverage,
        "counts_are_full_event": True,
        "usable_for_clipped_window": complete,
        "metrics": metrics,
        "metric_units": metric_units(metrics),
        "metric_count": len(event.pmu),
        "derived": derive_pmu_evidence(
            metrics,
            task_type=event.task_type,
            core=event.core,
            complete_in_window=complete,
            source=event.pmu_source or event.table,
            block_num=None,
            export_resolution=event.pmu_provenance.get("export_resolution"),
        ),
        "provenance": provenance,
    }


def pmu_summary(observations: Sequence[Observation]) -> dict[str, Any]:
    mapping = load_mapping()
    units = dict(mapping.get("pmu_units", {}))
    metric_unit_map = metric_units(
        {name: 0.0 for name in mapping.get("pmu_metrics", {})}
    )
    bound_events: list[dict[str, Any]] = []
    ambiguous_groups: dict[tuple[str, int], dict[str, Any]] = {}
    counts = {
        "bound_event_count": 0,
        "ambiguous_event_count": 0,
        "clipped_bound_event_count": 0,
        "absent_event_count": 0,
        "not_applicable_event_count": 0,
        "aicpu_event_count": 0,
        "aicpu_events_with_counters": 0,
        "raw_pmu_rows": 0,
    }
    limitations: list[dict[str, Any]] = []
    source_pmu_rows = sum(observation.stats.pmu_rows for observation in observations)
    source_ambiguous_groups = max(
        (observation.stats.ambiguous_groups for observation in observations), default=0
    )
    for observation in observations:
        for event in observation.events:
            if event.is_aicpu:
                counts["aicpu_event_count"] += 1
                if event.pmu_status == "bound" and event.pmu_kind == "aicpu" and event.pmu:
                    counts["aicpu_events_with_counters"] += 1
                elif event.pmu_status == "not_applicable":
                    counts["not_applicable_event_count"] += 1
                continue
            if event.is_comm:
                if event.pmu_status == "not_applicable":
                    counts["not_applicable_event_count"] += 1
                continue
            if event.pmu_status == "bound" and event.pmu:
                counts["bound_event_count"] += 1
                counts["raw_pmu_rows"] += int(event.pmu_provenance.get("raw_rows") or len(event.pmu))
                complete = event.complete_in(observation.window.start_ns, observation.window.end_ns)
                if not complete:
                    counts["clipped_bound_event_count"] += 1
                if len(bound_events) < 64:
                    bound_events.append(_pmu_event_payload(observation, event))
            elif event.pmu_status in {
                "ambiguous_task_occurrence",
                "duplicate_metric_rows",
                "incomplete_metric_value",
                "cardinality_unknown",
            }:
                counts["ambiguous_event_count"] += 1
                provenance = event.pmu_provenance
                global_id = provenance.get("global_task_id")
                if isinstance(global_id, int):
                    key = (observation.rank_id, global_id)
                    group = ambiguous_groups.get(key)
                    if group is None:
                        group = {
                            **provenance,
                            "rank_id": observation.rank_id,
                            "observation_id": observation.identity()["observation_id"],
                            "status": event.pmu_status,
                            "metric_units": metric_units(
                                {
                                    name: 0.0
                                    for name in (provenance.get("per_metric") or {})
                                }
                            ),
                        }
                        ambiguous_groups[key] = group
                        counts["raw_pmu_rows"] += int(provenance.get("raw_rows") or 0)
            elif event.pmu_status == "absent":
                counts["absent_event_count"] += 1
            elif event.pmu_status == "not_applicable":
                counts["not_applicable_event_count"] += 1
    if counts["ambiguous_event_count"]:
        add_limitation(
            limitations,
            "pmu_ambiguous_task_occurrence",
            "global",
            "One or more selected globalTaskIds repeat across full capture; TASK_PMU_INFO "
            "rows are group-scoped and were not assigned, summed, averaged, or picked for "
            "an individual TASK occurrence.",
            "high",
        )
    if counts["clipped_bound_event_count"]:
        add_limitation(
            limitations,
            "pmu_full_event_values_not_clipped_window",
            "global",
            "Some bound PMU values belong to events clipped by the observation window. "
            "They remain full-event counters and are not relabeled as clipped-window counts.",
            "medium",
        )
    if counts["aicpu_event_count"] and not counts["aicpu_events_with_counters"]:
        add_limitation(limitations, "aicpu_pmu_counters_missing", "global", PMU_REASON, "medium")
    if not counts["bound_event_count"] and not ambiguous_groups and counts["absent_event_count"]:
        add_limitation(
            limitations,
            "pmu_values_absent",
            "global",
            "PMU columns exist but selected events carry no numeric PMU values; counters stay unknown.",
            "medium",
        )
    if not counts["bound_event_count"] and not ambiguous_groups and not counts["absent_event_count"]:
        add_limitation(
            limitations,
            "pmu_not_available",
            "global",
            "No PMU table/columns or bound PMU values were available for the selected events.",
            "medium",
        )
    status = "observed" if counts["bound_event_count"] or ambiguous_groups else "not_available"
    if status == "observed" and counts["ambiguous_event_count"]:
        status = "partial"
    return {
        "schema_version": PMU_SCHEMA,
        "status": status,
        "attribution": "measurement_scope_only",
        "derived_attribution": "rule_candidate_not_root_cause",
        "counts_are_full_event": True,
        "scope": "bound/ambiguous non-AICPU, non-communication events; counters are full-event values",
        "raw_rows_scope": "source_stats.pmu_rows counts every PMU row read for selected compute ids; raw_pmu_rows counts bound/ambiguous assessable events only",
        "units": units,
        "metric_units": metric_unit_map,
        "counts": counts,
        "source_stats": {
            "pmu_rows": source_pmu_rows,
            "ambiguous_groups": source_ambiguous_groups,
        },
        "bound_events": bound_events,
        "bound_events_truncated": counts["bound_event_count"] > len(bound_events),
        "ambiguous_groups": list(ambiguous_groups.values()),
        "aicpu_counters": {
            "aicpu_event_count": counts["aicpu_event_count"],
            "counter_status": (
                "present"
                if counts["aicpu_events_with_counters"]
                else ("missing" if counts["aicpu_event_count"] else "not_applicable")
            ),
            "counter_source": "AICPU-specific event counters only",
        },
        "limitations": limitations,
    }




__all__ = [
    "PMU_REASON",
    "PMU_SCHEMA",
    "UNDERFEED_THRESHOLD",
    "WAIT_ANCHOR_REASON",
    "add_limitation",
    "aicpu_summary",
    "attach_wait_anchor",
    "pmu_summary",
    "underfeed_summary",
    "wait_anchor_summary",
]
