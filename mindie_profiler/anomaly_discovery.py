"""Interval-based anomaly discovery and an honest C7 summary fragment.

[Role]
Consumes measured step samples and kernel events. Computes interval unions,
overlap/exposure and the C7 step timing views; assesses Wait-Anchor false
hotspots and AICPU exposure only when event association plus interval overlap
support them. Emits a plain C7 (`mindie-profile-summary/1`) fragment that the
summary owner can merge without importing a controller/core API.

[Invariants]
- Wait-Anchor is flagged only when wait ratio, tiny duration and top total-cost
  rank hold AND a correlated event overlaps the derived wait interval. A name
  alone can never create the flag.
- AICPU exposure is derived from explicit task type plus AICPU/AI-Core interval
  overlap. Missing task type or reference core events yields uncertainty.
- Missing PMU counters only add a measurement limitation; they never become a
  claimed hardware bottleneck or a renamed root cause.
- No numeric value is invented for a missing view; the limitation is reported
  and the C7 field is `None` instead of zero/guesswork.

[Timeouts & Locks]
- Pure computation. No locks, sleeps, I/O, subprocesses or network.

[Side-Effect Policy]
- Read-only over caller-owned samples/events; returns fresh tuples and dicts.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .scene_detector import StepSample, SteadyWindow, coerce_step_sample
from .wait_provenance import WaitOverlapIndex, event_error, event_scope, overlap_scope, valid_events

AI_CPU = "aicpu"
AI_CORE = "aicore"
COMM = "comm"
OTHER = "other"
ASSESSED = "assessed"
NOT_APPLICABLE = "not_applicable"
UNCERTAIN = "uncertain"
CONFIRMED = "confirmed"
INSUFFICIENT = "insufficient"

Interval = tuple[float, float]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _valid_interval(value: Sequence[float]) -> Interval:
    if len(value) != 2:
        raise ValueError("interval_must_have_two_endpoints")
    start, end = float(value[0]), float(value[1])
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError("invalid_interval")
    return start, end


def merge_intervals(intervals: Sequence[Sequence[float]]) -> list[Interval]:
    """Merge half-open device intervals; touching intervals are coalesced."""
    items: list[Interval] = []
    for item in intervals:
        start, end = _valid_interval(item)
        if end > start:
            items.append((start, end))
    items.sort()
    if not items:
        return []
    merged: list[Interval] = [items[0]]
    for start, end in items[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def union_ms(intervals: Sequence[Sequence[float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals)) / 1000.0


def intersection_ms(target: Sequence[float], others: Sequence[Sequence[float]]) -> float:
    start, end = _valid_interval(target)
    if end <= start:
        return 0.0
    clipped: list[Interval] = []
    for other in others:
        other_start, other_end = _valid_interval(other)
        left, right = max(start, other_start), min(end, other_end)
        if right > left:
            clipped.append((left, right))
    return union_ms(clipped)


def overlap_ratio(target: Sequence[float], others: Sequence[Sequence[float]]) -> float | None:
    start, end = _valid_interval(target)
    duration = end - start
    if duration <= 0:
        return None
    return intersection_ms(target, others) * 1000.0 / duration


def _pairs(value: Any) -> tuple[tuple[str, float], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), float(v)) for k, v in value.items() if _finite(v)))
    return tuple((str(k), float(v)) for k, v in value if _finite(v))


@dataclass(frozen=True)
class KernelEvent:
    name: str
    step: int | None
    start_us: float
    duration_us: float
    wait_us: float | None = None
    task_type: str | None = None
    accelerator_core: str | None = None
    stream_id: int | None = None
    rank: int | None = None
    event_id: str | None = None
    related_event_ids: tuple[str, ...] = ()
    wait_start_us: float | None = None
    wait_end_us: float | None = None
    counters: tuple[tuple[str, float], ...] = ()
    observation_id: str | None = None
    window_start_ns: int | None = None
    window_end_ns: int | None = None
    window_label: str | None = None
    source_kind: str | None = None
    source_sha256: str | None = None
    source_table: str | None = None
    device_id: str | None = None
    global_task_id: int | None = None
    task_id: str | None = None
    semantic_block: str | None = None
    event_start_ns: int | None = None
    event_end_ns: int | None = None
    wait_ns: int | None = None
    wait_start_ns: int | None = None
    wait_end_ns: int | None = None
    hardware_scope: Mapping[str, Any] | None = None
    event_identity: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        related = self.related_event_ids or ()
        if isinstance(related, str):
            related = (related,)
        object.__setattr__(self, "related_event_ids", tuple(str(item) for item in related))
        object.__setattr__(self, "counters", _pairs(self.counters))
        object.__setattr__(self, "hardware_scope", dict(self.hardware_scope) if self.hardware_scope else None)
        object.__setattr__(self, "event_identity", dict(self.event_identity) if self.event_identity else None)

    @property
    def end_us(self) -> float:
        return self.start_us + self.duration_us

    @property
    def interval(self) -> Interval:
        return self.start_us, self.end_us

    @property
    def wait_interval(self) -> Interval | None:
        if self.wait_us is None or self.wait_us <= 0:
            return None
        if self.wait_start_us is not None and self.wait_end_us is not None:
            start, end = float(self.wait_start_us), float(self.wait_end_us)
            return (start, end) if end > start else None
        if self.wait_start_us is not None:
            return float(self.wait_start_us), float(self.wait_start_us) + float(self.wait_us)
        if self.wait_end_us is not None:
            return float(self.wait_end_us) - float(self.wait_us), float(self.wait_end_us)
        return max(0.0, self.start_us - float(self.wait_us)), self.start_us

    @property
    def task_kind(self) -> str | None:
        for raw in (self.task_type, self.accelerator_core):
            if raw is None:
                continue
            key = str(raw).strip().upper().replace("-", "_").replace(" ", "_")
            if not key:
                continue
            if "AI_CPU" in key or "AICPU" in key:
                return AI_CPU
            if "AI_CORE" in key or "AICORE" in key:
                return AI_CORE
            if "HCCL" in key or "COMM" in key:
                return COMM
            return OTHER
        return None






@dataclass(frozen=True)
class WaitAnchorReport:
    status: str
    findings: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...]

    @property
    def confirmed_findings(self) -> tuple[dict[str, Any], ...]:
        return tuple(item for item in self.findings if item["is_false_hotspot_risk"])

    def as_c7_list(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.findings]




def assess_wait_anchor(
    events: Sequence[KernelEvent],
    *,
    top_n: int = 10,
    min_overlap_ratio: float = 0.2,
    tiny_duration_us: float = 10.0,
) -> WaitAnchorReport:
    """Assess Wait-Anchor candidates; no event overlap means no false-hotspot flag."""
    valid, limitations = valid_events(events or ())
    if not valid:
        return WaitAnchorReport(UNCERTAIN, (), tuple(limitations + ["no_valid_kernel_events"]))
    if not _finite(min_overlap_ratio) or not 0 <= float(min_overlap_ratio) <= 1:
        limitations.append("invalid_min_overlap_ratio")
        min_overlap_ratio = 0.2
    top_n = max(1, int(top_n))
    groups: dict[tuple[Any, Any, Any, Any, str], list[KernelEvent]] = {}
    for event in valid:
        groups.setdefault(
            (event.observation_id, event.rank, event.step, event.semantic_block, event.name),
            [],
        ).append(event)
    rows: list[dict[str, Any]] = []
    for (observation_id, rank, step, semantic_block, name), group in groups.items():
        timed = [event for event in group if event.wait_us is not None]
        if not timed:
            limitations.append(f"missing_wait_time:{name}")
            continue
        if len(timed) != len(group):
            limitations.append(f"missing_wait_time_for_some_events:{name}")
        duration_avg = sum(event.duration_us for event in timed) / len(timed)
        wait_avg = sum(float(event.wait_us) for event in timed) / len(timed)
        total_cost = duration_avg + wait_avg
        ratio = wait_avg / total_cost if total_cost > 0 else 0.0
        rows.append({
            "op_name": name,
            "location": f"rank:{rank}" if rank is not None else None,
            "duration_us_avg": duration_avg,
            "wait_us_avg": wait_avg,
            "total_cost_us_avg": total_cost,
            "wait_ratio": ratio,
            "_rank": rank,
            "_step": step,
            "_observation_id": observation_id,
            "_semantic_block": semantic_block,
            "_group": timed,
        })
    if not rows:
        return WaitAnchorReport(
            UNCERTAIN, (), tuple(dict.fromkeys(limitations + ["missing_wait_time_for_all_events"]))
        )
    rows.sort(key=lambda row: row["total_cost_us_avg"], reverse=True)
    overlap_index = WaitOverlapIndex(valid)
    findings: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        candidate = (
            row["wait_ratio"] > 0.95
            and row["duration_us_avg"] < float(tiny_duration_us)
            and index < top_n
        )
        if not candidate:
            continue
        evidence: list[str] = []
        candidate_scopes: list[dict[str, Any]] = []
        confirmed_scopes: list[dict[str, Any]] = []
        overlap_pairs: list[dict[str, Any]] = []
        confirmed_ids: list[str] = []
        individual_candidates = 0
        for event in row["_group"]:
            wait_interval = event.wait_interval
            if wait_interval is None:
                continue
            scope = event_scope(event)
            candidate_scopes.append(scope)
            event_wait = float(event.wait_us)
            event_total = float(event.duration_us) + event_wait
            event_ratio = event_wait / event_total if event_total > 0 else 0.0
            if event_ratio <= 0.95 or float(event.duration_us) >= float(tiny_duration_us):
                continue
            individual_candidates += 1
            measured = overlap_index.find(
                event,
                wait_interval,
                min_overlap_ratio=float(min_overlap_ratio),
            )
            if measured is None:
                continue
            other, ratio, overlap_ns = measured
            pair = overlap_scope(event, other, ratio, overlap_ns=overlap_ns)
            if len(overlap_pairs) < 16:
                overlap_pairs.append(pair)
            identity = other.event_id or other.name
            if len(evidence) < 16:
                evidence.append(f"wait_overlap:{ratio:.3f}:event={identity}")
            confirmed_scopes.append(scope)
            confirmed_ids.append(str(scope["event_id"]))
        confirmed = bool(confirmed_ids)
        if confirmed:
            evidence_status, reason = CONFIRMED, None
        else:
            evidence_status = INSUFFICIENT
            if individual_candidates == 0:
                reason = "no_individual_event_met_wait_ratio_and_tiny_duration"
                limitations.append(f"wait_anchor_no_individual_candidate:{row['op_name']}")
            else:
                reason = "missing_correlated_event_overlap_for_wait_interval"
                limitations.append(f"wait_anchor_overlap_missing:{row['op_name']}")
        findings.append({
            "op_name": row["op_name"],
            "location": row["location"],
            "step": row["_step"],
            "observation_id": row["_observation_id"],
            "semantic_block": row["_semantic_block"],
            "duration_us_avg": row["duration_us_avg"],
            "wait_us_avg": row["wait_us_avg"],
            "total_cost_us_avg": row["total_cost_us_avg"],
            "wait_ratio": row["wait_ratio"],
            "is_false_hotspot_risk": confirmed,
            "wait_anchor_false_hotspot": True if confirmed else None,
            "evidence_status": evidence_status,
            "evidence": evidence,
            "reason": reason,
            "candidate_event_count": len(candidate_scopes),
            "candidate_event_ids": [str(item["event_id"]) for item in candidate_scopes[:64]],
            "candidate_events": candidate_scopes[:16],
            "confirmed_event_count": len(confirmed_ids),
            "confirmed_event_ids": confirmed_ids[:64],
            "confirmed_event_ids_truncated": len(confirmed_ids) > 64,
            "confirmed_events": confirmed_scopes[:16],
            "overlap_pairs": overlap_pairs,
            "candidate_event_ids_truncated": len(candidate_scopes) > 64,
            "hardware_scope": (
                candidate_scopes[0].get("hardware_scope") if candidate_scopes else None
            ),
        })
    return WaitAnchorReport("available", tuple(findings), tuple(dict.fromkeys(limitations)))


@dataclass(frozen=True)
class AicpuExposureReport:
    status: str
    aicpu_duration_ms: float | None
    masked_ms: float | None
    exposed_ms: float | None
    masked_ratio: float | None
    exposed_ratio: float | None
    classification: str | None
    counter_status: str
    limitations: tuple[str, ...]
    evidence: tuple[str, ...]

    def as_c7_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "aicpu_duration_ms": self.aicpu_duration_ms,
            "masked_ms": self.masked_ms,
            "exposed_ms": self.exposed_ms,
            "masked_ratio": self.masked_ratio,
            "aicpu_exposed_ratio": self.exposed_ratio,
            "classification": self.classification,
            "counter_status": self.counter_status,
            "attribution": "exposure_only" if self.classification else None,
            "limitations": list(self.limitations),
            "evidence": list(self.evidence),
        }


def _exposure_class(masked_ratio: float) -> str:
    if masked_ratio >= 0.9:
        return "AICPU_MASKED_BUT_UNDESIRABLE"
    if masked_ratio >= 0.2:
        return "AICPU_PARTIALLY_EXPOSED"
    return "AICPU_EXPOSED_NOT_ALLOWED"


def assess_aicpu_exposure(
    events: Sequence[KernelEvent],
    *,
    mask_all_non_aicpu: bool = False,
) -> AicpuExposureReport:
    """Classify AICPU exposure from explicit task type and interval overlap."""
    valid, limitations = valid_events(events or ())
    counters = {key for event in valid for key, _ in event.counters}
    counter_status = "present" if counters else "missing"
    if not counters:
        limitations.append("pmu_counters_missing")
    kinds = [event.task_kind for event in valid]
    if not valid:
        empty = not events
        return AicpuExposureReport(
            NOT_APPLICABLE if empty else UNCERTAIN,
            0.0 if empty else None, 0.0 if empty else None, 0.0 if empty else None,
            None, 0.0 if empty else None, None, counter_status,
            tuple(dict.fromkeys(limitations)), (),
        )
    if all(kind is None for kind in kinds):
        return AicpuExposureReport(
            UNCERTAIN, None, None, None, None, None, None, counter_status,
            tuple(dict.fromkeys(limitations + ["missing_task_type"])), (),
        )
    if any(kind is None for kind in kinds):
        limitations.append("missing_task_type_for_some_events")
    aicpu = [event for event in valid if event.task_kind == AI_CPU]
    aicpu_counters = {key for event in aicpu for key, _ in event.counters}
    counter_status = "present" if aicpu_counters else "missing"
    if not aicpu_counters and "pmu_counters_missing" not in limitations:
        limitations.append("pmu_counters_missing")
    if not aicpu:
        return AicpuExposureReport(
            NOT_APPLICABLE, 0.0, 0.0, 0.0, None, 0.0, None, counter_status,
            tuple(dict.fromkeys(limitations)), (),
        )
    mask_kinds = {AI_CORE, COMM, OTHER} if mask_all_non_aicpu else {AI_CORE}
    core = [event for event in valid if event.task_kind in mask_kinds]
    scopes: dict[tuple[int, Any], tuple[list[Interval], list[Interval]]] = {}
    for event in aicpu:
        entry = scopes.setdefault((event.step, event.rank), ([], []))
        entry[0].append(event.interval)
    for event in core:
        entry = scopes.setdefault((event.step, event.rank), ([], []))
        entry[1].append(event.interval)
    total_us = 0.0
    masked_us = 0.0
    evidence: list[str] = []
    assessable = 0
    for (step, rank), (aicpu_intervals, core_intervals) in sorted(scopes.items(), key=lambda item: str(item[0])):
        if not aicpu_intervals:
            continue
        merged_aicpu = merge_intervals(aicpu_intervals)
        duration_us = sum(end - start for start, end in merged_aicpu)
        if duration_us <= 0:
            continue
        if not core_intervals:
            limitations.append(f"missing_ai_core_events:step={step},rank={rank}")
            continue
        masked = sum(intersection_ms(interval, core_intervals) for interval in merged_aicpu) * 1000.0
        masked = min(masked, duration_us)
        total_us += duration_us
        masked_us += masked
        assessable += 1
        evidence.append(
            f"step={step},rank={rank},aicpu_union_ms={duration_us / 1000.0:.6f},masked_ms={masked / 1000.0:.6f}"
        )
    if total_us <= 0 or assessable == 0:
        return AicpuExposureReport(
            UNCERTAIN, None, None, None, None, None, None, counter_status,
            tuple(dict.fromkeys(limitations + ["no_assessable_aicpu_scope"])), tuple(evidence),
        )
    masked_ratio = min(1.0, max(0.0, masked_us / total_us))
    exposed_ratio = max(0.0, 1.0 - masked_ratio)
    if counter_status == "missing":
        limitations.append("aicpu_attribution_limited_to_interval_exposure")
    return AicpuExposureReport(
        ASSESSED,
        total_us / 1000.0,
        masked_us / 1000.0,
        (total_us - masked_us) / 1000.0,
        masked_ratio,
        exposed_ratio,
        _exposure_class(masked_ratio),
        counter_status,
        tuple(dict.fromkeys(limitations)),
        tuple(evidence),
    )


@dataclass(frozen=True)
class StepTiming:
    step: int | None
    wall_ms: float | None
    busy_union_ms: float | None
    kernel_sum_ms: float | None
    exposed_comm_ms: float | None
    host_gap_ms: float | None
    total_cost_ms: float | None
    underfeed_bubble_ratio: float | None
    aicpu_exposed_ratio: float | None
    limitations: tuple[str, ...] = ()

    def as_c7_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "wall_ms": self.wall_ms,
            "busy_union_ms": self.busy_union_ms,
            "kernel_sum_ms": self.kernel_sum_ms,
            "exposed_comm_ms": self.exposed_comm_ms,
            "host_gap_ms": self.host_gap_ms,
            "total_cost_ms": self.total_cost_ms,
            "underfeed_bubble_ratio": self.underfeed_bubble_ratio,
            "aicpu_exposed_ratio": self.aicpu_exposed_ratio,
            "limitations": list(self.limitations),
        }


def compute_step_timing(
    step: int | None,
    events: Sequence[KernelEvent],
    sample: StepSample | Mapping[str, Any] | None = None,
    *,
    host_gap_ms: float | None = None,
) -> StepTiming:
    """Compute C7 timing views without converting an unknown view into zero."""
    valid, limitations = valid_events(events or ())
    scoped = [event for event in valid if event.step == step]
    coerced = coerce_step_sample(sample) if sample is not None else None
    wall = coerced.wall_ms if coerced is not None and _finite(coerced.wall_ms) else None
    if wall is None and scoped:
        wall = (max(event.end_us for event in scoped) - min(event.start_us for event in scoped)) / 1000.0
        limitations.append("wall_from_event_span")
    busy = union_ms([event.interval for event in scoped]) if scoped else None
    kernel_sum = sum(event.duration_us for event in scoped) / 1000.0 if scoped else None
    if not scoped:
        limitations.append("no_kernel_events")
        if coerced is not None:
            busy = float(coerced.busy_union_ms) if _finite(coerced.busy_union_ms) else None
            kernel_sum = float(coerced.kernel_sum_ms) if _finite(coerced.kernel_sum_ms) else None
    total_cost: float | None = None
    if scoped and all(event.wait_us is not None for event in scoped):
        total_cost = sum(event.duration_us + float(event.wait_us) for event in scoped) / 1000.0
    elif scoped:
        limitations.append("missing_wait_time")
    elif coerced is not None and _finite(coerced.total_cost_ms):
        total_cost = float(coerced.total_cost_ms)
    exposed_comm: float | None = None
    if scoped and all(event.task_kind is not None for event in scoped):
        comm = [event.interval for event in scoped if event.task_kind == COMM]
        non_comm = [event.interval for event in scoped if event.task_kind != COMM]
        if comm:
            comm_union = merge_intervals(comm)
            comm_us = sum(end - start for start, end in comm_union)
            masked_us = sum(intersection_ms(interval, non_comm) for interval in comm_union) * 1000.0
            exposed_comm = max(0.0, comm_us - masked_us) / 1000.0
        else:
            exposed_comm = 0.0
    elif scoped:
        limitations.append("missing_task_type_exposed_comm")
    elif coerced is not None and _finite(coerced.exposed_comm_ms):
        exposed_comm = float(coerced.exposed_comm_ms)
    underfeed: float | None = None
    if wall is not None and busy is not None and wall > 0:
        underfeed = max(0.0, (wall - busy) / wall)
        if busy > wall:
            limitations.append("busy_union_exceeds_measured_wall")
    host_value = host_gap_ms if host_gap_ms is not None else (coerced.host_gap_ms if coerced else None)
    if host_value is not None and (not _finite(host_value) or host_value < 0):
        limitations.append("invalid_host_gap")
        host_value = None
    if host_value is None:
        limitations.append("host_gap_not_observed")
    aicpu_ratio: float | None = None
    if scoped:
        aicpu = assess_aicpu_exposure(scoped)
        aicpu_ratio = aicpu.exposed_ratio if aicpu.status == ASSESSED else None
        if aicpu.status != NOT_APPLICABLE:
            limitations.extend(aicpu.limitations)
    return StepTiming(
        step, wall, busy, kernel_sum, exposed_comm, host_value, total_cost, underfeed,
        aicpu_ratio, tuple(dict.fromkeys(limitations)),
    )


def _p50(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and _finite(value)]
    return float(statistics.median(clean)) if clean else None


def summarize_step_accounting(
    samples: Sequence[StepSample | Mapping[str, Any]] | None,
    steady_steps: Sequence[int] | SteadyWindow | None = None,
    events: Sequence[KernelEvent] | None = None,
    *,
    host_gaps_ms: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    """Build the C7 `step_accounting` fragment from measured inputs."""
    if isinstance(steady_steps, SteadyWindow):
        wanted = set(steady_steps.steady_steps) if steady_steps.status == "steady" else set()
    elif steady_steps is not None:
        wanted = {int(step) for step in steady_steps}
    else:
        wanted = None
    event_list = list(events or ())
    events_by_step: dict[int, list[KernelEvent]] = {}
    for event in event_list:
        if isinstance(event, KernelEvent) and event_error(event) is None:
            events_by_step.setdefault(event.step, []).append(event)
    ordered: list[StepSample] = []
    for item in samples or ():
        sample = coerce_step_sample(item)
        if sample is not None and sample.resolved_step is not None:
            ordered.append(sample)
    if not ordered and events_by_step:
        ordered = [StepSample(step=step, wall_ms=None) for step in sorted(events_by_step)]
    if wanted is not None:
        ordered = [sample for sample in ordered if sample.resolved_step in wanted]
    ordered.sort(key=lambda sample: sample.resolved_step)
    timings: list[StepTiming] = []
    limitations: list[str] = []
    for sample in ordered:
        step = int(sample.resolved_step)
        host_hint = host_gaps_ms.get(step) if host_gaps_ms else None
        timing = compute_step_timing(step, events_by_step.get(step, ()), sample, host_gap_ms=host_hint)
        for field in ("wall_ms", "busy_union_ms", "kernel_sum_ms", "exposed_comm_ms", "host_gap_ms", "total_cost_ms"):
            value = getattr(sample, field)
            if value is None:
                setattr_timing = getattr(timing, field)
                if setattr_timing is None:
                    limitations.append(f"missing_{field}:step={step}")
        timings.append(timing)
        limitations.extend(timing.limitations)
    if not ordered:
        limitations.append("no_samples_for_step_accounting")
    slowest_rank: dict[str, Any] | None = None
    rank_walls: dict[int, list[float]] = {}
    for sample in ordered:
        if sample.rank is not None and _finite(sample.wall_ms):
            rank_walls.setdefault(int(sample.rank), []).append(float(sample.wall_ms))
    if rank_walls:
        rank, values = max(rank_walls.items(), key=lambda item: statistics.median(item[1]))
        slowest_rank = {"rank": rank, "wall_ms_p50": float(statistics.median(values)), "dominant_excess_block": None}
    return {
        "steady_steps": [int(sample.resolved_step) for sample in ordered],
        "wall_ms_p50": _p50([timing.wall_ms for timing in timings]),
        "busy_union_ms_p50": _p50([timing.busy_union_ms for timing in timings]),
        "kernel_sum_ms_p50": _p50([timing.kernel_sum_ms for timing in timings]),
        "exposed_comm_ms_p50": _p50([timing.exposed_comm_ms for timing in timings]),
        "host_gap_ms_p50": _p50([timing.host_gap_ms for timing in timings]),
        "total_cost_ms_p50": _p50([timing.total_cost_ms for timing in timings]),
        "underfeed_bubble_ratio": _p50([timing.underfeed_bubble_ratio for timing in timings]),
        "aicpu_exposed_ratio": _p50([timing.aicpu_exposed_ratio for timing in timings]),
        "slowest_rank": slowest_rank,
        "step_count": len(ordered),
        "measurement_limitations": list(dict.fromkeys(limitations)),
    }


def _finding(finding_type: str, severity: str, confidence: str, summary: str, evidence_ref: str) -> dict[str, Any]:
    return {
        "finding_type": finding_type,
        "severity": severity,
        "confidence": confidence,
        "summary": summary,
        "evidence_ref": evidence_ref,
    }


def build_c7_fragment(
    *,
    samples: Sequence[StepSample | Mapping[str, Any]] | None = None,
    events: Sequence[KernelEvent] | None = None,
    steady_window: SteadyWindow | None = None,
    scene: Any = None,
    host_gaps_ms: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    """Return a plain C7 merge fragment; never a full fabricated summary."""
    accounting = summarize_step_accounting(
        samples,
        steady_window,
        events,
        host_gaps_ms=host_gaps_ms,
    )
    wait_report = assess_wait_anchor(events or ())
    aicpu_report = assess_aicpu_exposure(events or ())
    findings: list[dict[str, Any]] = []
    for item in wait_report.confirmed_findings:
        findings.append(_finding(
            "wait_anchor_masked",
            "medium",
            "high",
            (
                f"Wait-anchor candidate {item['op_name']} has wait_ratio "
                f"{item['wait_ratio']:.3f} and a correlated overlapping event; "
                "it must be demoted from root-cause ranking."
            ),
            "wait_anchor:" + str(item["op_name"]),
        ))
    for limitation in wait_report.limitations:
        if limitation.startswith("wait_anchor_overlap_missing:"):
            findings.append(_finding(
                "measurement_limitation", "low", "high",
                "A Wait-Anchor candidate lacks interval overlap evidence; no false-hotspot flag was emitted.",
                "wait_anchor:" + limitation.split(":", 1)[1],
            ))
    if aicpu_report.status == ASSESSED and aicpu_report.classification in {
        "AICPU_EXPOSED_NOT_ALLOWED", "AICPU_PARTIALLY_EXPOSED",
    }:
        ratio = aicpu_report.exposed_ratio or 0.0
        findings.append(_finding(
            "aicpu_fallback_exposed",
            "high" if ratio >= 0.2 else "medium",
            "high",
            (
                f"AICPU intervals are {aicpu_report.masked_ratio:.3f} masked by explicit "
                f"AI-Core overlap; exposed ratio is {ratio:.3f}. This is interval "
                "exposure evidence, not a PMU counter diagnosis."
            ),
            "aicpu:interval_exposure",
        ))
    has_aicpu = any(
        isinstance(event, KernelEvent) and event.task_kind == AI_CPU
        for event in (events or ())
    )
    if aicpu_report.counter_status == "missing" and has_aicpu:
        findings.append(_finding(
            "measurement_limitation", "low", "high",
            "PMU counters are absent; AICPU exposure is interval-based only and no hardware bottleneck is asserted.",
            "aicpu:pmu_counters",
        ))
    payload: dict[str, Any] = {
        "schema_version": "mindie-profile-summary/1",
        "step_accounting": accounting,
        "findings": findings,
        "anomaly_discovery": {
            "wait_anchor_status": wait_report.status,
            "aicpu_exposure_status": aicpu_report.status,
            "aicpu_exposure": aicpu_report.as_c7_dict(),
            "wait_anchor_ops": wait_report.as_c7_list(),
            "limitations": list(dict.fromkeys(
                wait_report.limitations
                + (aicpu_report.limitations if aicpu_report.status != NOT_APPLICABLE else ())
            )),
        },
    }
    if scene is not None:
        as_c7_scene = getattr(scene, "as_c7_scene", None)
        if callable(as_c7_scene):
            payload["scene"] = as_c7_scene()
        elif isinstance(scene, Mapping):
            payload["scene"] = dict(scene)
    return payload


def discover_anomalies(
    samples: Sequence[StepSample | Mapping[str, Any]] | None = None,
    events: Sequence[KernelEvent] | None = None,
    steady_window: SteadyWindow | None = None,
    scene: Any = None,
    host_gaps_ms: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    return build_c7_fragment(
        samples=samples,
        events=events,
        steady_window=steady_window,
        scene=scene,
        host_gaps_ms=host_gaps_ms,
    )

__all__ = [
    "AI_CPU",
    "AI_CORE",
    "COMM",
    "ASSESSED",
    "NOT_APPLICABLE",
    "UNCERTAIN",
    "KernelEvent",
    "WaitAnchorReport",
    "AicpuExposureReport",
    "StepTiming",
    "merge_intervals",
    "union_ms",
    "intersection_ms",
    "overlap_ratio",
    "assess_wait_anchor",
    "assess_aicpu_exposure",
    "compute_step_timing",
    "summarize_step_accounting",
    "build_c7_fragment",
    "discover_anomalies",
]
