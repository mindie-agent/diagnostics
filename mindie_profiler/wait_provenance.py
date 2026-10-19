"""Exact event, window, overlap, semantic, and hardware scope for Wait-Anchor.

[Role]
  Convert a mapped kernel event into an exact disclosed scope, build overlap
  pairs only inside one observation/window, and attach confirmed findings only
  to hotspot rows that retain the same event identity and window/source.

[Invariants]
  * Event IDs are composite hashes over rank/source kind/table/device/stream/
    task/original ns interval/locator; taskId or globalTaskId alone is never an
    identity.
  * Overlap is evaluated inside one observation/window only. Relative event
    timelines from different windows are never merged.
  * A confirmed group finding never turns an unrelated rank+name hotspot row
    into a confirmed hotspot. Non-evaluated siblings stay None with a reason.
  * Missing hardware/semantic metadata stays explicitly absent/unmapped; it is
    never inferred from a kernel name.

[Timeouts & Locks]
  Pure in-memory formatting over already-bounded records; no I/O or locks.

[Side-Effect Policy]
  Returns fresh dicts and mutates only the explicitly passed hotspot rows in
  ``attach_wait_anchor``.  No source artifact is read or changed.
"""
from __future__ import annotations

import bisect
import math
from typing import Any, Mapping, Sequence


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _event_ns(event: Any, field: str) -> int | None:
    value = getattr(event, field, None)
    if value is not None:
        return _int_or_none(value)
    window_start = getattr(event, "window_start_ns", None)
    relative = getattr(event, "start_us", None)
    if field == "event_start_ns" and window_start is not None and relative is not None:
        return int(window_start) + int(round(float(relative) * 1000.0))
    if field == "event_end_ns" and window_start is not None and relative is not None:
        duration = getattr(event, "duration_us", None)
        if duration is not None:
            return int(window_start) + int(round((float(relative) + float(duration)) * 1000.0))
    return None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def event_error(event: Any) -> str | None:
    if not isinstance(getattr(event, "name", None), str) or not getattr(event, "name", "").strip():
        return "missing_event_name"
    step = getattr(event, "step", None)
    if step is not None and (not isinstance(step, int) or isinstance(step, bool)):
        return "invalid_step"
    start_us = getattr(event, "start_us", None)
    duration_us = getattr(event, "duration_us", None)
    if not _finite(start_us) or not _finite(duration_us) or float(duration_us) <= 0:
        return "invalid_event_interval"
    if not math.isfinite(float(start_us) + float(duration_us)):
        return "invalid_event_interval"
    wait_us = getattr(event, "wait_us", None)
    if wait_us is not None and (not _finite(wait_us) or float(wait_us) < 0):
        return "invalid_wait_time"
    wait_start_us = getattr(event, "wait_start_us", None)
    if wait_start_us is not None and not _finite(wait_start_us):
        return "invalid_wait_interval"
    wait_end_us = getattr(event, "wait_end_us", None)
    if wait_end_us is not None and not _finite(wait_end_us):
        return "invalid_wait_interval"
    return None


def valid_events(events: Sequence[Any]) -> tuple[list[Any], list[str]]:
    valid: list[Any] = []
    limitations: list[str] = []
    for index, event in enumerate(events):
        if event is None or not hasattr(event, "name"):
            limitations.append(f"invalid_event:{index}")
            continue
        error = event_error(event)
        if error:
            limitations.append(f"{error}:{index}")
        else:
            valid.append(event)
    return valid, limitations


class WaitOverlapIndex:
    """Interval lookup scoped by observation/rank/device/step/stream.

    Wait-Anchor only needs the first qualifying same-scope overlap.  Building
    the index once avoids the previous full ``events x events`` scan on large
    real captures while preserving the same half-open interval semantics.
    """

    def __init__(self, events: Sequence[Any]) -> None:
        grouped: dict[tuple[Any, ...], list[Any]] = {}
        self.by_id: dict[str, list[Any]] = {}
        for event in events:
            event_id = getattr(event, "event_id", None)
            if event_id:
                self.by_id.setdefault(str(event_id), []).append(event)
            key = (
                getattr(event, "observation_id", None),
                getattr(event, "window_start_ns", None),
                getattr(event, "window_end_ns", None),
                getattr(event, "step", None),
                getattr(event, "rank", None),
                getattr(event, "device_id", None),
                getattr(event, "stream_id", None),
            )
            grouped.setdefault(key, []).append(event)
        self.groups: dict[tuple[Any, ...], tuple[list[float], list[float], list[Any]]] = {}
        for key, items in grouped.items():
            items.sort(key=lambda event: (float(event.start_us), str(getattr(event, "event_id", ""))))
            starts: list[float] = []
            prefix_max_end: list[float] = []
            max_end = -math.inf
            for event in items:
                starts.append(float(event.start_us))
                max_end = max(max_end, float(event.end_us))
                prefix_max_end.append(max_end)
            self.groups[key] = (starts, prefix_max_end, items)

    @staticmethod
    def _ratio_and_ns(wait_interval: tuple[float, float], other: Any) -> tuple[float, int] | None:
        start, end = wait_interval
        other_start = float(other.start_us)
        other_end = float(other.end_us)
        left, right = max(start, other_start), min(end, other_end)
        if right <= left:
            return None
        overlap_us = right - left
        duration_us = end - start
        if duration_us <= 0:
            return None
        return overlap_us / duration_us, int(round(overlap_us * 1000.0))

    def find(
        self,
        candidate: Any,
        wait_interval: tuple[float, float],
        *,
        min_overlap_ratio: float,
    ) -> tuple[Any, float, int] | None:
        key = (
            getattr(candidate, "observation_id", None),
            getattr(candidate, "window_start_ns", None),
            getattr(candidate, "window_end_ns", None),
            getattr(candidate, "step", None),
            getattr(candidate, "rank", None),
            getattr(candidate, "device_id", None),
            getattr(candidate, "stream_id", None),
        )
        group = self.groups.get(key)
        if group is not None:
            starts, prefix_max_end, items = group
            start, end = wait_interval
            index = bisect.bisect_left(starts, end)
            for position in range(index - 1, -1, -1):
                if prefix_max_end[position] <= start:
                    break
                other = items[position]
                if other is candidate:
                    continue
                measured = self._ratio_and_ns(wait_interval, other)
                if measured is not None and measured[0] >= min_overlap_ratio:
                    return other, measured[0], measured[1]
        for related_id in getattr(candidate, "related_event_ids", ()):
            for other in self.by_id.get(str(related_id), []):
                if other is candidate or not same_wait_scope(candidate, other):
                    continue
                measured = self._ratio_and_ns(wait_interval, other)
                if measured is not None and measured[0] >= min_overlap_ratio:
                    return other, measured[0], measured[1]
        return None


def same_wait_scope(first: Any, second: Any) -> bool:
    """True only for the same device/rank/step/observation/window scope."""
    if getattr(first, "step", None) != getattr(second, "step", None):
        return False
    first_observation = getattr(first, "observation_id", None)
    second_observation = getattr(second, "observation_id", None)
    if first_observation is not None or second_observation is not None:
        if first_observation != second_observation:
            return False
    if getattr(first, "window_start_ns", None) != getattr(second, "window_start_ns", None):
        return False
    if getattr(first, "window_end_ns", None) != getattr(second, "window_end_ns", None):
        return False
    first_rank = getattr(first, "rank", None)
    second_rank = getattr(second, "rank", None)
    if first_rank != second_rank:
        return False
    if getattr(first, "device_id", None) != getattr(second, "device_id", None):
        return False
    first_stream = getattr(first, "stream_id", None)
    second_stream = getattr(second, "stream_id", None)
    if first_stream is not None and second_stream is not None:
        first_id = getattr(first, "event_id", None)
        second_id = getattr(second, "event_id", None)
        if first_stream != second_stream and not (
            second_id in getattr(first, "related_event_ids", ())
            or first_id in getattr(second, "related_event_ids", ())
        ):
            return False
    return True


def event_scope(event: Any) -> dict[str, Any]:
    """Return the exact measured scope of one mapped event."""
    identity_payload = getattr(event, "event_identity", None) or {}
    payload = identity_payload.get("payload", {}) if isinstance(identity_payload, Mapping) else {}
    start_ns = _event_ns(event, "event_start_ns")
    end_ns = _event_ns(event, "event_end_ns")
    wait_ns = _int_or_none(getattr(event, "wait_ns", None))
    if wait_ns is None:
        wait_us = getattr(event, "wait_us", None)
        wait_ns = None if wait_us is None else int(round(float(wait_us) * 1000.0))
    wait_start_ns = _int_or_none(getattr(event, "wait_start_ns", None))
    if wait_start_ns is None and start_ns is not None and wait_ns is not None:
        wait_start_ns = start_ns - wait_ns
    wait_end_ns = _int_or_none(getattr(event, "wait_end_ns", None))
    if wait_end_ns is None and start_ns is not None and wait_ns is not None:
        wait_end_ns = start_ns
    identity = getattr(event, "event_id", None) or payload.get("event_id")
    if identity is None:
        identity = f"event:unavailable:{id(event)}"
    semantic_block = getattr(event, "semantic_block", None)
    hardware = getattr(event, "hardware_scope", None)
    if hardware is None:
        hardware_scope: dict[str, Any] = {
            "status": "not_available",
            "reason": "no hardware identity envelope was attached to this observation",
        }
    else:
        hardware_scope = dict(hardware)
    source_kind = getattr(event, "source_kind", None)
    source_table = getattr(event, "source_table", None) or getattr(event, "table", None)
    source_sha256 = getattr(event, "source_sha256", None)
    window_start = _int_or_none(getattr(event, "window_start_ns", None))
    window_end = _int_or_none(getattr(event, "window_end_ns", None))
    return {
        "event_id": str(identity),
        "event_identity_schema": payload.get("schema_version") if payload else None,
        "task_id": getattr(event, "task_id", None),
        "global_task_id": getattr(event, "global_task_id", None),
        "device_id": getattr(event, "device_id", None),
        "rank": getattr(event, "rank", None),
        "step": getattr(event, "step", None),
        "stream_id": getattr(event, "stream_id", None),
        "observation_id": getattr(event, "observation_id", None),
        "semantic_block": semantic_block,
        "semantic_scope": (
            {"status": "mapped", "semantic_block": semantic_block, "source": "classified_event_block"}
            if semantic_block is not None
            else {"status": "unmapped", "semantic_block": None, "reason": "no semantic block on event"}
        ),
        "window": {
            "start_ns": window_start,
            "end_ns": window_end,
            "label": getattr(event, "window_label", None),
        },
        "source": {
            "kind": source_kind,
            "table": source_table,
            "sha256": source_sha256,
        },
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ns": (None if start_ns is None or end_ns is None else max(0, end_ns - start_ns)),
        "wait_ns": wait_ns,
        "wait_start_ns": wait_start_ns,
        "wait_end_ns": wait_end_ns,
        "hardware_scope": hardware_scope,
        "locator": dict(payload.get("locator") or {}),
    }


def overlap_scope(candidate: Any, other: Any, ratio: float, overlap_ns: int | None = None) -> dict[str, Any]:
    """Return a candidate/overlap pair; caller guarantees same observation."""
    return {
        "candidate": event_scope(candidate),
        "overlap": event_scope(other),
        "overlap_ratio": float(ratio),
        "overlap_ns": overlap_ns,
        "scope": "same_observation_window",
    }


def _finding_overlap(finding: Mapping[str, Any], event_id: str) -> Mapping[str, Any] | None:
    for pair in finding.get("overlap_pairs", []):
        candidate = pair.get("candidate") or {}
        if candidate.get("event_id") == event_id:
            return pair
    return None


def _row_matches(row: Mapping[str, Any], scope: Mapping[str, Any]) -> tuple[bool, str | None]:
    locator = row.get("evidence_locator") or {}
    event_ids = {str(value) for value in locator.get("event_ids", []) if value}
    if not event_ids:
        return False, "hotspot locator sample has no exact event identity"
    event_id = str(scope.get("event_id"))
    if event_id not in event_ids:
        return False, "confirmed event is not in the retained hotspot locator sample"
    source = scope.get("source") or {}
    row_source = locator.get("source_sha256")
    if source.get("sha256") is not None:
        if row_source is None:
            return False, "hotspot locator has no source sha256"
        if source.get("sha256") != row_source:
            return False, "source sha256 differs from the hotspot locator source"
    window = scope.get("window") or {}
    row_windows = {
        (int(item[0]), int(item[1]))
        for item in locator.get("window_bounds_ns", [])
        if isinstance(item, (list, tuple)) and len(item) == 2
    }
    if window.get("start_ns") is not None and window.get("end_ns") is not None:
        if (int(window["start_ns"]), int(window["end_ns"])) not in row_windows:
            return False, "confirmed event window differs from the hotspot window"
    row_semantic = row.get("semantic_block")
    scope_semantic = scope.get("semantic_block")
    if row_semantic is not None and scope_semantic is not None and row_semantic != scope_semantic:
        return False, "semantic block differs from the hotspot semantic block"
    return True, None


def attach_wait_anchor(rows: Sequence[dict[str, Any]], wait_anchor: Mapping[str, Any]) -> None:
    """Attach only exact confirmed candidates; leave all siblings unconfirmed."""
    findings = list(wait_anchor.get("findings", []))
    confirmed_by_event: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for finding in findings:
        if not finding.get("is_false_hotspot_risk"):
            continue
        confirmed_ids = {str(value) for value in finding.get("confirmed_event_ids", []) if value}
        scopes = list(finding.get("confirmed_events") or []) + list(finding.get("candidate_events") or [])
        for scope in scopes:
            event_id = str(scope.get("event_id"))
            if event_id in confirmed_ids and event_id not in confirmed_by_event:
                confirmed_by_event[event_id] = (finding, scope)
    overall = str(wait_anchor.get("status") or "unknown")
    for row in rows:
        locator = row.get("evidence_locator") or {}
        row_ids = [str(value) for value in locator.get("event_ids", []) if value]
        match = None
        reason = None
        for event_id in row_ids:
            item = confirmed_by_event.get(event_id)
            if item is None:
                continue
            finding, scope = item
            ok, why = _row_matches(row, scope)
            if ok:
                match = (finding, scope)
                break
            reason = why
        if match is None:
            row["wait_anchor_false_hotspot"] = None
            row["wait_anchor_evidence"] = []
            row["wait_anchor_scope"] = None
            if overall == "unverified":
                row["wait_anchor_assessment"] = "unverified"
                row["wait_anchor_reason"] = wait_anchor.get("reason")
            elif row_ids:
                row["wait_anchor_assessment"] = "not_evaluated"
                row["wait_anchor_reason"] = (
                    reason
                    or "no exact confirmed Wait-Anchor candidate matched this retained hotspot locator"
                )
            else:
                row["wait_anchor_assessment"] = "not_evaluated"
                row["wait_anchor_reason"] = "hotspot locator sample has no exact event identity"
            continue
        finding, scope = match
        overlap = _finding_overlap(finding, str(scope.get("event_id")))
        if overlap is None:
            row["wait_anchor_false_hotspot"] = None
            row["wait_anchor_assessment"] = "not_evaluated"
            row["wait_anchor_reason"] = "confirmed candidate has no retained exact overlap pair"
            row["wait_anchor_evidence"] = []
            row["wait_anchor_scope"] = None
            continue
        row["wait_anchor_false_hotspot"] = True
        row["wait_anchor_assessment"] = "assessed"
        row["wait_anchor_reason"] = None
        row["wait_anchor_evidence"] = list(finding.get("evidence") or [])
        row["wait_anchor_scope"] = {
            "scope_level": "exact_event",
            "event_scope": dict(scope),
            "overlap_scope": None if overlap is None else dict(overlap),
            "candidate_event_ids": list(finding.get("candidate_event_ids") or []),
            "confirmed_event_ids": list(finding.get("confirmed_event_ids") or []),
            "group_scope": {
                "observation_id": finding.get("observation_id"),
                "semantic_block": finding.get("semantic_block"),
                "op_name": finding.get("op_name"),
                "candidate_event_count": finding.get("candidate_event_count"),
                "confirmed_event_count": finding.get("confirmed_event_count"),
            },
            "hardware_scope": scope.get("hardware_scope"),
            "semantic_scope": scope.get("semantic_scope"),
            "source_scope": scope.get("source"),
            "wait_us_avg": finding.get("wait_us_avg"),
            "wait_ratio": finding.get("wait_ratio"),
            "evidence_status": finding.get("evidence_status"),
        }


__all__ = [
    "event_error",
    "valid_events",
    "WaitOverlapIndex",
    "same_wait_scope",
    "event_scope",
    "overlap_scope",
    "attach_wait_anchor",
]
