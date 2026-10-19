"""Interval accounting for one explicit rank/window observation.

[Role]
  Compute wall, busy union, kernel sum, exposed communication, host/idle gap,
  AICPU exposure, and per-block attributed busy time from classified events.
  All arithmetic is integer-nanosecond interval math.

[Invariants]
  * Overlapping device events are counted once in ``busy_union`` and once per
    event in ``kernel_sum``; the two are deliberately not additive.
  * ``wall`` is the observed event span (max end - min start), not the
    caller's wider filter window.  ``host_gap = wall - busy_union`` is device
    idle time, not a measured host dispatch latency.
  * Exposed communication is comm-union minus non-comm-union; fused MoE
    communication is non-comm by classification and therefore not exposed.
  * Empty windows return ``available=False`` and None metrics, never zeros.

[Timeouts & Locks]
  O(n log n) sort over the bounded event list; no locks or I/O.

[Side-Effect Policy]
  Pure computation over immutable inputs.  No filesystem or global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .classify import ALL_BLOCKS, EVENT_BLOCKS

Interval = tuple[int, int]


def merge_intervals(intervals: Sequence[Interval]) -> list[Interval]:
    ordered = sorted((left, right) for left, right in intervals if right > left)
    if not ordered:
        return []
    merged: list[Interval] = [ordered[0]]
    for left, right in ordered[1:]:
        last_left, last_right = merged[-1]
        if left <= last_right:
            if right > last_right:
                merged[-1] = (last_left, right)
        else:
            merged.append((left, right))
    return merged


def union_length(intervals: Sequence[Interval]) -> int:
    return sum(right - left for left, right in merge_intervals(intervals))


def subtract_length(intervals: Sequence[Interval], cover: Sequence[Interval]) -> int:
    """Length of ``intervals`` not covered by ``cover`` (both merged here)."""

    uncovered = merge_intervals(intervals)
    covered = merge_intervals(cover)
    if not uncovered or not covered:
        return union_length(uncovered)
    total = 0
    cover_idx = 0
    for left, right in uncovered:
        cursor = left
        while cover_idx < len(covered) and covered[cover_idx][1] <= cursor:
            cover_idx += 1
        probe = cover_idx
        while probe < len(covered) and cursor < right:
            cover_left, cover_right = covered[probe]
            if cover_right <= cursor:
                probe += 1
                continue
            if cover_left >= right:
                break
            if cover_left > cursor:
                total += min(cover_left, right) - cursor
            cursor = max(cursor, cover_right)
            if cursor >= right:
                break
            probe += 1
        if cursor < right:
            total += right - cursor
        cover_idx = probe
    return max(0, total)


def _gaps(merged: Sequence[Interval]) -> list[int]:
    return [
        right[0] - left[1]
        for left, right in zip(merged[:-1], merged[1:])
        if right[0] > left[1]
    ]


@dataclass(slots=True)
class Accounting:
    available: bool
    event_count: int
    comm_event_count: int
    aicpu_event_count: int
    stream_count: int
    window_span_ns: int
    wall_ns: int | None = None
    busy_union_ns: int | None = None
    kernel_sum_ns: int | None = None
    exposed_comm_ns: int | None = None
    host_gap_ns: int | None = None
    total_cost_ns: int | None = None
    exposed_aicpu_ns: int | None = None
    underfeed_ratio: float | None = None
    aicpu_exposed_ratio: float | None = None
    coverage_ratio: float | None = None
    largest_internal_bubble_ns: int | None = None
    internal_bubble_total_ns: int | None = None
    block_union_ns: dict[str, int] = field(default_factory=dict)
    block_duration_ns: dict[str, int] = field(default_factory=dict)
    block_event_count: dict[str, int] = field(default_factory=dict)
    block_share_pct: dict[str, float | None] = field(default_factory=dict)
    first_event_ns: int | None = None
    last_event_ns: int | None = None

    def metric_ms(self, name: str) -> float | None:
        value = getattr(self, f"{name}_ns")
        return None if value is None else round(value / 1_000_000.0, 6)


def account_window(events: Sequence, window) -> Accounting:
    """Account one explicit window over already-classified events."""

    all_intervals: list[Interval] = []
    comm_intervals: list[Interval] = []
    noncomm_intervals: list[Interval] = []
    aicpu_intervals: list[Interval] = []
    non_aicpu_intervals: list[Interval] = []
    block_intervals: dict[str, list[Interval]] = {block: [] for block in EVENT_BLOCKS}
    block_duration: dict[str, int] = {block: 0 for block in EVENT_BLOCKS}
    block_events: dict[str, int] = {block: 0 for block in EVENT_BLOCKS}
    event_count = 0
    comm_events = 0
    aicpu_events = 0
    streams: set[str] = set()
    first_ns: int | None = None
    last_ns: int | None = None
    kernel_sum_ns = 0

    for event in events:
        clipped = event.clipped(window.start_ns, window.end_ns)
        if clipped is None:
            continue
        left, right = clipped
        duration = right - left
        event_count += 1
        kernel_sum_ns += duration
        all_intervals.append((left, right))
        streams.add(str(event.stream_id))
        first_ns = left if first_ns is None else min(first_ns, left)
        last_ns = right if last_ns is None else max(last_ns, right)
        block = event.block if event.block in block_intervals else "Unclassified"
        block_intervals[block].append((left, right))
        block_duration[block] += duration
        block_events[block] += 1
        if event.is_comm:
            comm_events += 1
            comm_intervals.append((left, right))
        else:
            noncomm_intervals.append((left, right))
        if event.is_aicpu:
            aicpu_events += 1
            aicpu_intervals.append((left, right))
        else:
            non_aicpu_intervals.append((left, right))

    base = Accounting(
        available=bool(event_count),
        event_count=event_count,
        comm_event_count=comm_events,
        aicpu_event_count=aicpu_events,
        stream_count=len(streams),
        window_span_ns=window.span_ns,
        block_union_ns={block: 0 for block in ALL_BLOCKS},
        block_duration_ns={block: 0 for block in ALL_BLOCKS},
        block_event_count={block: 0 for block in EVENT_BLOCKS},
        block_share_pct={block: None for block in ALL_BLOCKS},
    )
    if not event_count or first_ns is None or last_ns is None:
        return base

    busy_union_ns = union_length(all_intervals)
    wall_ns = max(0, last_ns - first_ns)
    host_gap_ns = max(0, wall_ns - busy_union_ns)
    exposed_comm_ns = subtract_length(comm_intervals, noncomm_intervals)
    exposed_aicpu_ns = subtract_length(aicpu_intervals, non_aicpu_intervals)
    merged = merge_intervals(all_intervals)
    gaps = _gaps(merged)
    attributed = 0
    for block in EVENT_BLOCKS:
        block_union = union_length(block_intervals[block])
        base.block_union_ns[block] = block_union
        base.block_duration_ns[block] = block_duration[block]
        base.block_event_count[block] = block_events[block]
        attributed += block_union
    for block in EVENT_BLOCKS:
        value = base.block_union_ns[block]
        base.block_share_pct[block] = (
            round(100.0 * value / attributed, 6) if attributed > 0 else None
        )
    base.block_union_ns["Host_Gap"] = host_gap_ns
    base.block_duration_ns["Host_Gap"] = host_gap_ns
    base.block_share_pct["Host_Gap"] = None
    base.wall_ns = wall_ns
    base.busy_union_ns = busy_union_ns
    base.kernel_sum_ns = kernel_sum_ns
    base.exposed_comm_ns = exposed_comm_ns
    base.host_gap_ns = host_gap_ns
    base.total_cost_ns = kernel_sum_ns + host_gap_ns
    base.exposed_aicpu_ns = exposed_aicpu_ns
    base.underfeed_ratio = round(host_gap_ns / wall_ns, 6) if wall_ns > 0 else None
    base.aicpu_exposed_ratio = round(exposed_aicpu_ns / wall_ns, 6) if wall_ns > 0 else None
    base.coverage_ratio = round(wall_ns / window.span_ns, 6) if window.span_ns > 0 else None
    base.largest_internal_bubble_ns = max(gaps) if gaps else 0
    base.internal_bubble_total_ns = sum(gaps)
    base.first_event_ns = first_ns
    base.last_event_ns = last_ns
    return base


def hotspot_aggregates(events: Sequence, window=None) -> list[dict]:
    """Group classified events by (rank, block, name) for hotspot reporting.

    When ``window`` is supplied, occurrence durations are clipped to it so the
    hotspot sum stays consistent with the window's ``kernel_sum``.
    """

    groups: dict[tuple[str, str, str], dict] = {}
    for event in events:
        if window is not None:
            clipped = event.clipped(window.start_ns, window.end_ns)
            if clipped is None:
                continue
            duration = clipped[1] - clipped[0]
        else:
            duration = event.duration_ns
        key = (event.rank_id, event.block, event.name or "<missing>")
        row = groups.get(key)
        if row is None:
            row = {
                "rank_id": event.rank_id,
                "semantic_block": event.block,
                "kernel_name": event.name or "<missing>",
                "duration_ns": 0,
                "count": 0,
                "durations": [],
                "bound_family": event.bound_family,
                "op_type": event.op_type,
                "input_shapes": event.input_shapes,
                "input_dtypes": event.input_dtypes,
                "input_formats": event.input_formats,
                "locators": [],
            }
            groups[key] = row
        row["duration_ns"] += duration
        row["count"] += 1
        row["durations"].append(duration)
        if len(row["locators"]) < 16:
            row["locators"].append(event.locator)
    return list(groups.values())
