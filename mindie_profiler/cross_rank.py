"""Cross-rank p50 statistics and slowest-rank attribution.

[Role]
  Aggregate per-window observations into per-rank and per-step medians, find
  the slowest rank by wall p50, and attribute its excess to one semantic
  block using only comparable rank medians.

[Invariants]
  * Every statistic is computed from measured window observations; missing
    values stay ``None`` and are excluded from a median, never treated as 0.
  * ``slowest_rank`` is chosen by wall p50 over the explicit steps, not by an
    average that can hide the rank that determines completion time.
  * ``dominant_excess_block`` is reported only when at least two ranks have
    that block's median; otherwise it is ``None`` with a reason.
  * No thermal/power conclusion is drawn from timing alone.

[Timeouts & Locks]
  O(n log n) medians over the bounded observation set; no I/O or locks.

[Side-Effect Policy]
  Pure computation over immutable observations.
"""

from __future__ import annotations

import statistics
from typing import Iterable, Sequence

from .classify import ALL_BLOCKS
from .events import Observation

def p50(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return round(float(statistics.median(clean)), 6)


def _ms(value: int | float | None) -> float | None:
    return None if value is None else round(float(value) / 1_000_000.0, 6)


def _metric(accounting, name: str) -> float | None:
    ns_name = f"{name}_ns"
    value = getattr(accounting, ns_name, None)
    return _ms(value)


def _block_ms(accounting, block: str) -> float | None:
    value = accounting.block_union_ns.get(block)
    return _ms(value)


def _block_share(accounting, block: str) -> float | None:
    return accounting.block_share_pct.get(block)


def _step_count(observations: Sequence[Observation]) -> int:
    return len({obs.window.step for obs in observations if obs.window.step is not None})


def aggregate_observations(observations: Sequence[Observation]) -> dict:
    usable = [obs for obs in observations if obs.accounting.available]
    steps = sorted({obs.window.step for obs in observations if obs.window.step is not None})
    return {
        "observation_count": len(observations),
        "observations_with_data": len(usable),
        "step_count": len(steps),
        "rank_count": len({obs.rank_id for obs in observations}),
        "steady_steps": steps,
        "request_wall_ms_p50": p50(_metric(obs.accounting, "request_wall") for obs in usable),
        "observed_event_span_ms_p50": p50(_metric(obs.accounting, "observed_event_span") for obs in usable),
        "device_busy_union_ms_p50": p50(_metric(obs.accounting, "busy_union") for obs in usable),
        "device_activity_sum_ms_p50": p50(_metric(obs.accounting, "device_activity_sum") for obs in usable),
        "exposed_comm_ms_p50": p50(_metric(obs.accounting, "exposed_comm") for obs in usable),
        "device_uncovered_ms_p50": p50(_metric(obs.accounting, "device_uncovered") for obs in usable),
        "total_cost_ms_p50": p50(_metric(obs.accounting, "total_cost") for obs in usable),
        "clipped_event_count": sum(obs.accounting.clipped_event_count for obs in usable),
        # Deprecated aliases.
        "wall_ms_p50": p50(_metric(obs.accounting, "observed_event_span") for obs in usable),
        "busy_union_ms_p50": p50(_metric(obs.accounting, "busy_union") for obs in usable),
        "kernel_sum_ms_p50": p50(_metric(obs.accounting, "device_activity_sum") for obs in usable),
        "host_gap_ms_p50": p50(_metric(obs.accounting, "device_uncovered") for obs in usable),
        "exposed_aicpu_ms_p50": p50(_metric(obs.accounting, "exposed_aicpu") for obs in usable),
        "underfeed_bubble_ratio_p50": p50(obs.accounting.underfeed_ratio for obs in usable),
        "aicpu_exposed_ratio_p50": p50(obs.accounting.aicpu_exposed_ratio for obs in usable),
        "coverage_ratio_p50": p50(obs.accounting.coverage_ratio for obs in usable),
    }


def rank_summaries(observations: Sequence[Observation]) -> list[dict]:
    by_rank: dict[str, list[Observation]] = {}
    for obs in observations:
        by_rank.setdefault(obs.rank_id, []).append(obs)
    rows: list[dict] = []
    for rank_id in sorted(by_rank):
        group = by_rank[rank_id]
        usable = [obs for obs in group if obs.accounting.available]
        rank_index = next((obs.rank_index for obs in group if obs.rank_index is not None), None)
        row = {
            "rank": rank_index if rank_index is not None else rank_id,
            "rank_id": rank_id,
            "rank_index": rank_index,
            "observations": len(group),
            "observations_with_data": len(usable),
            "empty_windows": len(group) - len(usable),
            "steps": sorted({obs.window.step for obs in group if obs.window.step is not None}),
            "event_count": sum(obs.accounting.event_count for obs in group),
            "comm_event_count": sum(obs.accounting.comm_event_count for obs in group),
            "clipped_event_count": sum(obs.accounting.clipped_event_count for obs in group),
            "request_wall_ms_p50": p50(_metric(obs.accounting, "request_wall") for obs in usable),
            "observed_event_span_ms_p50": p50(_metric(obs.accounting, "observed_event_span") for obs in usable),
            "device_busy_union_ms_p50": p50(_metric(obs.accounting, "busy_union") for obs in usable),
            "device_activity_sum_ms_p50": p50(_metric(obs.accounting, "device_activity_sum") for obs in usable),
            "exposed_comm_ms_p50": p50(_metric(obs.accounting, "exposed_comm") for obs in usable),
            "device_uncovered_ms_p50": p50(_metric(obs.accounting, "device_uncovered") for obs in usable),
            "total_cost_ms_p50": p50(_metric(obs.accounting, "total_cost") for obs in usable),
            "wall_ms_p50": p50(_metric(obs.accounting, "observed_event_span") for obs in usable),
            "busy_union_ms_p50": p50(_metric(obs.accounting, "busy_union") for obs in usable),
            "kernel_sum_ms_p50": p50(_metric(obs.accounting, "device_activity_sum") for obs in usable),
            "host_gap_ms_p50": p50(_metric(obs.accounting, "device_uncovered") for obs in usable),
            "underfeed_bubble_ratio_p50": p50(obs.accounting.underfeed_ratio for obs in usable),
            "aicpu_exposed_ratio_p50": p50(obs.accounting.aicpu_exposed_ratio for obs in usable),
            "coverage_ratio_p50": p50(obs.accounting.coverage_ratio for obs in usable),
            "block_union_ms_p50": {
                block: p50(_block_ms(obs.accounting, block) for obs in usable) for block in ALL_BLOCKS
            },
            "block_share_pct_p50": {
                block: p50(_block_share(obs.accounting, block) for obs in usable) for block in ALL_BLOCKS
            },
        }
        rows.append(row)
    return rows


def slowest_rank(summaries: Sequence[dict]) -> dict | None:
    candidates = [row for row in summaries if row.get("wall_ms_p50") is not None]
    if not candidates:
        return None
    slowest = max(candidates, key=lambda row: (row["wall_ms_p50"], row.get("busy_union_ms_p50") or 0.0, row["rank_id"]))
    others = [row for row in summaries if row["rank_id"] != slowest["rank_id"]]
    excesses: list[tuple[float, str]] = []
    for block in ALL_BLOCKS:
        mine = slowest["block_union_ms_p50"].get(block)
        peers = [row["block_union_ms_p50"].get(block) for row in others]
        peers = [value for value in peers if value is not None]
        if mine is None or not peers:
            continue
        baseline = float(statistics.median(peers))
        excesses.append((round(mine - baseline, 6), block))
    dominant = max(excesses, default=None)
    return {
        "rank": slowest["rank"],
        "rank_id": slowest["rank_id"],
        "rank_index": slowest["rank_index"],
        "wall_ms_p50": slowest["wall_ms_p50"],
        "busy_union_ms_p50": slowest["busy_union_ms_p50"],
        "dominant_excess_block": dominant[1] if dominant and dominant[0] > 0 else None,
        "dominant_excess_ms": dominant[0] if dominant and dominant[0] > 0 else None,
        "baseline_rank_count": len(others),
        "dominant_excess_reason": (
            None
            if dominant and dominant[0] > 0
            else ("no comparable rank/block median" if not excesses else "slowest rank has no positive block excess")
        ),
    }


def cross_rank_stats(observations: Sequence[Observation]) -> dict:
    summaries = rank_summaries(observations)
    walls = [row["wall_ms_p50"] for row in summaries if row["wall_ms_p50"] is not None]
    slowest = slowest_rank(summaries)
    skew = round(max(walls) - min(walls), 6) if len(walls) >= 2 else None
    ratio = round(max(walls) / min(walls), 6) if len(walls) >= 2 and min(walls) > 0 else None
    return {
        "rank_summaries": summaries,
        "slowest_rank": slowest,
        "rank_count_with_data": len(walls),
        "wall_skew_ms": skew,
        "wall_ratio": ratio,
    }
