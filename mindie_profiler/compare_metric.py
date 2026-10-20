"""Duration metric comparison and distribution-free confidence intervals.

[Role]
  Build one C7 comparison metric entry (baseline/candidate p50, delta,
  direction, relative delta, and optional median-order-statistic interval)
  from already-aligned real observations.

[Invariants]
  * Missing values stay ``None``; one missing side makes the metric
    ``not_comparable`` and never produces a zero delta.
  * The interval method requires at least six per-sample observations and
    keeps baseline/candidate samples unpaired; it never invents a paired
    delta interval.
  * Ratio units report ``delta_pp``; time units stay in milliseconds.
  * No I/O, no randomness, and no mutation of caller values.

[Timeouts & Locks]
  Pure arithmetic over bounded sample lists; no locks.

[Side-Effect Policy]
  No filesystem, network, or mutable-global side effects.
"""

from __future__ import annotations

import math
from typing import Any


def _cdf(k: int, n: int) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) for i in range(k + 1)) / (2**n)


def median_interval(values: list[float]) -> list[float] | None:
    if len(values) < 6:
        return None
    ordered = sorted(values)
    rank = 0
    for candidate in range(1, len(ordered) // 2 + 1):
        if _cdf(candidate - 1, len(ordered)) <= 0.025:
            rank = candidate
        else:
            break
    return [ordered[rank - 1], ordered[len(ordered) - rank]] if rank else None


def num(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def metric(
    unit: str,
    baseline: Any,
    candidate: Any,
    b_values: list[float] | None = None,
    c_values: list[float] | None = None,
) -> dict[str, Any]:
    """Compare one metric from two aligned measurements."""

    b_value, c_value = num(baseline), num(candidate)
    entry: dict[str, Any] = {
        "unit": unit,
        "status": "compared" if b_value is not None and c_value is not None else "not_comparable",
        "baseline_p50": b_value,
        "candidate_p50": c_value,
        "delta": None,
        "relative_delta_pct": None,
        "direction": None,
        "baseline_sample_count": len(b_values) if b_values else 0,
        "candidate_sample_count": len(c_values) if c_values else 0,
        "confidence_interval": None,
    }
    if unit == "ratio":
        entry["delta_pp"] = None
    if b_value is None or c_value is None:
        entry["reason"] = "baseline_or_candidate_value_missing"
        return entry
    delta = c_value - b_value
    entry["delta"] = delta
    entry["direction"] = "increase" if delta > 0 else "decrease" if delta < 0 else "unchanged"
    if unit == "ratio":
        entry["delta_pp"] = delta * 100.0
    if b_value != 0:
        entry["relative_delta_pct"] = delta / abs(b_value) * 100.0
    if b_values and c_values:
        b_interval, c_interval = median_interval(b_values), median_interval(c_values)
        if b_interval is not None and c_interval is not None:
            entry["confidence_interval"] = {
                "level": 0.95,
                "method": "distribution_free_median_order_statistics",
                "baseline": b_interval,
                "candidate": c_interval,
                "delta": None,
                "delta_reason": "per_sample_differences_are_not_paired_by_step",
            }
        else:
            entry["confidence_interval_reason"] = "fewer_than_6_per_sample_observations"
    else:
        entry["confidence_interval_reason"] = "per_sample_observations_unavailable"
    return entry


__all__ = ["median_interval", "metric"]
