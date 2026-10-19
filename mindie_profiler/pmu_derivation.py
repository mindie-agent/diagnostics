"""Source-backed, full-event-only TASK_PMU derivation.

[Role]
  Map actual TASK_PMU_INFO/kernel_details metrics to canonical units and derive
  descriptive pipe-time fractions, ratio/time consistency, dominant measured
  pipes, and official msprof-analyze cube/vector candidate rules.

[Invariants]
  * Raw ``*_ratio`` values are fractions of the corresponding AIC/AIV total
    time, never wall-clock utilization or causal proof.  Pipe fractions may
    overlap and can sum above 1; they are not normalised.
  * A rule candidate is emitted only when every required actual metric exists;
    it is a heuristic candidate from the pinned official rule table, not a
    confirmed root cause.
  * ``cube_utilization_pct`` is a separate CSV percent metric; it is never
    derived from ``aic_mac_ratio``.
  * PMU values are full-event counters.  Clipped events retain full counters
    but their derived evidence is explicitly not usable for clipped-window
    inference.
  * Missing metrics/units stay absent/unknown; no fabricated values.

[Timeouts & Locks]
  Pure in-memory mapping and arithmetic; no I/O after the one cached mapping
  read.

[Side-Effect Policy]
  No file writes, network, or source mutation.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Mapping

from .sources_db import load_mapping

PMU_ANALYSIS_SCHEMA = "mindie-pmu-derived/1"
RULE_SOURCE = "msprof-analyze/advisor/rules/en/aicore_performance.yaml"
RULE_SOURCE_PATH = "msprof_analyze/advisor/rules/en/aicore_performance.yaml"
RULE_SOURCE_VERSION = "26.1.0"
RULE_SOURCE_COMMIT = "639eb8702c1884813dfab59ddb371f7838caab6d"

_AIC_PIPES = ("mac", "scalar", "mte1", "mte2", "fixpipe")
_AIV_PIPES = ("vec", "scalar", "mte2", "mte3", "mac")
_CUBE_REQUIRED = ("aic_mac_ratio", "aic_mte2_ratio")
_VECTOR_REQUIRED = ("aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio")


@lru_cache(maxsize=1)
def _mapping() -> dict[str, Any]:
    return dict(load_mapping())


def metric_units(metrics: Mapping[str, float]) -> dict[str, str]:
    specs = _mapping().get("pmu_metrics", {})
    return {
        name: (str(specs.get(name, {}).get("unit")) if specs.get(name, {}).get("unit") else "unknown")
        for name in sorted(metrics)
    }


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _number(metrics: Mapping[str, float], name: str) -> float | None:
    value = metrics.get(name)
    return float(value) if _finite(value) else None


def _scope_evidence(
    metrics: Mapping[str, float], prefix: str, pipes: tuple[str, ...],
    export_resolution: Mapping[str, float],
) -> dict[str, Any]:
    total_time = _number(metrics, f"{prefix}_total_time")
    total_cycles = _number(metrics, f"{prefix}_total_cycles")
    checks: list[dict[str, Any]] = []
    fractions: dict[str, float] = {}
    for pipe in pipes:
        ratio = _number(metrics, f"{prefix}_{pipe}_ratio")
        pipe_time = _number(metrics, f"{prefix}_{pipe}_time")
        if ratio is None or total_time is None:
            continue
        fractions[pipe] = ratio
        if pipe_time is None:
            checks.append({"pipe": pipe, "status": "time_missing"})
            continue
        expected = ratio * total_time
        residual = expected - pipe_time
        tolerance = max(1e-6, abs(pipe_time) * 1e-6)
        resolution_names = (f"{prefix}_{pipe}_ratio", f"{prefix}_total_time", f"{prefix}_{pipe}_time")
        resolution = [_number(export_resolution, name) for name in resolution_names]
        envelope = None
        if all(value is not None for value in resolution):
            ratio_unit, total_unit, pipe_unit = resolution
            # One displayed unit permits either rounding or truncation. This
            # bounds representation differences, not hardware measurement error.
            envelope = abs(total_time) * ratio_unit + abs(ratio) * total_unit + ratio_unit * total_unit + pipe_unit
        if abs(residual) <= tolerance:
            status = "consistent"
        elif envelope is not None and abs(residual) <= envelope:
            status = "compatible_with_export_precision"
        else:
            status = "inconsistent"
        checks.append(
            {
                "pipe": pipe,
                "ratio": ratio,
                "pipe_time_ns": pipe_time,
                "expected_time_ns": expected,
                "residual_ns": residual,
                "status": status,
                "export_precision_envelope_ns": envelope,
                "export_precision_policy": "one_display_unit_per_factor" if envelope is not None else None,
            }
        )
    dominant = None
    if fractions:
        name, value = max(fractions.items(), key=lambda item: (item[1], item[0]))
        dominant = {"pipe": name, "time_fraction": value}
    return {
        "total_time_ns": total_time,
        "total_cycles": total_cycles,
        "pipe_time_fractions": fractions,
        "sum_pipe_time_fractions": (sum(fractions.values()) if fractions else None),
        "fractions_overlap": (sum(fractions.values()) > 1.0 + 1e-9) if fractions else None,
        "ratio_time_consistency": checks,
        "ratio_time_all_consistent": (
            False if any(item["status"] == "inconsistent" for item in checks)
            else True if checks and all(item["status"] == "consistent" for item in checks)
            else None
        ),
        "dominant_pipe_by_time": dominant,
    }


def _frequency(total_time: float | None, total_cycles: float | None, block_num: int | None) -> dict[str, Any]:
    if total_time is None or total_cycles is None:
        return {"status": "not_available", "reason": "time_or_cycles_missing"}
    if total_time <= 0:
        return {"status": "not_available", "reason": "no_active_time"}
    if block_num is None or block_num <= 0:
        return {"status": "not_available", "reason": "block_num_missing"}
    return {
        "status": "inferred",
        "hz": total_cycles / (total_time * block_num) * 1e9,
        "basis": "total_cycles / (total_time_ns * block_num)",
        "scope_note": "inferred per-block relation; not generalized across A2/A3 core types",
    }


def _rule_meta(role: str, thresholds: Mapping[str, float]) -> dict[str, Any]:
    return {
        "role": role,
        "thresholds": dict(thresholds),
        "source": RULE_SOURCE,
        "source_path": RULE_SOURCE_PATH,
        "source_version": RULE_SOURCE_VERSION,
        "source_commit": RULE_SOURCE_COMMIT,
        "source_url": f"https://gitcode.com/Ascend/msprof-analyze/blob/{RULE_SOURCE_COMMIT}/{RULE_SOURCE_PATH}",
        "interpretation": "per_event_thresholds_from_rule_table",
        "upstream_implementation_reproduced": False,
        "confidence": "heuristic_candidate",
        "causal": False,
    }


def _cube_rule(metrics: Mapping[str, float], role: str | None) -> dict[str, Any]:
    if role not in {"aic", "mixed"}:
        return {
            "status": "not_applicable",
            "reason": "not_an_ai_core_event",
            "matched": False,
            "candidate_bound": None,
            "causal": False,
        }
    missing = [name for name in _CUBE_REQUIRED if _number(metrics, name) is None]
    if missing:
        return {
            "status": "metric_missing",
            "missing": missing,
            "required": list(_CUBE_REQUIRED),
            "matched": False,
            "candidate_bound": None,
            "causal": False,
        }
    mac = float(metrics["aic_mac_ratio"])
    mte2 = float(metrics["aic_mte2_ratio"])
    meta = _rule_meta("cube", {"aic_mac_ratio": 0.8, "aic_mte2_ratio": 0.95})
    matched = mac >= 0.8 or mte2 >= 0.95
    bound = None
    if mac >= 0.8 and mte2 >= 0.95:
        bound = "mac_and_mte2_bound"
    elif mac >= 0.8:
        bound = "mac_bound"
    elif mte2 >= 0.95:
        bound = "mte2_bound"
    return {**meta, "status": "available", "matched": matched, "candidate_bound": bound, "metrics": {"aic_mac_ratio": mac, "aic_mte2_ratio": mte2}}


def _vector_rule(metrics: Mapping[str, float], role: str | None) -> dict[str, Any]:
    if role not in {"aiv", "mixed"}:
        return {
            "status": "not_applicable",
            "reason": "not_an_ai_vector_event",
            "matched": False,
            "candidate_bound": None,
            "causal": False,
        }
    missing = [name for name in _VECTOR_REQUIRED if _number(metrics, name) is None]
    if missing:
        return {
            "status": "metric_missing",
            "missing": missing,
            "required": list(_VECTOR_REQUIRED),
            "matched": False,
            "candidate_bound": None,
            "causal": False,
        }
    vec = float(metrics["aiv_vec_ratio"])
    mte2 = float(metrics["aiv_mte2_ratio"])
    mte3 = float(metrics["aiv_mte3_ratio"])
    total = vec + mte2 + mte3
    meta = _rule_meta("vector", {"vec_plus_mte2_plus_mte3": 0.9, "individual": 0.7})
    matched = total >= 0.9 or max(vec, mte2, mte3) >= 0.7
    bound = None
    if total >= 0.9:
        bound = "vec_mte2_mte3_bound"
    elif vec >= 0.7:
        bound = "vec_bound"
    elif mte2 >= 0.7:
        bound = "mte2_bound"
    elif mte3 >= 0.7:
        bound = "mte3_bound"
    return {
        **meta,
        "status": "available",
        "matched": matched,
        "candidate_bound": bound,
        "metrics": {"aiv_vec_ratio": vec, "aiv_mte2_ratio": mte2, "aiv_mte3_ratio": mte3, "vec_plus_mte2_plus_mte3": total},
    }


def derive_pmu_evidence(
    metrics: Mapping[str, float],
    *,
    task_type: str | None = None,
    core: str | None = None,
    complete_in_window: bool,
    source: str | None = None,
    block_num: int | None = None,
    export_resolution: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return units, fractions, consistency, rules, and explicit scope."""
    folded = f"{task_type or ''} {core or ''}".upper()
    textual_role = None
    if "VECTOR" in folded or "AIV" in folded:
        textual_role = "aiv"
    elif "CORE" in folded or "AIC" in folded or "CUBE" in folded:
        textual_role = "aic"
    units = metric_units(metrics)
    aic = _scope_evidence(metrics, "aic", _AIC_PIPES, export_resolution or {})
    aiv = _scope_evidence(metrics, "aiv", _AIV_PIPES, export_resolution or {})
    aic_active = bool(aic.get("total_time_ns") and aic["total_time_ns"] > 0) or any(
        value > 0 for value in (aic.get("pipe_time_fractions") or {}).values()
    )
    aiv_active = bool(aiv.get("total_time_ns") and aiv["total_time_ns"] > 0) or any(
        value > 0 for value in (aiv.get("pipe_time_fractions") or {}).values()
    )
    if aic_active and aiv_active:
        role = "mixed"
    elif aiv_active:
        role = "aiv"
    elif aic_active:
        role = "aic"
    else:
        role = textual_role
    for scope, prefix in ((aic, "aic"), (aiv, "aiv")):
        scope["frequency"] = _frequency(
            scope.get("total_time_ns"),
            scope.get("total_cycles"),
            block_num,
        )
    cube = _cube_rule(metrics, role)
    vector = _vector_rule(metrics, role)
    cube_pct = _number(metrics, "cube_utilization_pct")
    return {
        "schema_version": PMU_ANALYSIS_SCHEMA,
        "attribution": "rule_candidate_not_root_cause",
        "causal": False,
        "task_role": role,
        "metric_units": units,
        "unmapped_metrics": sorted(name for name, unit in units.items() if unit == "unknown"),
        "scope": {
            "kind": "full_event",
            "source": source,
            "complete_in_window": bool(complete_in_window),
            "usable_for_clipped_window": bool(complete_in_window),
            "derived_for_clipped_window": bool(complete_in_window),
        },
        "aic": aic,
        "aiv": aiv,
        "cube_rule": cube,
        "vector_rule": vector,
        "cube_utilization_pct": {
            "value": cube_pct,
            "unit": "percent",
            "status": "present" if cube_pct is not None else "absent",
            "reason": None if cube_pct is not None else "not_exported_in_db",
            "not_equivalent_to": "aic_mac_ratio",
            "source": "kernel_details.csv:cube_utilization(%)" if cube_pct is not None else None,
        },
    }


__all__ = ["metric_units", "derive_pmu_evidence", "PMU_ANALYSIS_SCHEMA"]
