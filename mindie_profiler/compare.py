"""C7 cross-run comparison for real ``mindie-profile-summary/1`` reports.

[Role]
Pure, read-only A/B comparison of C7 dictionaries produced by
``mindie_profiler.analyze_profile``.  It aligns real ``window_accounting``
rows by ``(step, rank_id)`` and compares step, block, and optional hierarchy
measurements.  It never reads a DB, contacts a host, or re-runs a workload.

[Invariants]
- Both inputs must declare ``schema_version == mindie-profile-summary/1``.
- Timing is compared only when scene facts, clock domain, rank scope, and
  steady steps are compatible.  ``clock_domain == "unknown"`` is never a match.
- Partial windows are re-aggregated only from the common real
  ``window_accounting`` rows; a declared p50 is not re-used for a different
  step/rank set.  Missing observations are reported, never imputed.
- Missing values stay ``None``; this module never substitutes zero.
- Confidence intervals use only aligned per-step/per-rank observations.
- Layer/structure alignment needs a stable event-sequence shape and
  ``layer_id + ordinal`` identity.  It never matches by display name alone and
  never treats DB content hashes or query timing/cache fields as structural
  identity.  Missing structure evidence yields no layer conclusion but does
  not invalidate an otherwise strict step comparison.

[Timeouts & Locks]
None: this module is CPU-only, has no locks, performs no I/O, and never sleeps.

[Side-Effect Policy]
No filesystem, network, environment, process, or mutable-global side effects.
Input dictionaries are treated as immutable and are not modified.
"""

from __future__ import annotations

import hashlib
import json
import math
from statistics import median
from typing import Any, Iterable, Mapping

from .compare_acceptance import (
    build_median_tables,
    classify_operator_sequence,
    compare_run_binding,
)

SUMMARY_SCHEMA = "mindie-profile-summary/1"
COMPARISON_SCHEMA = "mindie-profile-comparison/1"

_ROW_METRICS = {
    "wall_ms_p50": "wall_ms",
    "busy_union_ms_p50": "busy_union_ms",
    "kernel_sum_ms_p50": "kernel_sum_ms",
    "exposed_comm_ms_p50": "exposed_comm_ms",
    "host_gap_ms_p50": "host_gap_ms",
    "total_cost_ms_p50": "total_cost_ms",
}
_RATIO_METRICS = ("underfeed_bubble_ratio", "aicpu_exposed_ratio")
_BLOCK_ORDER = ("Attention", "MoE", "Comm", "Norm", "Copy_Layout", "Unclassified", "Host_Gap")
_REQUIRED_SCENE = ("framework", "phase")
_OPTIONAL_SCENE = ("tp", "dp", "graph_mode")


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _lim(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **{k: v for k, v in extra.items() if v is not None}}


def _dedupe(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = (item.get("code"), item.get("source"), item.get("field"), repr(item.get("details")))
        if key not in seen:
            seen.add(key)
            out.append(dict(item))
    return out


def _canon(value: Any) -> str | None:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError):
        return None


def _hash(value: Any) -> str | None:
    text = _canon(value)
    return hashlib.sha256(text.encode()).hexdigest() if text is not None else None


def _schema_bad(summary: Any, label: str) -> dict[str, Any] | None:
    if not isinstance(summary, Mapping):
        return _lim("summary_not_an_object", f"{label} summary is not an object", source=label)
    if summary.get("schema_version") != SUMMARY_SCHEMA:
        return _lim(
            "unsupported_schema_version",
            f"{label} schema_version is not {SUMMARY_SCHEMA}",
            source=label,
            field="schema_version",
            details={"expected": SUMMARY_SCHEMA, "actual": summary.get("schema_version")},
        )
    return None


def _scene(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    value = summary.get("scene")
    return value if isinstance(value, Mapping) else {}


def _input_identity(summary: Mapping[str, Any]) -> Any:
    value = _scene(summary).get("input_signature")
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() and value.strip().lower() != "unknown" else None
    if isinstance(value, (Mapping, list, tuple, set)):
        return value if value else None
    return value


def _rank_scope(summary: Mapping[str, Any]) -> tuple[list[str] | None, dict[str, Any] | None]:
    accounting = summary.get("step_accounting")
    raw = accounting.get("rank_scope") if isinstance(accounting, Mapping) else None
    if not isinstance(raw, list) or not raw or any(
        not isinstance(item, str) or not item for item in raw
    ):
        return None, _lim("rank_scope_unavailable", "step_accounting.rank_scope must be a non-empty string list")
    return sorted(set(raw)), None


def _steady_steps(summary: Mapping[str, Any]) -> tuple[list[int] | None, dict[str, Any] | None]:
    accounting = summary.get("step_accounting")
    raw = accounting.get("steady_steps") if isinstance(accounting, Mapping) else None
    if not isinstance(raw, list) or not raw or any(_int(item) is None for item in raw):
        return None, _lim("steady_steps_unavailable", "step_accounting.steady_steps must be a non-empty integer list")
    return sorted(set(_int(item) for item in raw)), None


def _window_rows(summary: Mapping[str, Any]) -> tuple[dict[tuple[int, str], Mapping[str, Any]] | None, list[dict[str, Any]]]:
    raw = summary.get("window_accounting")
    if not isinstance(raw, list):
        return None, [_lim("window_accounting_missing", "window_accounting must be a list")]
    rows: dict[tuple[int, str], Mapping[str, Any]] = {}
    limits: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            limits.append(_lim("window_row_invalid", "window_accounting contains a non-object row"))
            continue
        step = _int(item.get("step"))
        rank_id = item.get("rank_id")
        if step is None:
            continue
        if not isinstance(rank_id, str) or not rank_id:
            limits.append(_lim("window_row_rank_invalid", "window row has no rank_id"))
            continue
        key = (step, rank_id)
        if key in rows:
            limits.append(_lim("window_row_duplicate", f"duplicate window row {key!r}",
                               details={"step": step, "rank_id": rank_id}))
            continue
        rows[key] = item
    return rows, limits


def _block_declared(summary: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    raw = summary.get("semantic_block_breakdown")
    if not isinstance(raw, list):
        return {}, [_lim("semantic_block_breakdown_missing", "semantic_block_breakdown is not a list")]
    result: dict[str, Mapping[str, Any]] = {}
    limits: list[dict[str, Any]] = []
    for item in raw:
        name = item.get("block") if isinstance(item, Mapping) else None
        if not isinstance(name, str) or not name:
            limits.append(_lim("semantic_block_name_missing", "semantic block has no name"))
        elif name in result:
            limits.append(_lim("semantic_block_duplicate", f"{name} appears twice"))
        else:
            result[name] = item
    return result, limits

def _source_content(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    identity = summary.get("source_identity")
    if not isinstance(identity, Mapping):
        return None
    content: dict[str, Any] = {}
    for key in ("db_files_sha256", "csv_files_sha256"):
        value = identity.get(key)
        if isinstance(value, Mapping) and value:
            content[key] = dict(value)
    sources = identity.get("sources")
    if isinstance(sources, list):
        hashes = {
            item["rank_id"]: item["sha256"]
            for item in sources
            if isinstance(item, Mapping)
            and isinstance(item.get("rank_id"), str)
            and isinstance(item.get("sha256"), str)
        }
        if hashes:
            content["sources_sha256"] = hashes
    return content or None


def _structure_parts(summary: Mapping[str, Any]) -> tuple[Any, list[Mapping[str, Any]] | None]:
    container = summary.get("structure")
    if not isinstance(container, Mapping):
        return None, None
    shape = container.get("event_sequence_shape")
    raw_layers = container.get("layers")
    layers = [item for item in raw_layers if isinstance(item, Mapping)] if isinstance(raw_layers, list) else None
    return shape, layers


def _layer_identity(layer: Mapping[str, Any]) -> tuple[str | None, int | None, str | None]:
    layer_id = layer.get("layer_id")
    ordinal = _int(layer.get("ordinal"))
    name = layer.get("name") if isinstance(layer.get("name"), str) else None
    return (layer_id if isinstance(layer_id, str) and layer_id else None, ordinal, name)


def _layer_ids(layers: list[Mapping[str, Any]]) -> tuple[list[tuple[str, int]] | None, list[str]]:
    ids: list[tuple[str, int]] = []
    names: list[str] = []
    for layer in layers:
        layer_id, ordinal, name = _layer_identity(layer)
        if layer_id is None or ordinal is None:
            names.append(name or "<unnamed>")
            continue
        ids.append((layer_id, ordinal))
        names.append(name or layer_id)
    if len(ids) != len(layers):
        return None, names
    return ids, names


def _structural_reuse(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    b_shape, b_layers = _structure_parts(baseline)
    c_shape, c_layers = _structure_parts(candidate)
    result: dict[str, Any] = {
        "status": "not_available",
        "allowed": False,
        "reason_codes": [],
        "baseline_source_content": _source_content(baseline),
        "candidate_source_content": _source_content(candidate),
        "event_sequence_shape": {"baseline_sha256": _hash(b_shape) if b_shape is not None else None,
                                 "candidate_sha256": _hash(c_shape) if c_shape is not None else None},
    }
    b_ids = _layer_ids(b_layers)[0] if b_layers is not None else None
    c_ids = _layer_ids(c_layers)[0] if c_layers is not None else None
    if b_shape is None and c_shape is None and b_ids is None and c_ids is None:
        result["reason_codes"] = ["structure_evidence_unavailable"]
        return result
    reasons: list[str] = []
    if b_shape is None or c_shape is None:
        reasons.append("event_sequence_shape_unavailable")
    elif _hash(b_shape) != _hash(c_shape):
        reasons.append("event_sequence_shape_mismatch")
    if b_ids is None or c_ids is None or not b_ids or not c_ids:
        reasons.append("layer_identity_unavailable")
    elif sorted(b_ids) != sorted(c_ids):
        reasons.append("layer_identity_mismatch")
    result["reason_codes"] = reasons
    result["status"] = "rejected" if reasons else "verified"
    result["allowed"] = not reasons
    return result


def _cdf(k: int, n: int) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) for i in range(k + 1)) / (2**n)


def _interval(values: list[float]) -> list[float] | None:
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


def _metric(unit: str, baseline: Any, candidate: Any,
            b_values: list[float] | None = None, c_values: list[float] | None = None) -> dict[str, Any]:
    b_value, c_value = _num(baseline), _num(candidate)
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
        b_interval, c_interval = _interval(b_values), _interval(c_values)
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


def _column_values(rows: Mapping[tuple[int, str], Mapping[str, Any]],
                   keys: Iterable[tuple[int, str]], field: str) -> tuple[list[float], list[tuple[int, str]]]:
    values: list[float] = []
    missing: list[tuple[int, str]] = []
    for key in keys:
        value = _num(rows[key].get(field))
        if value is None:
            missing.append(key)
        else:
            values.append(value)
    return values, missing


def _block_values(rows: Mapping[tuple[int, str], Mapping[str, Any]],
                  keys: Iterable[tuple[int, str]], block: str,
                  field: str) -> tuple[list[float], list[tuple[int, str]]]:
    values: list[float] = []
    missing: list[tuple[int, str]] = []
    for key in keys:
        mapping = rows[key].get(field)
        value = _num(mapping.get(block)) if isinstance(mapping, Mapping) else None
        if value is None:
            missing.append(key)
        else:
            values.append(value)
    return values, missing


def _not_comparable(matching: Mapping[str, Any], applicability: Mapping[str, Any],
                    reasons: list[dict[str, Any]], limitations: list[dict[str, Any]],
                    reuse: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": COMPARISON_SCHEMA,
        "status": "not_comparable",
        "comparable": False,
        "partial": False,
        "reasons": reasons,
        "limitations": _dedupe([*limitations, *reasons]),
        "matching": dict(matching),
        "applicability": dict(applicability),
        "structural_reuse": dict(reuse) if reuse is not None else None,
        "step_accounting": None,
        "semantic_blocks": None,
        "hierarchy": None,
    }


def _hierarchy(baseline: Mapping[str, Any], candidate: Mapping[str, Any],
               reuse: Mapping[str, Any]) -> dict[str, Any]:
    if not reuse.get("allowed"):
        status = "not_available" if reuse.get("status") == "not_available" else "not_comparable"
        return {
            "status": status,
            "layers": [],
            "baseline_source_content": reuse.get("baseline_source_content"),
            "candidate_source_content": reuse.get("candidate_source_content"),
            "limitations": [_lim(
                "structure_evidence_unavailable" if status == "not_available" else "structure_reuse_not_established",
                "no layer conclusion is possible without verified event-shape and layer identity evidence",
                details={"reason_codes": list(reuse.get("reason_codes", []))},
            )],
        }
    _, b_layers = _structure_parts(baseline)
    _, c_layers = _structure_parts(candidate)
    assert b_layers is not None and c_layers is not None
    b_map = {(layer_id, ordinal): layer for layer in b_layers
             for layer_id, ordinal, _ in [_layer_identity(layer)] if layer_id is not None and ordinal is not None}
    c_map = {(layer_id, ordinal): layer for layer in c_layers
             for layer_id, ordinal, _ in [_layer_identity(layer)] if layer_id is not None and ordinal is not None}
    results: list[dict[str, Any]] = []
    limits: list[dict[str, Any]] = []
    for key in sorted(b_map):
        b_layer, c_layer = b_map[key], c_map.get(key)
        name = _layer_identity(b_layer)[2] or key[0]
        if c_layer is None:
            limits.append(_lim("layer_missing_in_candidate", f"layer {key!r} has no counterpart",
                               details={"layer_id": key[0], "ordinal": key[1]}))
            results.append({"layer": name, "layer_id": key[0], "ordinal": key[1],
                            "status": "not_comparable", "reason": "layer_missing_in_candidate"})
            continue
        b_metrics = b_layer.get("metrics") if isinstance(b_layer.get("metrics"), Mapping) else b_layer
        c_metrics = c_layer.get("metrics") if isinstance(c_layer.get("metrics"), Mapping) else c_layer
        metric_results = {
            metric: _metric("ms", b_metrics.get(metric), c_metrics.get(metric))
            for metric in ("busy_union_ms_p50", "wall_ms_p50", "host_gap_ms_p50")
        }
        compared = all(item["status"] == "compared" for item in metric_results.values())
        results.append({
            "layer": name,
            "layer_id": key[0],
            "ordinal": key[1],
            "status": "compared" if compared else "not_comparable",
            "match_basis": "event_sequence_shape+layer_id+ordinal",
            "metrics": metric_results,
        })
        if not compared:
            limits.append(_lim("layer_metric_missing", f"layer {key!r} has a missing metric"))
    return {
        "status": "comparable" if results and all(item["status"] == "compared" for item in results)
        and not limits else "comparable_with_limitations" if any(
            item["status"] == "compared" for item in results
        ) else "not_comparable",
        "layers": results,
        "baseline_source_content": reuse.get("baseline_source_content"),
        "candidate_source_content": reuse.get("candidate_source_content"),
        "limitations": limits,
    }


def compare_profiles(
    baseline: Any, candidate: Any, *, require_binding: bool = False
) -> dict[str, Any]:
    """Compare two real C7 reports; incompatibility is returned, never raised."""
    hard: list[dict[str, Any]] = []
    limits: list[dict[str, Any]] = []
    matching: dict[str, Any] = {}
    applicability: dict[str, Any] = {}
    for label, summary in (("baseline", baseline), ("candidate", candidate)):
        problem = _schema_bad(summary, label)
        if problem:
            hard.append(problem)
    if hard:
        return _not_comparable(matching, applicability, hard, limits)
    assert isinstance(baseline, Mapping) and isinstance(candidate, Mapping)

    scene = _scene(baseline)
    other_scene = _scene(candidate)
    scene_source = scene.get("scene_source")
    if scene_source != "provided" or other_scene.get("scene_source") != "provided":
        hard.append(_lim("scene_not_provided", "scene facts were not provided; no scene match is assumed",
                         details={"baseline_scene_source": scene_source,
                                  "candidate_scene_source": other_scene.get("scene_source")}))
    matching["scene"] = {}
    for field in (*_REQUIRED_SCENE, *_OPTIONAL_SCENE):
        b_value, c_value = scene.get(field), other_scene.get(field)
        entry = {"baseline": b_value, "candidate": c_value, "status": "match"}
        if b_value is None or c_value is None:
            entry["status"] = "unavailable"
            limits.append(_lim("scene_field_unavailable", f"scene.{field} is missing from at least one report",
                               field=f"scene.{field}"))
            if field in _REQUIRED_SCENE:
                hard.append(_lim(f"scene_{field}_unavailable", f"scene.{field} is required for comparison"))
        elif b_value != c_value:
            entry["status"] = "mismatch"
            hard.append(_lim(f"scene_{field}_mismatch", f"scene.{field} differs",
                             field=f"scene.{field}", details={"baseline": b_value, "candidate": c_value}))
        matching["scene"][field] = entry

    b_clock, c_clock = scene.get("clock_domain"), other_scene.get("clock_domain")
    clock_entry = {"baseline": b_clock, "candidate": c_clock, "status": "match"}
    if b_clock is None or c_clock is None or b_clock == "unknown" or c_clock == "unknown":
        clock_entry["status"] = "unknown"
        hard.append(_lim("clock_domain_unknown", "clock_domain is unknown; timing is not comparable"))
    elif b_clock != c_clock:
        clock_entry["status"] = "mismatch"
        hard.append(_lim("clock_domain_mismatch", "clock_domain differs",
                         details={"baseline": b_clock, "candidate": c_clock}))
    matching["clock_domain"] = clock_entry

    b_input, c_input = _input_identity(baseline), _input_identity(candidate)
    input_entry = {
        "baseline_sha256": _hash(b_input) if b_input is not None else None,
        "candidate_sha256": _hash(c_input) if c_input is not None else None,
        "status": "unavailable",
    }
    if b_input is None and c_input is None:
        limits.append(_lim("input_identity_unavailable",
                           "C7 reports do not carry an input signature; workload equality is unverified"))
    elif b_input is None or c_input is None:
        limits.append(_lim("input_identity_incomplete",
                           "only one report declares scene.input_signature"))
    elif input_entry["baseline_sha256"] is None or input_entry["candidate_sha256"] is None:
        limits.append(_lim("input_identity_unhashable",
                           "scene.input_signature cannot be canonicalized for exact comparison"))
    elif input_entry["baseline_sha256"] != input_entry["candidate_sha256"]:
        input_entry["status"] = "mismatch"
        hard.append(_lim("input_identity_mismatch", "input signatures differ"))
    else:
        input_entry["status"] = "match"
    matching["input"] = input_entry

    binding_entry, binding_hard, binding_limits = compare_run_binding(
        baseline, candidate, required=require_binding
    )
    matching["run_binding"] = binding_entry
    hard.extend(binding_hard)
    limits.extend(binding_limits)

    b_ranks, b_rank_problem = _rank_scope(baseline)
    c_ranks, c_rank_problem = _rank_scope(candidate)
    rank_entry = {"baseline": b_ranks, "candidate": c_ranks, "common": None, "status": "unknown"}
    matching["rank_scope"] = rank_entry
    if b_rank_problem or c_rank_problem:
        hard.extend(item for item in (b_rank_problem, c_rank_problem) if item)
    else:
        assert b_ranks is not None and c_ranks is not None
        common_ranks = sorted(set(b_ranks) & set(c_ranks))
        rank_entry["common"] = common_ranks
        if not common_ranks:
            rank_entry["status"] = "disjoint"
            hard.append(_lim("rank_scope_disjoint", "the reports share no rank_id"))
        elif b_ranks == c_ranks:
            rank_entry["status"] = "match"
        else:
            rank_entry["status"] = "partial_overlap"
            limits.append(_lim("rank_scope_partial_overlap",
                               "rank scopes differ; only the common rank_id set is compared",
                               details={"baseline": b_ranks, "candidate": c_ranks}))

    b_slow = baseline.get("step_accounting", {}).get("slowest_rank")
    c_slow = candidate.get("step_accounting", {}).get("slowest_rank")
    slowest = {
        "baseline": b_slow.get("rank_id") if isinstance(b_slow, Mapping) else None,
        "candidate": c_slow.get("rank_id") if isinstance(c_slow, Mapping) else None,
        "status": "unavailable",
    }
    if slowest["baseline"] is not None and slowest["candidate"] is not None:
        slowest["status"] = "match" if slowest["baseline"] == slowest["candidate"] else "changed"
    matching["slowest_rank"] = slowest

    b_steps, b_step_problem = _steady_steps(baseline)
    c_steps, c_step_problem = _steady_steps(candidate)
    step_entry: dict[str, Any] = {
        "baseline": b_steps, "candidate": c_steps, "common": None, "missing": [],
        "alignment": None, "basis": None, "status": "unknown",
    }
    matching["steps"] = step_entry
    if b_step_problem or c_step_problem:
        hard.extend(item for item in (b_step_problem, c_step_problem) if item)
    if hard:
        return _not_comparable(matching, applicability, hard, limits)
    assert b_steps is not None and c_steps is not None
    common_steps = sorted(set(b_steps) & set(c_steps))
    step_entry["common"] = common_steps
    if not common_steps:
        step_entry["status"] = "disjoint"
        hard.append(_lim("steady_step_windows_disjoint", "the reports share no steady step"))
        return _not_comparable(matching, applicability, hard, limits)
    if set(b_steps) == set(c_steps):
        step_entry["status"] = "same_steps"
    else:
        step_entry["status"] = "partial_overlap"
        limits.append(_lim("partial_step_overlap",
                           "steady steps differ; only common steps are compared",
                           details={"baseline": b_steps, "candidate": c_steps}))

    b_rows, b_row_limits = _window_rows(baseline)
    c_rows, c_row_limits = _window_rows(candidate)
    limits.extend(b_row_limits)
    limits.extend(c_row_limits)
    if b_rows is None or c_rows is None:
        hard.append(_lim("window_accounting_unavailable", "both reports need window_accounting rows"))
        return _not_comparable(matching, applicability, hard, limits)
    assert b_rows is not None and c_rows is not None
    assert b_ranks is not None and c_ranks is not None
    common_ranks = sorted(set(b_ranks) & set(c_ranks))
    expected_keys = [(step, rank) for step in common_steps for rank in common_ranks]
    missing = [key for key in expected_keys if key not in b_rows or key not in c_rows]
    aligned = [key for key in expected_keys if key in b_rows and key in c_rows]
    step_entry["missing"] = [{"step": step, "rank_id": rank} for step, rank in missing]
    step_entry["alignment"] = "complete" if not missing else "incomplete"
    if missing:
        limits.append(_lim("incomplete_common_observations",
                           "some common step/rank observations are absent; they are excluded, not imputed",
                           details={"missing": step_entry["missing"]}))
    if not aligned:
        hard.append(_lim("no_aligned_window_observations",
                         "no common step/rank observation exists in both reports"))
        return _not_comparable(matching, applicability, hard, limits)
    exact = (set(b_steps) == set(c_steps) and set(b_ranks) == set(c_ranks) and not missing)
    step_entry["basis"] = "declared_p50" if exact else "recomputed_from_window_accounting"
    applicability.update({
        "scope": "common steady steps and common rank_id set",
        "baseline_steps": b_steps,
        "candidate_steps": c_steps,
        "compared_steps": common_steps,
        "baseline_ranks": b_ranks,
        "candidate_ranks": c_ranks,
        "compared_ranks": common_ranks,
        "aligned_observation_count": len(aligned),
        "missing_observation_count": len(missing),
        "step_basis": step_entry["basis"],
        "input_identity": input_entry["status"],
        "notes": [],
    })
    if not exact:
        applicability["notes"].append(
            "medians are re-aggregated from aligned window_accounting observations"
        )
    if input_entry["status"] != "match":
        applicability["notes"].append("workload identity is unverified")

    accounting_b = baseline["step_accounting"]
    accounting_c = candidate["step_accounting"]
    assert isinstance(accounting_b, Mapping) and isinstance(accounting_c, Mapping)
    metrics: dict[str, Any] = {}
    partial = not exact
    for key, field in _ROW_METRICS.items():
        b_values, b_missing = _column_values(b_rows, aligned, field)
        c_values, c_missing = _column_values(c_rows, aligned, field)
        if b_missing or c_missing:
            metrics[key] = _metric("ms", None, None, b_values, c_values)
            metrics[key]["reason"] = "incomplete_observation_coverage"
            metrics[key]["missing_observations"] = {
                "baseline": [{"step": step, "rank_id": rank} for step, rank in b_missing],
                "candidate": [{"step": step, "rank_id": rank} for step, rank in c_missing],
            }
            limits.append(_lim("metric_incomplete_observation_coverage",
                               f"{key} is null for at least one aligned observation",
                               field=f"step_accounting.{key}"))
            partial = True
            continue
        b_value = accounting_b.get(key) if exact else median(b_values)
        c_value = accounting_c.get(key) if exact else median(c_values)
        metrics[key] = _metric("ms", b_value, c_value, b_values, c_values)
        if metrics[key]["status"] != "compared":
            partial = True
            limits.append(_lim("metric_value_missing", f"{key} is missing from a report p50",
                               field=f"step_accounting.{key}"))
    for key in _RATIO_METRICS:
        b_values, b_missing = _column_values(b_rows, aligned, key)
        c_values, c_missing = _column_values(c_rows, aligned, key)
        if b_missing or c_missing:
            metrics[key] = _metric("ratio", None, None, b_values, c_values)
            metrics[key]["reason"] = "incomplete_observation_coverage"
            limits.append(_lim("metric_incomplete_observation_coverage",
                               f"{key} is null for at least one aligned observation",
                               field=f"step_accounting.{key}"))
            partial = True
            continue
        b_value = accounting_b.get(key) if exact else median(b_values)
        c_value = accounting_c.get(key) if exact else median(c_values)
        metrics[key] = _metric("ratio", b_value, c_value, b_values, c_values)
        if metrics[key]["status"] != "compared":
            partial = True
            limits.append(_lim("metric_value_missing", f"{key} is missing from a report p50",
                               field=f"step_accounting.{key}"))

    b_blocks, b_block_limits = _block_declared(baseline)
    c_blocks, c_block_limits = _block_declared(candidate)
    limits.extend(b_block_limits)
    limits.extend(c_block_limits)
    block_names = sorted(set(b_blocks) | set(c_blocks) | set(_BLOCK_ORDER),
                         key=lambda name: (_BLOCK_ORDER.index(name) if name in _BLOCK_ORDER else len(_BLOCK_ORDER), name))
    block_results: list[dict[str, Any]] = []
    for name in block_names:
        b_block, c_block = b_blocks.get(name), c_blocks.get(name)
        b_values, b_missing = _block_values(b_rows, aligned, name, "block_union_ms")
        c_values, c_missing = _block_values(c_rows, aligned, name, "block_union_ms")
        entry = {
            "block": name, "status": "not_comparable",
            "baseline_busy_union_ms_p50": None, "candidate_busy_union_ms_p50": None,
            "delta_ms": None, "relative_delta_pct": None,
            "baseline_share_of_busy_pct": None, "candidate_share_of_busy_pct": None,
            "share_delta_pp": None, "confidence_interval": None,
            "confidence_interval_reason": "per_sample_observations_unavailable",
        }
        if b_missing or c_missing:
            entry["reason"] = "incomplete_block_observation_coverage"
            entry["missing_observations"] = {
                "baseline": [{"step": step, "rank_id": rank} for step, rank in b_missing],
                "candidate": [{"step": step, "rank_id": rank} for step, rank in c_missing],
            }
            limits.append(_lim("block_incomplete_observation_coverage",
                               f"{name} block union is null for at least one aligned observation",
                               details={"block": name}))
            partial = True
            block_results.append(entry)
            continue
        b_value = _num(b_block.get("busy_union_ms_p50")) if b_block is not None else None
        c_value = _num(c_block.get("busy_union_ms_p50")) if c_block is not None else None
        if not exact:
            b_value, c_value = median(b_values), median(c_values)
        if b_value is None or c_value is None:
            entry["reason"] = "semantic_block_value_missing"
            limits.append(_lim("semantic_block_value_missing", f"{name} has no declared busy_union_ms_p50",
                               details={"block": name}))
            partial = True
            block_results.append(entry)
            continue
        entry["baseline_busy_union_ms_p50"] = b_value
        entry["candidate_busy_union_ms_p50"] = c_value
        entry["delta_ms"] = c_value - b_value
        entry["status"] = "compared"
        if b_value != 0:
            entry["relative_delta_pct"] = (c_value - b_value) / abs(b_value) * 100.0
        metric_entry = _metric("ms", b_value, c_value, b_values, c_values)
        entry["confidence_interval"] = metric_entry["confidence_interval"]
        if metric_entry.get("confidence_interval_reason"):
            entry["confidence_interval_reason"] = metric_entry["confidence_interval_reason"]

        b_shares, b_share_missing = _block_values(b_rows, aligned, name, "block_share_pct")
        c_shares, c_share_missing = _block_values(c_rows, aligned, name, "block_share_pct")
        b_share = _num(b_block.get("share_of_busy_pct")) if b_block is not None else None
        c_share = _num(c_block.get("share_of_busy_pct")) if c_block is not None else None
        if not exact and b_shares and c_shares and not b_share_missing and not c_share_missing:
            b_share, c_share = median(b_shares), median(c_shares)
        entry["baseline_share_of_busy_pct"] = b_share
        entry["candidate_share_of_busy_pct"] = c_share
        if b_share is not None and c_share is not None:
            entry["share_delta_pp"] = c_share - b_share
        elif name != "Host_Gap" or (b_share is not None) != (c_share is not None):
            limits.append(_lim("semantic_block_share_missing", f"{name} has no comparable share",
                               details={"block": name}))
        block_results.append(entry)

    reuse = _structural_reuse(baseline, candidate)
    hierarchy = _hierarchy(baseline, candidate, reuse)
    limits.extend(hierarchy.get("limitations", []))
    operator_sequence = classify_operator_sequence(baseline, candidate)
    if operator_sequence.get("status") == "classified" and hierarchy.get("status") != "comparable":
        operator_sequence["reuse_recommendation"] = "unavailable_without_explicit_layers"
    limits.extend(operator_sequence.get("limitations", []))
    if not any(entry["status"] == "compared" for entry in metrics.values()):
        hard.append(_lim("no_comparable_step_metrics", "no step metric could be compared"))
        return _not_comparable(matching, applicability, hard, limits, reuse)

    if any(entry["status"] != "compared" for entry in metrics.values()):
        partial = True
    if hierarchy.get("status") != "comparable":
        partial = True
    applicability["confidence_intervals"] = (
        "computed_from_aligned_window_observations"
        if any(entry.get("confidence_interval") for entry in metrics.values())
        else "unavailable_fewer_than_6_aligned_observations"
    )
    status = "comparable_with_limitations" if partial or limits else "comparable"
    tables = build_median_tables(metrics, block_results, hierarchy)
    return {
        "schema_version": COMPARISON_SCHEMA,
        "status": status,
        "comparable": True,
        "partial": partial,
        "reasons": [],
        "limitations": _dedupe(limits),
        "matching": matching,
        "applicability": applicability,
        "structural_reuse": reuse,
        "step_accounting": {
            "status": "comparable" if all(
                entry["status"] == "compared" for entry in metrics.values()
            ) else "comparable_with_limitations",
            "metrics": metrics,
        },
        "semantic_blocks": {
            "status": "comparable" if block_results and all(
                entry["status"] == "compared" for entry in block_results
            ) else "comparable_with_limitations" if block_results else "not_comparable",
            "blocks": block_results,
        },
        "hierarchy": hierarchy,
        "operator_sequence": operator_sequence,
        "tables": tables,
    }


__all__ = ["COMPARISON_SCHEMA", "SUMMARY_SCHEMA", "compare_profiles"]
