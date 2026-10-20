"""C7 ``mindie-profile-summary/1`` report assembly.

[Role]
  Turn measured observations into the compact, data-traceable C7 JSON
  contract consumed by other MindIE packages.  The report separates wall,
  busy union, kernel sum, exposed communication, host/idle gap, seven semantic
  blocks, rank medians, and the slowest rank.

[Invariants]
  * Missing data is ``None``; the report never substitutes zero or a model
    prediction for a measurement.
  * Modeling fields stay ``unmodeled`` unless ``analyze_profile`` receives an
    explicit provider and complete user-supplied kernel inputs; the report
    itself never invents an ideal lower bound or predicted time.
  * Every hotspot carries source sha256/table/filter/task ids where available;
    bounded locator samples are marked separately from full measured totals.
  * ``steady_steps`` contains only caller-supplied steps.  Untagged windows
    remain untagged in ``window_accounting``.

[Timeouts & Locks]
  O(n log n) medians and sorts over bounded observations/events; no I/O,
  except the optional atomic report write.

[Side-Effect Policy]
  Writes only the requested output JSON via a temporary file + ``os.replace``;
  no network and no mutation of source artifacts.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .accounting import hotspot_aggregates, percentile
from .classify import ALL_BLOCKS, MAPPING_RULES
from .cross_rank import aggregate_observations, cross_rank_stats, p50
from .events import Observation, PARSER_VERSION, REPORT_SCHEMA, SourceRef


def _ms(value: int | float | None) -> float | None:
    return None if value is None else round(float(value) / 1_000_000.0, 6)


def _source_kind(sources: Sequence[SourceRef]) -> str:
    kinds = {source.kind for source in sources}
    if kinds == {"db"}:
        return "ascend_pytorch_profiler_db"
    if kinds == {"csv"}:
        return "kernel_details_csv"
    return "mixed"


def _source_sha_map(sources: Sequence[SourceRef], kind: str) -> dict[str, str | None]:
    return {
        source.rank_id: source.sha256
        for source in sources
        if source.kind == kind
    }


def _schema_version(sources: Sequence[SourceRef]) -> str | dict[str, str]:
    values = sorted({source.schema_version for source in sources})
    if len(values) == 1:
        return values[0]
    return {source.rank_id: source.schema_version for source in sources}


def _metric_definitions() -> dict[str, str]:
    return {
        "request_wall_ms": "external client/server request duration; None unless explicitly supplied",
        "window_duration_ms": "query window length (window.end_ns - window.start_ns), not a request duration",
        "observed_event_span_ms": "max clipped event end - min clipped event start inside the explicit window",
        "device_busy_union_ms": "length of the union of clipped device event intervals; overlapping events counted once",
        "device_activity_sum_ms": "sum of clipped device event durations; overlapping events counted per event",
        "exposed_comm_ms": "communication union minus non-communication union; fused MoE communication is not exposed",
        "device_uncovered_ms": "observed_event_span minus device_busy_union; device idle, not a host-dispatch measurement",
        "host_dispatch_gap_ms": "measured Host/Runtime to device launch gap; None without correlate evidence",
        "total_cost_ms": "device_activity_sum + device_uncovered; not an end-to-end wall and not additive with exposed_comm",
        "clipped_event_count": "events crossed by a window boundary; kept in activity sums but excluded from per-call P50/P90/Max",
        "underfeed_bubble_ratio": "device_uncovered / observed_event_span",
        "block_union_ms": "per-block union of clipped events; shares are normalised over the sum of block unions",
        "wall_ms": "deprecated alias for observed_event_span_ms",
        "busy_union_ms": "deprecated alias for device_busy_union_ms",
        "kernel_sum_ms": "deprecated alias for device_activity_sum_ms",
        "host_gap_ms": "deprecated alias for device_uncovered_ms",
        "steady_steps": "only explicitly supplied steps; kernel details have no step-id column",
        "window_start_us/window_end_us": "offsets from window_origin_ns; exact bounds remain integer window_start_ns/window_end_ns",
    }


def _id_list(values: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def _filter_sql(locator: dict[str, Any], task_ids: Sequence[int], row_indices: Sequence[int], rowids: Sequence[int]) -> str:
    table = locator.get("table") or "kernel_details.csv"
    if table == "COMMUNICATION_OP":
        return f"COMMUNICATION_OP WHERE rowid IN ({_id_list(rowids)})"
    if table in {"COMPUTE_TASK_INFO", "COMMUNICATION_SCHEDULE_TASK_INFO"}:
        return f"{table} WHERE globalTaskId IN ({_id_list(task_ids)})"
    return f"kernel_details.csv WHERE row_index IN ({_id_list(row_indices)})"


def _hotspot_key(row: dict, group_by: str) -> tuple:
    if group_by == "kernel":
        return (row["rank_id"], row["semantic_block"], row["kernel_name"])
    return (
        row["rank_id"],
        row["semantic_block"],
        row["kernel_name"],
        row.get("input_shapes", ""),
        row.get("input_dtypes", ""),
        row.get("input_formats", ""),
    )


def _execution_engine(row: dict) -> str | None:
    engines = sorted(str(item) for item in row.get("execution_engines", ()) if item)
    if not engines:
        return None
    return engines[0] if len(engines) == 1 else "mixed"


def _bound_family(row: dict) -> str:
    families = sorted(str(item) for item in row.get("bound_families", ()) if item and item != "unknown")
    return families[0] if len(families) == 1 else "unknown"


def _hotspots(
    observations: Sequence[Observation],
    sources: dict[str, SourceRef],
    top_n: int,
    *,
    group_by: str = "variant",
) -> list[dict]:
    if group_by not in {"variant", "kernel"}:
        raise ValueError(f"group_by must be variant or kernel, got {group_by!r}")
    grouped: dict[tuple, dict] = {}
    total_duration = 0
    for observation in observations:
        if not observation.accounting.available:
            continue
        total_duration += observation.accounting.device_activity_sum_ns or 0
        identity_by_locator = {id(event.locator): event.identity() for event in observation.events}
        for row in hotspot_aggregates(observation.events, observation.window):
            enriched_locators = []
            for locator in row["locators"]:
                identity = identity_by_locator.get(id(locator)) or {}
                enriched_locators.append({**locator, "event_id": identity.get("event_id")})
            row["locators"] = enriched_locators
            key = _hotspot_key(row, group_by)
            current = grouped.get(key)
            if current is None:
                current = {
                    **row,
                    "durations": list(row["durations"]),
                    "locators": list(row["locators"]),
                    "execution_engines": set(row.get("execution_engines", ())),
                    "bound_families": set(row.get("bound_families", ())),
                    "last_observation": observation,
                    "window_labels": [observation.window.label or str(observation.window.step)],
                    "window_bounds_ns": [[observation.window.start_ns, observation.window.end_ns]],
                    "variant_inputs": {
                        (row.get("input_shapes", ""), row.get("input_dtypes", ""), row.get("input_formats", ""))
                    },
                }
                grouped[key] = current
            else:
                current["duration_ns"] += row["duration_ns"]
                current["count"] += row["count"]
                current["clipped_event_count"] += row["clipped_event_count"]
                current["durations"].extend(row["durations"])
                current["execution_engines"].update(row.get("execution_engines", ()))
                current["bound_families"].update(row.get("bound_families", ()))
                current["variant_inputs"].add(
                    (row.get("input_shapes", ""), row.get("input_dtypes", ""), row.get("input_formats", ""))
                )
                if len(current["locators"]) < 16:
                    current["locators"].extend(row["locators"][: 16 - len(current["locators"])])
                label = observation.window.label or str(observation.window.step)
                current["window_labels"].append(label)
                current["window_bounds_ns"].append([observation.window.start_ns, observation.window.end_ns])
    rows: list[dict] = []
    for row in grouped.values():
        observation = row["last_observation"]
        source = sources.get(row["rank_id"])
        locator = row["locators"][0] if row["locators"] else {}
        task_ids = sorted(
            {
                int(item["task_global_id"])
                for item in row["locators"]
                if item.get("task_global_id") is not None
            }
        )
        row_indices = sorted(
            {
                int(item["row_index"])
                for item in row["locators"]
                if item.get("row_index") is not None
            }
        )
        rowids = sorted(
            {
                int(item["rowid"])
                for item in row["locators"]
                if item.get("rowid") is not None
            }
        )
        event_ids = sorted(
            {
                str(item["event_id"])
                for item in row["locators"]
                if item.get("event_id")
            }
        )
        evidence = {
            "db_file": source.basename if source and source.kind == "db" else None,
            "csv_file": source.basename if source and source.kind == "csv" else None,
            "table": locator.get("table"),
            "filter_sql": _filter_sql(locator, task_ids, row_indices, rowids),
            "task_global_ids": task_ids,
            "row_indices": row_indices,
            "rowids": rowids,
            "ids_truncated": row["count"] > len(row["locators"]),
            "total_contributing_events": row["count"],
            "retained_event_locators": len(row["locators"]),
            "event_ids": event_ids,
            "event_id_count": len(event_ids),
            "event_ids_truncated": row["count"] > len(event_ids),
            "filter_scope": "locator_sample",
            "measurement_basis": "all clipped event occurrences in the requested windows; filter_sql selects the retained locator sample",
            "window_labels": row["window_labels"],
            "window_bounds_ns": row["window_bounds_ns"],
            "rank_id": row["rank_id"],
            "source_sha256": source.sha256 if source else None,
        }
        variant_inputs = sorted(row.get("variant_inputs", ()))
        variant_count = len(variant_inputs) if group_by == "kernel" else 1
        if group_by == "kernel" and variant_count > 1:
            input_shapes = None
            input_dtypes = None
            input_formats = None
        else:
            input_shapes = row.get("input_shapes") or None
            input_dtypes = row.get("input_dtypes") or None
            input_formats = row.get("input_formats") or None
        has_shape = bool(input_shapes)
        reason = (
            "model provider/inputs not configured; no ideal lower bound or prediction was requested"
            if has_shape
            else "missing input shape/dtype/layout and no model provider configured"
        )
        durations = row["durations"]
        rows.append(
            {
                "rank": observation.rank_index if observation.rank_index is not None else row["rank_id"],
                "rank_id": row["rank_id"],
                "rank_index": observation.rank_index,
                "semantic_block": row["semantic_block"],
                "kernel_name": row["kernel_name"],
                "input_shapes": input_shapes,
                "input_dtypes": input_dtypes,
                "input_formats": input_formats,
                "variant_count": variant_count,
                "variant_inputs": [
                    {"input_shapes": shapes, "input_dtypes": dtypes, "input_formats": formats}
                    for shapes, dtypes, formats in variant_inputs
                ] if group_by == "kernel" else [],
                "group_by": group_by,
                "execution_engine": _execution_engine(row),
                "bound_family": _bound_family(row),
                "measured_device_ms_p50": p50(_ms(value) for value in durations),
                "measured_device_ms_p90": _ms(percentile(durations, 0.9)),
                "measured_device_ms_max": _ms(max(durations) if durations else None),
                "measured_device_ms_total": _ms(row["duration_ns"]),
                "share_of_busy_pct": (
                    round(100.0 * row["duration_ns"] / total_duration, 6) if total_duration > 0 else None
                ),
                "share_basis": "sum of clipped event durations / device_activity_sum",
                "representative_window": observation.window.label or str(observation.window.step),
                "call_count": row["count"],
                "complete_call_count": len(durations),
                "clipped_event_count": row["clipped_event_count"],
                "distribution_basis": "complete_in_window_events_only",
                "modeling_alignment": {
                    "status": "unmodeled",
                    "ideal_hardware_lower_bound_ms": None,
                    "model_predicted_ms": None,
                    "model_provider": None,
                    "model_assumptions": None,
                    "unmodeled_reason": reason,
                },
                "wait_anchor_false_hotspot": None,
                "wait_anchor_assessment": "not_evaluated",
                "wait_anchor_reason": "exact event/window Wait-Anchor attachment was not evaluated for this hotspot",
                "wait_anchor_evidence": [],
                "wait_anchor_scope": None,
                "evidence_locator": evidence,
            }
        )
    rows.sort(key=lambda item: (item["measured_device_ms_total"] or 0.0, item["kernel_name"]), reverse=True)
    return rows[:top_n]


def _window_accounting(observations: Sequence[Observation]) -> list[dict]:
    rows: list[dict] = []
    for observation in observations:
        accounting = observation.accounting
        rows.append(
            {
                "rank_id": observation.rank_id,
                "rank_index": observation.rank_index,
                "step": observation.window.step,
                "label": observation.window.label,
                "window_origin_ns": observation.window.start_ns,
                "window_start_ns": observation.window.start_ns,
                "window_end_ns": observation.window.end_ns,
                "window_start_us": 0.0,
                "window_end_us": round(observation.window.span_ns / 1000.0, 3),
                "event_count": accounting.event_count,
                "complete_event_count": accounting.complete_event_count,
                "clipped_event_count": accounting.clipped_event_count,
                "comm_event_count": accounting.comm_event_count,
                "stream_count": accounting.stream_count,
                "request_wall_ms": _ms(accounting.request_wall_ns),
                "window_duration_ms": _ms(accounting.window_span_ns),
                "observed_event_span_ms": _ms(accounting.observed_event_span_ns),
                "device_busy_union_ms": _ms(accounting.busy_union_ns),
                "device_activity_sum_ms": _ms(accounting.device_activity_sum_ns),
                "exposed_comm_ms": _ms(accounting.exposed_comm_ns),
                "device_uncovered_ms": _ms(accounting.device_uncovered_ns),
                "host_dispatch_gap_ms": _ms(accounting.host_dispatch_gap_ns),
                "total_cost_ms": _ms(accounting.total_cost_ns),
                # Deprecated aliases: same measured values, explicit old names.
                "wall_ms": _ms(accounting.observed_event_span_ns),
                "busy_union_ms": _ms(accounting.busy_union_ns),
                "kernel_sum_ms": _ms(accounting.device_activity_sum_ns),
                "host_gap_ms": _ms(accounting.device_uncovered_ns),
                "exposed_aicpu_ms": _ms(accounting.exposed_aicpu_ns),
                "underfeed_bubble_ratio": accounting.underfeed_ratio,
                "aicpu_exposed_ratio": accounting.aicpu_exposed_ratio,
                "coverage_ratio": accounting.coverage_ratio,
                "largest_internal_bubble_ms": _ms(accounting.largest_internal_bubble_ns),
                "block_union_ms": {
                    block: (_ms(accounting.block_union_ns.get(block)) if accounting.available else None)
                    for block in ALL_BLOCKS
                },
                "block_share_pct": {
                    block: accounting.block_share_pct.get(block) for block in ALL_BLOCKS
                },
                "source": {
                    "kind": observation.source.kind,
                    "basename": observation.source.basename,
                    "sha256": observation.source.sha256,
                    "schema_version": observation.source.schema_version,
                },
                "query": {
                    "rows_read": observation.stats.rows_read,
                    "rows_selected": observation.stats.rows_selected,
                    "bad_rows": observation.stats.bad_rows,
                    "elapsed_ms": observation.stats.elapsed_ms,
                },
            }
        )
    return rows


def _block_breakdown(observations: Sequence[Observation]) -> list[dict]:
    usable = [obs for obs in observations if obs.accounting.available]
    rows: list[dict] = []
    for block in ALL_BLOCKS:
        values = [
            _ms(obs.accounting.block_union_ns.get(block))
            for obs in usable
            if obs.accounting.block_union_ns.get(block) is not None
        ]
        shares = [
            obs.accounting.block_share_pct.get(block)
            for obs in usable
            if obs.accounting.block_share_pct.get(block) is not None
        ]
        rows.append(
            {
                "block": block,
                "busy_union_ms_p50": p50(values),
                "share_of_busy_pct": p50(shares),
                "mapping_rule": MAPPING_RULES[block],
                "event_count": sum(obs.accounting.block_event_count.get(block, 0) for obs in usable),
                "basis": (
                    "device uncovered = observed event span - device busy union"
                    if block == "Host_Gap"
                    else "per-block union; shares normalised over attributed block unions"
                ),
            }
        )
    return rows


def _by_step(observations: Sequence[Observation]) -> list[dict]:
    grouped: dict[int, list[Observation]] = {}
    for observation in observations:
        if observation.window.step is not None:
            grouped.setdefault(observation.window.step, []).append(observation)
    rows: list[dict] = []
    for step in sorted(grouped):
        aggregate = aggregate_observations(grouped[step])
        rows.append({"step": step, **aggregate})
    return rows


def _limitations_findings(limitations: Sequence[dict]) -> list[dict]:
    return [
        {
            "finding_type": "measurement_limitation",
            "severity": item.get("severity", "medium"),
            "confidence": "high",
            "summary": item.get("detail", ""),
            "evidence_ref": item.get("code", "limitation"),
            "limitation_code": item.get("code"),
        }
        for item in limitations
    ]


def build_report(
    *,
    sources: Sequence[SourceRef],
    observations: Sequence[Observation],
    requested_steps: Sequence[int],
    scene: dict[str, Any],
    scene_provided: bool,
    limitations: Sequence[dict],
    query_timings: Sequence[dict],
    elapsed_ms: float,
    schema_rows: Sequence[dict],
    output_path: str | None,
    top_n: int,
    group_by: str = "variant",
) -> dict:
    steps = sorted({int(step) for step in requested_steps})
    step_observations = [
        observation for observation in observations if observation.window.step is not None
    ]
    step_aggregate = aggregate_observations(step_observations) if steps else aggregate_observations([])
    step_cross = cross_rank_stats(step_observations) if steps else cross_rank_stats([])
    analysis_observations = step_observations if steps else list(observations)
    cross = cross_rank_stats(analysis_observations)
    source_by_rank = {source.rank_id: source for source in sources}
    step_accounting = {
        "steady_steps": steps,
        "observed_steps": step_aggregate["steady_steps"],
        "step_status": "explicit" if steps else "not_provided",
        "rank_scope": sorted({observation.rank_id for observation in observations}),
        "aggregation": "p50_over_step_rank_observations",
        "request_wall_ms_p50": step_aggregate["request_wall_ms_p50"],
        "observed_event_span_ms_p50": step_aggregate["wall_ms_p50"],
        "device_busy_union_ms_p50": step_aggregate["busy_union_ms_p50"],
        "device_activity_sum_ms_p50": step_aggregate["kernel_sum_ms_p50"],
        "exposed_comm_ms_p50": step_aggregate["exposed_comm_ms_p50"],
        "device_uncovered_ms_p50": step_aggregate["host_gap_ms_p50"],
        "total_cost_ms_p50": step_aggregate["total_cost_ms_p50"],
        "clipped_event_count": step_aggregate["clipped_event_count"],
        # Deprecated aliases.
        "wall_ms_p50": step_aggregate["wall_ms_p50"],
        "busy_union_ms_p50": step_aggregate["busy_union_ms_p50"],
        "kernel_sum_ms_p50": step_aggregate["kernel_sum_ms_p50"],
        "host_gap_ms_p50": step_aggregate["host_gap_ms_p50"],
        "underfeed_bubble_ratio": step_aggregate["underfeed_bubble_ratio_p50"],
        "aicpu_exposed_ratio": step_aggregate["aicpu_exposed_ratio_p50"],
        "coverage_ratio_p50": step_aggregate["coverage_ratio_p50"],
        "slowest_rank": step_cross["slowest_rank"],
        "by_step": _by_step(observations),
        "by_rank": step_cross["rank_summaries"] if steps else cross["rank_summaries"],
        "cross_rank": {
            "rank_count_with_data": step_cross["rank_count_with_data"] if steps else cross["rank_count_with_data"],
            "wall_skew_ms": step_cross["wall_skew_ms"] if steps else cross["wall_skew_ms"],
            "wall_ratio": step_cross["wall_ratio"] if steps else cross["wall_ratio"],
        },
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "source_identity": {
            "source_kind": _source_kind(sources),
            "db_files_sha256": _source_sha_map(sources, "db"),
            "csv_files_sha256": _source_sha_map(sources, "csv"),
            "exporter_schema_version": _schema_version(sources),
            "parser_version": PARSER_VERSION,
            "query_elapsed_ms": round(float(elapsed_ms), 6),
            "timing_scope": "schema probe + bounded explicit-window reads (plus sha256 only with --hash); capture, export, and first cold full-table scan are excluded",
            "cache_state": "path_size_mtime",
            "cache_scope": "content sha256 is opt-in (--hash); the default identity is path+size_bytes+mtime_ns+schema_version",
            "query_timings": list(query_timings),
            "sources": [
                {
                    "rank_id": source.rank_id,
                    "kind": source.kind,
                    "basename": source.basename,
                    "path": source.path,
                    "sha256": source.sha256,
                    "size_bytes": source.size_bytes,
                    "mtime_ns": source.mtime_ns,
                    "schema_version": source.schema_version,
                    "schema_variant": source.schema_variant,
                    "header_sha256": source.header_sha256,
                    "optional_missing": list(source.optional_missing),
                    "notes": list(source.notes),
                    "content_identity": source.content_identity(),
                }
                for source in sources
            ],
            "source_fingerprints": {
                source.rank_id: source.content_identity() for source in sources
            },
            "schema": list(schema_rows),
        },
        "scene": {
            "framework": None,
            "phase": None,
            "tp": None,
            "dp": None,
            "graph_mode": None,
            **(scene or {}),
            "clock_domain": (scene or {}).get("clock_domain") or "unknown",
            "scene_source": "provided" if scene_provided else "not_provided",
        },
        "step_accounting": step_accounting,
        "semantic_block_breakdown": _block_breakdown(analysis_observations),
        "top_hotspot_kernels": _hotspots(
            analysis_observations, source_by_rank, top_n, group_by=group_by
        ),
        "hotspot_grouping": group_by,
        "window_accounting": _window_accounting(observations),
        "findings": _limitations_findings(limitations),
        "limitations": list(limitations),
        "metric_definitions": _metric_definitions(),
        "time_origin_policy": (
            "window_accounting.window_start_us/window_end_us are relative to window_origin_ns; "
            "exact integer nanosecond bounds are window_start_ns/window_end_ns"
        ),
        "artifacts": {
            "report_json": output_path,
            "report_md": None,
            "report_html_v2": None,
            "remote_output_dir": str(Path(output_path).resolve().parent) if output_path else None,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return report


def write_report(report: dict, output_path: str | Path) -> Path:
    """Atomically write the C7 JSON; no partial file is left on failure."""

    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return destination
