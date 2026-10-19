"""Streaming kernel_details.csv source with verified header resolution.

[Role]
  Probe and stream the classic CANN ``kernel_details.csv`` projection, resolve
  the real 46-column export and its documented aliases, and return events for
  caller-supplied explicit windows.

[Invariants]
  * Header aliases are case-insensitive and come from the mapping file; a
    missing Name/start/duration(or end) column raises ``ProfilerSchemaError``.
  * CSV times are microseconds and are converted once to integer nanoseconds.
    Wait Time(us) becomes wait_ns; aic/aiv time columns become ns PMU values;
    ratios/cycles/percent stay in their native units.
  * The scan is single-pass and bounded by a row budget and wall-clock
    deadline; invalid timing rows are counted, not converted to zero.
  * No step id is fabricated: a CSV row only belongs to a window the caller
    supplied explicitly.

[Timeouts & Locks]
  ``timeout_s`` (default 30s) is checked every 1024 rows; no lock is held.

[Side-Effect Policy]
  Read-only streaming of the supplied CSV; no writes and no network.
"""

from __future__ import annotations

import csv
import hashlib
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from .events import Event, ProfilerQueryError, ProfilerSchemaError, QueryStats, Window
from .sources_db import load_mapping


@dataclass(frozen=True, slots=True)
class CsvSchema:
    path: str
    header: tuple[str, ...]
    header_sha256: str
    columns: Mapping[str, int]
    block_column: str | None
    mix_block_column: str | None
    wait_column: str | None
    pmu_columns: Mapping[str, int]
    pmu_units: Mapping[str, str]
    pmu_scales: Mapping[str, float]
    pmu_missing: tuple[str, ...]
    optional_missing: tuple[str, ...]
    size_bytes: int

    @property
    def schema_version(self) -> str:
        return f"kernel_details_csv_{len(self.header)}col"

    @property
    def schema_variant(self) -> str:
        variant = "csv_" + ("blockDim" if self.block_column == "Block Dim" else "verified_aliases")
        if self.wait_column:
            variant += "_wait"
        if self.pmu_columns:
            variant += "_pmu"
        return variant

    @property
    def source_kind(self) -> str:
        return "kernel_details_csv"


def _fold(value: str) -> str:
    return str(value).strip().lower()


def _resolve(header: list[str], aliases: list[str]) -> int | None:
    columns = {_fold(name): index for index, name in enumerate(header)}
    for alias in aliases:
        if _fold(alias) in columns:
            return columns[_fold(alias)]
    return None


def _header_digest(header: tuple[str, ...]) -> str:
    return hashlib.sha256("\x1f".join(header).encode("utf-8")).hexdigest()


def probe_csv(path: str | Path, *, mapping: Mapping[str, Any] | None = None) -> CsvSchema:
    mapping = mapping or load_mapping()
    csv_path = Path(path)
    if not csv_path.is_file():
        raise ProfilerSchemaError(f"kernel_details.csv not found: {csv_path}")
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
    except OSError as exc:
        raise ProfilerSchemaError(f"cannot read {csv_path.name}: {exc}") from exc
    if not header:
        raise ProfilerSchemaError(f"{csv_path.name}: empty CSV (no header)")
    header = tuple(str(name).strip() for name in header)
    aliases = mapping["csv"]["aliases"]
    required = mapping["csv"]["required_fields"]
    columns: dict[str, int] = {}
    missing: list[str] = []
    for logical in required:
        index = _resolve(list(header), aliases[logical])
        if index is None:
            missing.append(f"{logical} ({'/'.join(aliases[logical])})")
        else:
            columns[logical] = index
    if not any(_resolve(list(header), aliases[key]) is not None for key in mapping["csv"]["duration_or_end"]):
        missing.append("duration_us or end_us")
    if missing:
        raise ProfilerSchemaError(f"{csv_path.name}: unknown CSV schema, missing {', '.join(missing)}")
    for logical, alias_list in aliases.items():
        index = _resolve(list(header), alias_list)
        if index is not None:
            columns[logical] = index
    pmu_columns: dict[str, int] = {}
    pmu_units: dict[str, str] = {}
    pmu_scales: dict[str, float] = {}
    pmu_missing: list[str] = []
    for canonical, spec in mapping["pmu_metrics"].items():
        csv_column = spec.get("csv_column")
        if not csv_column:
            continue
        index = _resolve(list(header), [str(csv_column)])
        if index is None:
            pmu_missing.append(canonical)
            continue
        pmu_columns[canonical] = index
        pmu_units[canonical] = str(spec.get("unit", "unknown"))
        pmu_scales[canonical] = float(spec.get("csv_scale", 1.0))
    optional_missing = tuple(
        logical
        for logical in (
            "core", "type", "stream_id", "task_id", "device_id",
            "block_dim", "mix_block_dim", "input_shapes", "input_dtypes", "input_formats",
            "wait_us",
        )
        if logical not in columns
    )
    if not pmu_columns:
        optional_missing = optional_missing + ("pmu_columns",)
    return CsvSchema(
        path=str(csv_path.resolve()),
        header=header,
        header_sha256=_header_digest(header),
        columns=columns,
        block_column=header[columns["block_dim"]] if "block_dim" in columns else None,
        mix_block_column=header[columns["mix_block_dim"]] if "mix_block_dim" in columns else None,
        wait_column=header[columns["wait_us"]] if "wait_us" in columns else None,
        pmu_columns=pmu_columns,
        pmu_units=pmu_units,
        pmu_scales=pmu_scales,
        pmu_missing=tuple(pmu_missing),
        optional_missing=optional_missing,
        size_bytes=csv_path.stat().st_size,
    )


def _cell(row: list[str], index: int | None) -> str:
    if index is None or index >= len(row):
        return ""
    return str(row[index]).strip()


def _number(text: str) -> Decimal | None:
    if not text:
        return None
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def _us_to_ns(value: Decimal) -> int:
    return int((value * Decimal(1000)).to_integral_value())


class CsvSource:
    """A probed CSV.  Its single pass fills all requested windows."""

    def __init__(self, schema: CsvSchema, *, rank_id: str, rank_index: int | None) -> None:
        self.schema = schema
        self.rank_id = rank_id
        self.rank_index = rank_index

    def read_windows(
        self,
        windows: list[Window],
        *,
        max_rows: int = 5_000_000,
        timeout_s: float = 30.0,
    ) -> dict[int, tuple[list[Event], QueryStats]]:
        started = time.perf_counter()
        deadline = started + max(0.1, timeout_s)
        results = {
            index: (list(), QueryStats(rank_id=self.rank_id, window_label=window.label or str(window.step), budget=max_rows))
            for index, window in enumerate(windows)
        }
        stats = QueryStats(rank_id=self.rank_id, window_label="csv_scan", budget=max_rows)
        columns = self.schema.columns
        path = Path(self.schema.path)
        bad = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row_index, row in enumerate(reader):
                if row_index % 1024 == 0 and time.perf_counter() > deadline:
                    raise ProfilerQueryError(
                        f"{path.name}: CSV scan exceeded {timeout_s:.1f}s after {row_index} rows"
                    )
                if row_index >= max_rows:
                    raise ProfilerQueryError(
                        f"{path.name}: CSV row budget {max_rows} exceeded; narrow the capture or raise --max-rows"
                    )
                stats.rows_read += 1
                start_us = _number(_cell(row, columns.get("start_us")))
                duration_us = _number(_cell(row, columns.get("duration_us")))
                if duration_us is None:
                    end_us = _number(_cell(row, columns.get("end_us")))
                    duration_us = None if end_us is None or start_us is None else end_us - start_us
                if start_us is None or duration_us is None or duration_us < 0:
                    bad += 1
                    continue
                if duration_us == 0:
                    continue
                start_ns = _us_to_ns(start_us)
                end_ns = start_ns + _us_to_ns(duration_us)
                if end_ns <= start_ns:
                    bad += 1
                    continue
                name = _cell(row, columns.get("name"))
                core = _cell(row, columns.get("core"))
                op_type = _cell(row, columns.get("type")) or core
                wait_us = _number(_cell(row, columns.get("wait_us")))
                if self.schema.wait_column is None:
                    wait_ns, wait_status = None, "not_available"
                elif wait_us is None:
                    wait_ns, wait_status = None, "absent"
                elif wait_us < 0:
                    wait_ns, wait_status = None, "invalid_value"
                else:
                    wait_ns, wait_status = _us_to_ns(wait_us), "bound"
                core_key = core.upper().replace(" ", "_")
                pmu_applicable = not (
                    "AI_CPU" in core_key
                    or "AICPU" in core_key
                    or "COMMUNICATION" in core_key
                    or "HCCL" in core_key
                )
                pmu: dict[str, float] = {}
                pmu_resolution: dict[str, float] = {}
                if pmu_applicable:
                    for canonical, index in self.schema.pmu_columns.items():
                        value = _number(_cell(row, index))
                        if value is not None:
                            pmu[canonical] = float(value) * self.schema.pmu_scales[canonical]
                            if value.is_finite():
                                pmu_resolution[canonical] = (
                                    float(Decimal(1).scaleb(value.as_tuple().exponent))
                                    * self.schema.pmu_scales[canonical]
                                )
                if not self.schema.pmu_columns:
                    pmu_status = "not_available"
                elif not pmu_applicable:
                    pmu_status = "not_applicable"
                elif not pmu:
                    pmu_status = "absent"
                else:
                    pmu_status = "bound"
                event = Event(
                    rank_id=self.rank_id,
                    rank_index=self.rank_index,
                    name=name,
                    op_type=op_type,
                    task_type=core,
                    core=core,
                    stream_id=_cell(row, columns.get("stream_id")),
                    task_id=_cell(row, columns.get("task_id")),
                    device_id=_cell(row, columns.get("device_id")),
                    start_ns=start_ns,
                    end_ns=end_ns,
                    source_kind=self.schema.source_kind,
                    table="kernel_details.csv",
                    input_shapes=_cell(row, columns.get("input_shapes")),
                    input_dtypes=_cell(row, columns.get("input_dtypes")),
                    input_formats=_cell(row, columns.get("input_formats")),
                    wait_ns=wait_ns,
                    wait_status=wait_status,
                    wait_source=(self.schema.wait_column if self.schema.wait_column else None),
                    wait_provenance={
                        "table": "kernel_details.csv",
                        "column": self.schema.wait_column,
                        "row_index": row_index,
                        "status": wait_status,
                        "raw_wait_ns": wait_ns,
                    },
                    pmu=tuple(sorted(pmu.items())),
                    pmu_status=pmu_status,
                    pmu_source="kernel_details.csv",
                    pmu_kind="aicore" if pmu_applicable else "not_applicable",
                    pmu_provenance={
                        "table": "kernel_details.csv",
                        "row_index": row_index,
                        "status": pmu_status,
                        "raw_rows": len(pmu),
                        "metric_count": len(pmu),
                        "binding": "per_row",
                        "export_resolution": pmu_resolution,
                    },
                    locator={
                        "table": "kernel_details.csv",
                        "row_index": row_index,
                        "task_id": _cell(row, columns.get("task_id")),
                        "stream_id": _cell(row, columns.get("stream_id")),
                        "device_id": _cell(row, columns.get("device_id")),
                    },
                )
                for index, window in enumerate(windows):
                    if event.start_ns < window.end_ns and event.end_ns > window.start_ns:
                        results[index][0].append(event)
                        window_stats = results[index][1]
                        if wait_status == "bound":
                            window_stats.wait_bound_rows += 1
                        if pmu_status == "bound":
                            window_stats.pmu_bound_rows += 1
                            window_stats.pmu_rows += len(pmu)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 6)
        for index, (events, window_stats) in results.items():
            window_stats.rows_read = stats.rows_read
            window_stats.rows_selected = len(events)
            window_stats.compute_rows = len(events)
            window_stats.bad_rows = bad
            window_stats.elapsed_ms = elapsed_ms
            events.sort(key=lambda item: (item.start_ns, item.end_ns, item.task_id))
        return results
