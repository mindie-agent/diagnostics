"""Read-only, schema-checked Ascend profiler sqlite source.

[Role]
  Probe the verified torch_npu/CANN sqlite tables, resolve STRING_IDS once,
  and execute bounded read-only window queries that become ``Event`` rows.

[Invariants]
  * The connection is opened with ``mode=ro`` and ``PRAGMA query_only=ON``.
  * Required tables/columns come from ``knowledge/db_source_mapping.json``;
    an absent table/column or unsupported exporter version raises
    ``ProfilerSchemaError`` instead of guessing or emitting zero rows.
  * Column identifiers are validated, table names come from the mapping, and
    all time values are bound parameters.
  * Each query has a wall-clock deadline, a row budget, and a deterministic
    ``ORDER BY``.  Over-budget reads raise; they are never silently truncated.
  * Missing STRING_IDS entries are reported and named ``<unresolved:id>``;
    no unknown value becomes the string ``0``.
  * CTI waitNs and TASK_PMU_INFO are full-capture group measurements.  They
    bind to an event only when that globalTaskId has exactly one full-capture
    TASK occurrence and the metric rows are unambiguous; repeated ids remain
    group-scoped and are never summed, averaged, picked, or assigned.

[Timeouts & Locks]
  ``query_timeout_s`` (default 20s) is enforced by a sqlite progress handler;
  sqlite's own busy timeout is bounded to the same value.  No locks held after
  a query returns; connections close in ``finally``.

[Side-Effect Policy]
  Read-only filesystem access to the supplied DB; no writes, no network, and
  no data copied off the source machine.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from .events import Event, ProfilerQueryError, ProfilerSchemaError, QueryStats, Window

MAPPING_PATH = Path(__file__).resolve().parent / "knowledge" / "db_source_mapping.json"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PMU_BUDGET_MIN = 100_000
_PMU_BUDGET_MAX = 2_000_000
_PMU_BUDGET_FACTOR = 8


@functools.lru_cache(maxsize=1)
def load_mapping(path: str | Path | None = None) -> dict[str, Any]:
    mapping_path = Path(path) if path is not None else MAPPING_PATH
    try:
        doc = json.loads(mapping_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProfilerSchemaError(f"db source mapping missing: {mapping_path}") from exc
    except json.JSONDecodeError as exc:
        raise ProfilerSchemaError(f"{mapping_path.name}: invalid JSON: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("version") != 1:
        raise ProfilerSchemaError(f"{mapping_path.name}: version must be 1")
    for table, columns in doc.get("db_columns", {}).items():
        if not table.endswith("_required"):
            continue
        if not isinstance(columns, list) or not all(_IDENTIFIER.match(str(c)) for c in columns):
            raise ProfilerSchemaError(f"{mapping_path.name}: bad column list {table}")
    pmu_metrics = doc.get("pmu_metrics")
    if not isinstance(pmu_metrics, dict) or not pmu_metrics:
        raise ProfilerSchemaError(f"{mapping_path.name}: pmu_metrics must be a non-empty mapping")
    for name, spec in pmu_metrics.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("db_metric"), str) or not spec["db_metric"]:
            raise ProfilerSchemaError(f"{mapping_path.name}: pmu_metrics.{name} needs db_metric")
        if not isinstance(spec.get("unit"), str) or not spec["unit"]:
            raise ProfilerSchemaError(f"{mapping_path.name}: pmu_metrics.{name} needs unit")
        if "csv_column" in spec and not isinstance(spec["csv_column"], str):
            raise ProfilerSchemaError(f"{mapping_path.name}: pmu_metrics.{name}.csv_column must be a string")
        if "csv_scale" in spec and not isinstance(spec["csv_scale"], (int, float)):
            raise ProfilerSchemaError(f"{mapping_path.name}: pmu_metrics.{name}.csv_scale must be numeric")
    return doc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DbSchema:
    path: str
    schema_version: str
    schema_variant: str
    tables: tuple[str, ...]
    optional_missing: tuple[str, ...]
    block_column: str | None
    mix_block_column: str | None
    shape_columns: tuple[str, ...]
    wait_column: str | None
    pmu_table_present: bool
    pmu_metric_count: int
    has_wait_ns: bool
    size_bytes: int

    @property
    def exporter_schema_version(self) -> str:
        if self.schema_version == "unversioned":
            return "torch_npu_db_unversioned"
        return f"torch_npu_db_v{self.schema_version}"

    @property
    def source_kind(self) -> str:
        return "ascend_pytorch_profiler_db"


def _connect(path: Path, timeout_s: float) -> sqlite3.Connection:
    if not path.is_file():
        raise ProfilerQueryError(f"profiler db not found: {path}")
    uri = "file:" + quote(str(path.resolve()), safe="/") + "?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True, timeout=max(0.1, timeout_s))
    except sqlite3.Error as exc:
        raise ProfilerQueryError(f"cannot open profiler db read-only: {path}: {exc}") from exc
    con.execute("PRAGMA query_only = ON")
    con.execute(f"PRAGMA busy_timeout = {int(max(1, timeout_s) * 1000)}")
    return con


def _table_names(con: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(con: sqlite3.Connection, table: str) -> set[str]:
    if not _IDENTIFIER.match(table):
        raise ProfilerSchemaError(f"unsafe table identifier: {table!r}")
    return {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})")}


def _first_present(columns: set[str], aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in columns:
            return alias
    return None


def _missing_columns(columns: set[str], required: list[str]) -> list[str]:
    return [column for column in required if column not in columns]


def probe_db(path: str | Path, *, timeout_s: float = 20.0) -> DbSchema:
    """Fail closed unless every required verified column is present."""

    mapping = load_mapping()
    db_path = Path(path)
    con = _connect(db_path, timeout_s)
    try:
        tables = _table_names(con)
        names = mapping["tables"]
        missing: dict[str, list[str]] = {}
        required_by_table = {
            names["task"]: mapping["db_columns"]["task_required"],
            names["compute"]: mapping["db_columns"]["compute_required"],
            names["strings"]: ["id", "value"],
        }
        for table, required in required_by_table.items():
            if table not in tables:
                missing[table] = list(required)
                continue
            absent = _missing_columns(_columns(con, table), required)
            if absent:
                missing[table] = absent
        optional_missing: list[str] = []
        pmu_table_present = False
        for key in ("comm", "comm_sched", "pmu"):
            table = names[key]
            required = mapping["db_columns"][f"{key}_required"]
            if table not in tables:
                optional_missing.append(table)
                continue
            absent = _missing_columns(_columns(con, table), required)
            if absent:
                missing[table] = absent
            elif key == "pmu":
                pmu_table_present = True
        if names["meta"] in tables:
            absent = _missing_columns(_columns(con, names["meta"]), ["name", "value"])
            if absent:
                missing[names["meta"]] = absent
        if missing:
            detail = ", ".join(f"{table}({','.join(cols)})" for table, cols in sorted(missing.items()))
            raise ProfilerSchemaError(f"{db_path.name}: unknown/incomplete profiler schema: missing {detail}")

        version = "unversioned"
        if names["meta"] in tables:
            meta_columns = _columns(con, names["meta"])
            if {"name", "value"} <= meta_columns:
                row = con.execute(
                    f"SELECT value FROM {names['meta']} WHERE name = ? ORDER BY rowid DESC LIMIT 1",
                    ("SCHEMA_VERSION",),
                ).fetchone()
                if row and row[0] not in (None, ""):
                    version = str(row[0]).strip()
        supported = set(mapping["exporter_schema"]["supported_versions"])
        prefixes = tuple(mapping["exporter_schema"]["supported_prefixes"])
        if version != "unversioned" and version not in supported and not version.startswith(prefixes):
            raise ProfilerSchemaError(
                f"{db_path.name}: unsupported exporter SCHEMA_VERSION {version!r}; "
                f"supported={sorted(supported)} prefixes={list(prefixes)}"
            )
        if version == "unversioned" and not mapping["exporter_schema"]["unversioned_columns_verified"]:
            raise ProfilerSchemaError(f"{db_path.name}: META_DATA.SCHEMA_VERSION missing")

        compute_columns = _columns(con, names["compute"])
        aliases = mapping["db_columns"]["compute_alias_groups"]
        block_col = _first_present(compute_columns, aliases["block_dim"])
        mix_col = _first_present(compute_columns, aliases["mix_block_dim"])
        shape_columns = tuple(
            column
            for column in mapping["db_columns"].get("compute_optional", [])
            if column in compute_columns
        )
        wait_column = _first_present(
            compute_columns, list(mapping["db_columns"].get("compute_wait_columns", []))
        )
        wait_ns = wait_column is not None
        variant_bits = [f"v{version}"]
        if "blockDim" in compute_columns:
            variant_bits.append("blockDim")
        elif "blockNum" in compute_columns:
            variant_bits.append("blockNum")
        else:
            variant_bits.append("noBlock")
        if wait_ns:
            variant_bits.append("waitNs")
        if pmu_table_present:
            variant_bits.append("pmu")
        return DbSchema(
            path=str(db_path.resolve()),
            schema_version=version,
            schema_variant="_".join(variant_bits),
            tables=tuple(sorted(tables)),
            optional_missing=tuple(sorted(optional_missing)),
            block_column=block_col,
            mix_block_column=mix_col,
            shape_columns=shape_columns,
            wait_column=wait_column,
            pmu_table_present=pmu_table_present,
            pmu_metric_count=len(mapping["pmu_metrics"]),
            has_wait_ns=wait_ns,
            size_bytes=db_path.stat().st_size,
        )
    except sqlite3.Error as exc:
        raise ProfilerSchemaError(f"{db_path.name}: schema probe failed: {exc}") from exc
    finally:
        con.close()


def _resolve_strings(con: sqlite3.Connection, table: str) -> dict[int, str]:
    try:
        return {int(row[0]): str(row[1]) if row[1] is not None else "" for row in con.execute(f"SELECT id, value FROM {table}")}
    except sqlite3.Error as exc:
        raise ProfilerSchemaError(f"cannot read {table}: {exc}") from exc


def _text(strings: Mapping[int, str], value: Any, unresolved: list[int]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            key = int(value)
        except ValueError:
            return value
    else:
        try:
            key = int(value)
        except (TypeError, ValueError):
            return ""
    resolved = strings.get(key)
    if resolved is None:
        unresolved.append(key)
        return f"<unresolved:{key}>"
    return resolved


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


class DbSource:
    """A probed read-only DB.  One instance is reused across windows."""

    def __init__(self, schema: DbSchema, *, mapping: Mapping[str, Any] | None = None) -> None:
        self.schema = schema
        self.mapping = dict(mapping or load_mapping())
        self._names = self.mapping["tables"]
        self._strings: dict[int, str] | None = None
        self._task_occurrences: dict[int, int] | None = None
        self._occurrence_scan_ms = 0.0

    def _connection(self, timeout_s: float) -> sqlite3.Connection:
        return _connect(Path(self.schema.path), timeout_s)

    def _load_strings(self, con: sqlite3.Connection) -> dict[int, str]:
        if self._strings is None:
            self._strings = _resolve_strings(con, self._names["strings"])
        return self._strings

    def _task_occurrence_counts(self, con: sqlite3.Connection) -> dict[int, int]:
        """Full-capture TASK occurrence count per globalTaskId.

        This is deliberately a full-capture query, not a window query: a
        globalTaskId repeated outside the selected window still makes the
        CTI waitNs/PMU group ambiguous for the selected occurrence.
        """

        if self._task_occurrences is None:
            started = time.perf_counter()
            self._task_occurrences = {
                int(row[0]): int(row[1])
                for row in con.execute(
                    f"SELECT globalTaskId, COUNT(*) FROM {self._names['task']} GROUP BY globalTaskId"
                )
            }
            self._occurrence_scan_ms = round((time.perf_counter() - started) * 1000.0, 6)
        return self._task_occurrences

    def capture_range_ns(self, *, timeout_s: float = 20.0) -> tuple[int, int] | None:
        con = self._connection(timeout_s)
        try:
            lows: list[int] = []
            highs: list[int] = []
            for table in (self._names["task"], self._names["comm"]):
                if table not in self.schema.tables:
                    continue
                row = con.execute(f"SELECT MIN(startNs), MAX(endNs) FROM {table}").fetchone()
                if row and row[0] is not None and row[1] is not None:
                    lows.append(int(row[0]))
                    highs.append(int(row[1]))
            if not lows or not highs:
                return None
            return min(lows), max(highs)
        except sqlite3.Error as exc:
            raise ProfilerQueryError(f"cannot read capture range from {self.schema.path}: {exc}") from exc
        finally:
            con.close()

    def _pmu_budget(self, max_rows: int) -> int:
        return min(_PMU_BUDGET_MAX, max(_PMU_BUDGET_MIN, int(max_rows) * _PMU_BUDGET_FACTOR))

    def _load_pmu_rows(
        self,
        con: sqlite3.Connection,
        global_task_ids: list[int],
        strings: Mapping[int, str],
        unresolved: list[int],
        *,
        budget: int,
    ) -> tuple[dict[int, list[tuple[str, float | None]]], int, int]:
        table = self._names["pmu"]
        rows_by_id: dict[int, list[tuple[str, float | None]]] = {}
        read_rows = 0
        bad_values = 0
        if not global_task_ids or table not in self.schema.tables:
            return rows_by_id, read_rows, bad_values
        for offset in range(0, len(global_task_ids), 500):
            chunk = global_task_ids[offset : offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            sql = (
                f"SELECT globalTaskId, name, value FROM {table} "
                f"WHERE globalTaskId IN ({placeholders})"
            )
            for global_id, name_id, value in con.execute(sql, chunk):
                read_rows += 1
                if read_rows > budget:
                    raise ProfilerQueryError(
                        f"{Path(self.schema.path).name}: PMU row budget {budget} exceeded for "
                        f"{len(global_task_ids)} selected globalTaskIds"
                    )
                metric = _text(strings, name_id, unresolved)
                numeric: float | None = None
                if value is not None:
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        bad_values += 1
                rows_by_id.setdefault(int(global_id), []).append((metric, numeric))
        return rows_by_id, read_rows, bad_values

    @staticmethod
    def _wait_binding(
        raw_wait: Any,
        occurrences: int | None,
        wait_column: str | None,
    ) -> tuple[int | None, str]:
        if wait_column is None:
            return None, "not_available"
        if raw_wait is None:
            return None, "absent"
        try:
            wait_ns = int(raw_wait)
        except (TypeError, ValueError):
            return None, "invalid_value"
        if wait_ns < 0:
            return None, "invalid_value"
        if occurrences is None:
            return None, "cardinality_unknown"
        if occurrences > 1:
            return None, "ambiguous_task_occurrence"
        return wait_ns, "bound"

    @staticmethod
    def _pmu_group_summary(
        global_id: int,
        rows: list[tuple[str, float | None]],
        occurrences: int,
        status: str,
    ) -> dict[str, Any]:
        per_metric: dict[str, dict[str, Any]] = {}
        for metric, value in rows:
            entry = per_metric.setdefault(
                metric,
                {"rows": 0, "values": 0, "distinct_values": 0, "_values": []},
            )
            entry["rows"] += 1
            if value is not None:
                entry["values"] += 1
                entry["_values"].append(float(value))
        for entry in per_metric.values():
            values = entry.pop("_values")
            entry["distinct_values"] = len(set(values))
            entry["all_zero"] = bool(values) and all(value == 0.0 for value in values)
            entry["min"] = min(values) if values else None
            entry["max"] = max(values) if values else None
        return {
            "table": "TASK_PMU_INFO",
            "global_task_id": global_id,
            "task_occurrences_full_capture": occurrences,
            "status": status,
            "raw_rows": len(rows),
            "metric_count": len(per_metric),
            "per_metric": per_metric,
            "binding": "group_scoped_only",
        }

    def _bind_pmu(
        self,
        event: Event,
        rows: list[tuple[str, float | None]],
        occurrences: int | None,
        group_cache: dict[tuple[int, str], dict[str, Any]],
    ) -> None:
        global_id = event.locator.get("task_global_id")
        if not self.schema.pmu_table_present:
            event.pmu_status = "not_available"
            return
        if not isinstance(global_id, int):
            event.pmu_status = "cardinality_unknown"
            return
        if not rows:
            event.pmu_status = "absent"
            event.pmu_provenance = {
                "table": "TASK_PMU_INFO",
                "global_task_id": global_id,
                "task_occurrences_full_capture": occurrences,
                "status": "absent",
                "raw_rows": 0,
            }
            return
        per_metric: dict[str, int] = {}
        for metric, _value in rows:
            per_metric[metric] = per_metric.get(metric, 0) + 1
        if occurrences is None:
            status = "cardinality_unknown"
        elif occurrences > 1:
            status = "ambiguous_task_occurrence"
        elif any(count != 1 for count in per_metric.values()):
            status = "duplicate_metric_rows"
        elif any(value is None for _metric, value in rows):
            status = "incomplete_metric_value"
        else:
            status = "bound"
        event.pmu_status = status
        if status == "bound":
            meta = self.mapping["pmu_metrics"]
            canonical: dict[str, float] = {}
            for metric, value in rows:
                spec = meta.get(metric, {})
                key = str(spec.get("db_metric", metric))
                factor = float(spec.get("db_scale", 1.0))
                canonical[key] = float(value) * factor
            event.pmu = tuple(sorted(canonical.items()))
            event.pmu_provenance = {
                "table": "TASK_PMU_INFO",
                "global_task_id": global_id,
                "task_occurrences_full_capture": occurrences,
                "status": "bound",
                "raw_rows": len(rows),
                "metric_count": len(per_metric),
                "binding": "unique_task_occurrence",
            }
            return
        cache_key = (global_id, status)
        summary = group_cache.get(cache_key)
        if summary is None:
            summary = self._pmu_group_summary(global_id, rows, occurrences or 0, status)
            group_cache[cache_key] = summary
        event.pmu_provenance = summary

    def iter_window(
        self,
        window: Window,
        *,
        max_rows: int = 1_000_000,
        timeout_s: float = 20.0,
    ) -> tuple[list[Event], QueryStats]:
        started = time.perf_counter()
        con = self._connection(timeout_s)
        deadline = started + max(0.1, timeout_s)
        con.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 20_000)
        stats = QueryStats(rank_id="", window_label=window.label or str(window.step), budget=max_rows)
        events: list[Event] = []
        unresolved: list[int] = []
        try:
            strings = self._load_strings(con)
            occurrences = self._task_occurrence_counts(con)
            stats.measurement_scan_ms += self._occurrence_scan_ms
            names = self._names
            compute = names["compute"]
            block = f"c.{self.schema.block_column}" if self.schema.block_column else "NULL"
            mix = f"c.{self.schema.mix_block_column}" if self.schema.mix_block_column else "NULL"
            shape_exprs = []
            for column in ("inputShapes", "inputDataTypes", "inputFormats"):
                expression = f"c.{column}" if column in self.schema.shape_columns else "NULL"
                shape_exprs.append(f"{expression} AS {column}")
            wait_expr = f"c.{self.schema.wait_column}" if self.schema.wait_column else "NULL"
            sql = (
                "SELECT t.globalTaskId, t.startNs, t.endNs, t.deviceId, t.streamId, t.taskId, "
                "t.modelId, t.contextId, c.name, c.opType, c.taskType, "
                + ", ".join(shape_exprs)
                + ", " + wait_expr + " AS wait_ns, " + block + " AS block_dim, " + mix + " AS mix_block_dim "
                f"FROM {names['task']} t JOIN {compute} c ON c.globalTaskId = t.globalTaskId "
                "WHERE t.endNs > ? AND t.startNs < ? ORDER BY t.startNs, t.globalTaskId LIMIT ?"
            )
            rows = con.execute(sql, (window.start_ns, window.end_ns, max_rows + 1)).fetchall()
            stats.rows_read += len(rows)
            if len(rows) > max_rows:
                raise ProfilerQueryError(
                    f"{Path(self.schema.path).name}: window {window.label or window.step} "
                    f"exceeded row budget {max_rows}; narrow the window or raise --max-rows"
                )
            compute_events: list[Event] = []
            pmu_ids: set[int] = set()
            for row in rows:
                if row[1] is None or row[2] is None or _as_int(row[2]) <= _as_int(row[1]):
                    stats.bad_rows += 1
                    continue
                global_id = _as_optional_int(row[0])
                occurrence_count = occurrences.get(global_id) if global_id is not None else None
                wait_ns, wait_status = self._wait_binding(row[14], occurrence_count, self.schema.wait_column)
                if wait_status == "bound":
                    stats.wait_bound_rows += 1
                elif wait_status == "ambiguous_task_occurrence":
                    stats.wait_ambiguous_rows += 1
                name = _text(strings, row[8], unresolved)
                op_type = _text(strings, row[9], unresolved)
                core = _text(strings, row[10], unresolved)
                event = Event(
                    rank_id="",
                    rank_index=None,
                    name=name,
                    op_type=op_type,
                    task_type=core,
                    core=core,
                    stream_id=_as_text(row[4]),
                    task_id=_as_text(row[5]),
                    device_id=_as_text(row[3]),
                    start_ns=_as_int(row[1]),
                    end_ns=_as_int(row[2]),
                    source_kind=self.schema.source_kind,
                    table=compute,
                    input_shapes=_text(strings, row[11], unresolved),
                    input_dtypes=_text(strings, row[12], unresolved),
                    input_formats=_text(strings, row[13], unresolved),
                    wait_ns=wait_ns,
                    wait_status=wait_status,
                    wait_source=(f"COMPUTE_TASK_INFO.{self.schema.wait_column}" if self.schema.wait_column else None),
                    wait_provenance={
                        "table": compute,
                        "column": self.schema.wait_column,
                        "global_task_id": global_id,
                        "task_occurrences_full_capture": occurrence_count,
                        "raw_wait_ns": None if row[14] is None else _as_int(row[14]),
                        "status": wait_status,
                    },
                    pmu_kind="aicore",
                    pmu_source="TASK_PMU_INFO",
                    task_occurrences_full_capture=occurrence_count,
                    locator={
                        "table": compute,
                        "task_global_id": global_id,
                        "task_id": _as_text(row[5]),
                        "stream_id": _as_text(row[4]),
                        "device_id": _as_text(row[3]),
                        "task_occurrences_full_capture": occurrence_count,
                    },
                )
                compute_events.append(event)
                events.append(event)
                if global_id is not None:
                    pmu_ids.add(global_id)
            stats.compute_rows = len(compute_events)
            stats.ambiguous_groups = len(
                {
                    event.locator.get("task_global_id")
                    for event in compute_events
                    if (event.task_occurrences_full_capture or 0) > 1
                }
            )
            if self.schema.pmu_table_present and pmu_ids:
                pmu_started = time.perf_counter()
                pmu_rows, pmu_read, bad_pmu = self._load_pmu_rows(
                    con,
                    sorted(pmu_ids),
                    strings,
                    unresolved,
                    budget=self._pmu_budget(max_rows),
                )
                stats.measurement_scan_ms += round((time.perf_counter() - pmu_started) * 1000.0, 6)
                stats.pmu_rows = pmu_read
                stats.pmu_bad_values = bad_pmu
                group_cache: dict[tuple[int, str], dict[str, Any]] = {}
                for event in compute_events:
                    global_id = event.locator.get("task_global_id")
                    rows_for_id = pmu_rows.get(global_id, []) if isinstance(global_id, int) else []
                    self._bind_pmu(
                        event,
                        rows_for_id,
                        event.task_occurrences_full_capture,
                        group_cache,
                    )
                    if event.pmu_status == "bound":
                        stats.pmu_bound_rows += 1
                    elif event.pmu_status in {
                        "ambiguous_task_occurrence",
                        "duplicate_metric_rows",
                        "incomplete_metric_value",
                        "cardinality_unknown",
                    }:
                        stats.pmu_ambiguous_rows += 1

            comm = names["comm"]
            if comm in self.schema.tables:
                comm_sql = (
                    "SELECT rowid, opName, opType, startNs, endNs, deviceId "
                    f"FROM {comm} WHERE endNs > ? AND startNs < ? ORDER BY startNs, rowid LIMIT ?"
                )
                comm_rows = con.execute(comm_sql, (window.start_ns, window.end_ns, max_rows + 1)).fetchall()
                stats.rows_read += len(comm_rows)
                if len(comm_rows) > max_rows:
                    raise ProfilerQueryError(
                        f"{Path(self.schema.path).name}: comm window {window.label or window.step} "
                        f"exceeded row budget {max_rows}"
                    )
                for row in comm_rows:
                    if row[3] is None or row[4] is None or _as_int(row[4]) <= _as_int(row[3]):
                        stats.bad_rows += 1
                        continue
                    events.append(
                        Event(
                            rank_id="",
                            rank_index=None,
                            name=_text(strings, row[1], unresolved),
                            op_type=_text(strings, row[2], unresolved),
                            task_type="COMMUNICATION",
                            core="COMMUNICATION",
                            stream_id="",
                            task_id="",
                            device_id=_as_text(row[5]),
                            start_ns=_as_int(row[3]),
                            end_ns=_as_int(row[4]),
                            source_kind=self.schema.source_kind,
                            table=comm,
                            wait_status="not_applicable",
                            pmu_status="not_applicable",
                            pmu_kind="not_applicable",
                            locator={
                                "table": comm,
                                "rowid": _as_optional_int(row[0]),
                                "device_id": _as_text(row[5]),
                            },
                        )
                    )
                stats.comm_rows = len(comm_rows)

            sched = names["comm_sched"]
            if sched in self.schema.tables:
                sched_sql = (
                    "SELECT s.globalTaskId, s.name, s.opType, s.taskType, t.startNs, t.endNs, "
                    "t.deviceId, t.streamId, t.taskId "
                    f"FROM {sched} s JOIN {names['task']} t ON t.globalTaskId = s.globalTaskId "
                    "WHERE t.endNs > ? AND t.startNs < ? ORDER BY t.startNs, s.globalTaskId LIMIT ?"
                )
                sched_rows = con.execute(sched_sql, (window.start_ns, window.end_ns, max_rows + 1)).fetchall()
                stats.rows_read += len(sched_rows)
                if len(sched_rows) > max_rows:
                    raise ProfilerQueryError(
                        f"{Path(self.schema.path).name}: comm-schedule window {window.label or window.step} "
                        f"exceeded row budget {max_rows}"
                    )
                for row in sched_rows:
                    if row[4] is None or row[5] is None or _as_int(row[5]) <= _as_int(row[4]):
                        stats.bad_rows += 1
                        continue
                    global_id = _as_optional_int(row[0])
                    events.append(
                        Event(
                            rank_id="",
                            rank_index=None,
                            name=_text(strings, row[1], unresolved),
                            op_type=_text(strings, row[2], unresolved),
                            task_type=_text(strings, row[3], unresolved),
                            core=_text(strings, row[3], unresolved),
                            stream_id=_as_text(row[7]),
                            task_id=_as_text(row[8]),
                            device_id=_as_text(row[6]),
                            start_ns=_as_int(row[4]),
                            end_ns=_as_int(row[5]),
                            source_kind=self.schema.source_kind,
                            table=sched,
                            wait_status="not_applicable",
                            pmu_status="not_applicable",
                            pmu_kind="not_applicable",
                            task_occurrences_full_capture=(occurrences.get(global_id) if global_id is not None else None),
                            locator={
                                "table": sched,
                                "task_global_id": global_id,
                                "stream_id": _as_text(row[7]),
                            },
                        )
                    )
                stats.comm_sched_rows = len(sched_rows)
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise ProfilerQueryError(
                    f"{Path(self.schema.path).name}: query exceeded {timeout_s:.1f}s for "
                    f"window {window.label or window.step}"
                ) from exc
            raise ProfilerQueryError(f"{Path(self.schema.path).name}: query failed: {exc}") from exc
        except sqlite3.Error as exc:
            raise ProfilerQueryError(f"{Path(self.schema.path).name}: query failed: {exc}") from exc
        finally:
            con.set_progress_handler(None, 0)
            con.close()

        if len(events) > max_rows:
            raise ProfilerQueryError(
                f"{Path(self.schema.path).name}: window {window.label or window.step} combined "
                f"event budget {max_rows} exceeded"
            )
        events.sort(key=lambda item: (item.start_ns, item.end_ns, item.table, item.task_id))
        stats.rows_selected = len(events)
        stats.unresolved_string_ids = len(set(unresolved))
        stats.elapsed_ms = round((time.perf_counter() - started) * 1000.0, 6)
        return events, stats
