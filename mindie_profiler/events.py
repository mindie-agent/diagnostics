"""Core records shared by the MindIE profiler pipeline.

[Role]
  Define the immutable event, explicit time-window, source-identity, and
  per-window observation records passed between source adapters, accounting,
  cross-rank statistics, and the C7 report writer.

[Invariants]
  * Internal time is integer nanoseconds.  CSV microseconds are converted
    once at the adapter boundary; reports convert to milliseconds once.
  * A missing measurement is ``None``, never ``0``.  Zero is only emitted
    when a real event/interval is zero length.
  * ``step`` is copied from the caller's explicit window.  No source adapter
    invents a step id from kernel order or timestamps.
  * ``wait_ns`` and ``pmu`` are measured full-capture values with explicit
    status/provenance; ambiguous task occurrences keep them empty.

[Timeouts & Locks]
  This module is pure data.  It performs no I/O and holds no locks.

[Side-Effect Policy]
  No filesystem, network, sqlite, or global-state mutation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

PARSER_VERSION = "mindie_profiler/0.0.3"
REPORT_SCHEMA = "mindie-profile-summary/1"


EVENT_IDENTITY_SCHEMA = "mindie-event-identity/1"


def composite_event_identity(
    *,
    rank_id: str,
    rank_index: int | None,
    source_kind: str,
    table: str,
    device_id: str,
    stream_id: str,
    task_id: str,
    start_ns: int,
    end_ns: int,
    locator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a stable identity from exact occurrence fields.

    Plain ``taskId``/``globalTaskId`` is insufficient: task ids repeat in real
    captures and global task ids repeat across full-capture occurrence groups.
    The original integer interval and the source/table/rank/locator occurrence
    are part of the hash; missing fields stay ``None`` rather than being guessed.
    """
    payload = {
        "schema_version": EVENT_IDENTITY_SCHEMA,
        "rank_id": str(rank_id),
        "rank_index": rank_index,
        "source_kind": str(source_kind),
        "table": str(table),
        "device_id": str(device_id),
        "stream_id": str(stream_id),
        "task_id": str(task_id),
        "start_ns": int(start_ns),
        "end_ns": int(end_ns),
        "locator": dict(locator or {}),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return {"schema_version": EVENT_IDENTITY_SCHEMA, "event_id": f"event:{digest}", "payload": payload}


class ProfilerError(RuntimeError):
    """Base class for explicit, user-actionable profiler failures."""


class ProfilerSchemaError(ProfilerError):
    """The source schema is unknown or lacks required verified columns."""


class ProfilerQueryError(ProfilerError):
    """A bounded read failed, timed out, or exceeded its row budget."""


class ProfilerUsageError(ProfilerError):
    """The request cannot be honoured without inventing step/rank context."""


@dataclass(frozen=True, slots=True)
class Window:
    """One explicitly supplied analysis window (all times are ns)."""

    start_ns: int
    end_ns: int
    step: int | None = None
    label: str | None = None
    source: str = "explicit"

    def __post_init__(self) -> None:
        if self.end_ns <= self.start_ns:
            raise ProfilerUsageError(
                f"window {self.label or self.step!r} must have end_ns > start_ns "
                f"(got {self.start_ns}..{self.end_ns})"
            )

    @property
    def span_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(slots=True)
class Event:
    """One device event reconstructed from the DB or kernel_details.csv."""

    rank_id: str
    rank_index: int | None
    name: str
    op_type: str
    task_type: str
    core: str
    stream_id: str
    task_id: str
    device_id: str
    start_ns: int
    end_ns: int
    source_kind: str
    table: str
    block: str = "Unclassified"
    mapping_rule: str = "residual_unmapped"
    bound_family: str = "unknown"
    is_comm: bool = False
    is_aicpu: bool = False
    input_shapes: str = ""
    input_dtypes: str = ""
    input_formats: str = ""
    wait_ns: int | None = None
    wait_status: str = "not_available"
    wait_source: str | None = None
    wait_provenance: dict[str, Any] = field(default_factory=dict)
    pmu: tuple[tuple[str, float], ...] = ()
    pmu_status: str = "not_available"
    pmu_source: str | None = None
    pmu_kind: str = "unknown"
    pmu_provenance: dict[str, Any] = field(default_factory=dict)
    task_occurrences_full_capture: int | None = None
    locator: dict[str, Any] = field(default_factory=dict)

    def identity(self) -> dict[str, Any]:
        return composite_event_identity(
            rank_id=self.rank_id,
            rank_index=self.rank_index,
            source_kind=self.source_kind,
            table=self.table,
            device_id=self.device_id,
            stream_id=self.stream_id,
            task_id=self.task_id,
            start_ns=self.start_ns,
            end_ns=self.end_ns,
            locator=self.locator,
        )

    @property
    def event_id(self) -> str:
        return str(self.identity()["event_id"])

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)

    def clipped(self, start_ns: int, end_ns: int) -> tuple[int, int] | None:
        left = max(self.start_ns, start_ns)
        right = min(self.end_ns, end_ns)
        if right <= left:
            return None
        return left, right

    def complete_in(self, start_ns: int, end_ns: int) -> bool:
        return self.start_ns >= start_ns and self.end_ns <= end_ns

    def pmu_map(self) -> dict[str, float]:
        return dict(self.pmu)


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Identity of one rank's chosen kernel event source."""

    rank_id: str
    rank_index: int | None
    kind: str
    path: str
    basename: str
    sha256: str | None
    size_bytes: int
    schema_version: str
    schema_variant: str
    header_sha256: str | None = None
    optional_missing: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class QueryStats:
    """Measured cost and coverage of one bounded source read."""

    rank_id: str
    window_label: str
    rows_read: int = 0
    rows_selected: int = 0
    compute_rows: int = 0
    comm_rows: int = 0
    comm_sched_rows: int = 0
    bad_rows: int = 0
    unresolved_string_ids: int = 0
    pmu_rows: int = 0
    pmu_bad_values: int = 0
    wait_bound_rows: int = 0
    wait_ambiguous_rows: int = 0
    pmu_bound_rows: int = 0
    pmu_ambiguous_rows: int = 0
    ambiguous_groups: int = 0
    measurement_scan_ms: float = 0.0
    elapsed_ms: float = 0.0
    budget: int | None = None

    def merge(self, other: "QueryStats") -> None:
        self.rows_read += other.rows_read
        self.rows_selected += other.rows_selected
        self.compute_rows += other.compute_rows
        self.comm_rows += other.comm_rows
        self.comm_sched_rows += other.comm_sched_rows
        self.bad_rows += other.bad_rows
        self.unresolved_string_ids += other.unresolved_string_ids
        self.pmu_rows += other.pmu_rows
        self.pmu_bad_values += other.pmu_bad_values
        self.wait_bound_rows += other.wait_bound_rows
        self.wait_ambiguous_rows += other.wait_ambiguous_rows
        self.pmu_bound_rows += other.pmu_bound_rows
        self.pmu_ambiguous_rows += other.pmu_ambiguous_rows
        self.ambiguous_groups += other.ambiguous_groups
        self.measurement_scan_ms = round(self.measurement_scan_ms + other.measurement_scan_ms, 6)
        self.elapsed_ms = round(self.elapsed_ms + other.elapsed_ms, 6)


@dataclass(slots=True)
class Observation:
    """Accounting for one rank inside one explicit window."""

    rank_id: str
    rank_index: int | None
    window: Window
    accounting: Any
    events: list[Event]
    source: SourceRef
    stats: QueryStats

    def identity(self) -> dict[str, Any]:
        payload = {
            "rank_id": self.rank_id,
            "rank_index": self.rank_index,
            "source_kind": self.source.kind,
            "source_sha256": self.source.sha256,
            "window_start_ns": self.window.start_ns,
            "window_end_ns": self.window.end_ns,
            "step": self.window.step,
            "label": self.window.label,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
        return {"observation_id": f"observation:{digest}", "payload": payload}
