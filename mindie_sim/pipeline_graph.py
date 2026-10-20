"""Layer 2 graph parsing, phase expansion and duration calibration.

[Role]
  Own the explicit microstage graph schema, two/multi-phase expansion with
  super-period edge distances, per-pipe FIFO closure edges, and strict duration
  calibration (including rejection of raw PipeUtilization percentages).

[Invariants]
  * Expansion maps tile distance ``d`` at phase ``p`` to
    ``dst_phase=(p+d)%phases`` and super-period distance ``(p+d)//phases``.
  * Stage names, edges and FIFO entries are caller input; no operator issue
    order is embedded in this module.
  * A missing/null/invalid/percentage duration yields an uncalibrated result;
    it is never replaced by zero or an estimated number.

[Timeouts & Locks]
  Pure mapping/integer computation.  No I/O, locks, sleeps or subprocesses.

[Side-Effect Policy]
  Reads caller mappings only; no filesystem, network or device mutation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


class PipelineError(ValueError):
    """Stable pipeline-input validation failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _int_value(value: Any, reason: str) -> int:
    if isinstance(value, bool):
        raise PipelineError(reason)
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise PipelineError(reason) from exc
    if numeric != value and not isinstance(value, str):
        raise PipelineError(reason)
    return numeric


class PipelineError(ValueError):
    """Stable pipeline-input validation failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _int_value(value: Any, reason: str) -> int:
    if isinstance(value, bool):
        raise PipelineError(reason)
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise PipelineError(reason) from exc
    if numeric != value and not isinstance(value, str):
        raise PipelineError(reason)
    return numeric


@dataclass(frozen=True)
class PipelineStage:
    name: str
    pipe: str | None = None
    transaction_index: int | None = None


@dataclass(frozen=True)
class PipelineEdge:
    src: str
    dst: str
    distance: int
    kind: str = "semaphore"
    same_slot: str | None = None


@dataclass(frozen=True)
class PipelineGraph:
    name: str
    phases: int
    tiles_per_period: int
    stages: tuple[PipelineStage, ...]
    edges: tuple[PipelineEdge, ...]
    pipe_fifo: Mapping[str, tuple[tuple[str, int], ...]] = field(default_factory=dict)
    relaxation_checks: tuple[Mapping[str, Any], ...] = ()
    transactions_per_loop: int | None = None
    variant: str | None = None
    source: str = "<mapping>"


@dataclass(frozen=True)
class ExpandedNode:
    node_id: str
    stage: str
    phase: int
    pipe: str | None = None


@dataclass(frozen=True)
class ExpandedEdge:
    src: str
    dst: str
    distance: int
    kind: str
    base_src: str
    base_dst: str
    phase: int


@dataclass(frozen=True)
class ExpandedGraph:
    nodes: tuple[ExpandedNode, ...]
    edges: tuple[ExpandedEdge, ...]
    index: Mapping[str, int]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


@dataclass(frozen=True)
class DurationCalibration:
    status: str
    durations_ns: Mapping[str, float]
    uncalibrated_stages: tuple[str, ...]
    reasons: tuple[str, ...]
    provenance: str | None = None
    extra_keys: tuple[str, ...] = ()
    node_durations_ns: Mapping[str, float] = field(default_factory=dict)

    @property
    def calibrated(self) -> bool:
        return self.status == "calibrated"


def _stage_from_value(value: Any) -> PipelineStage:
    if isinstance(value, str):
        name = value.strip()
        if not name:
            raise PipelineError("stage_name_empty")
        return PipelineStage(name=name)
    if not isinstance(value, Mapping):
        raise PipelineError("stage_invalid")
    name = str(value.get("name", "")).strip()
    if not name:
        raise PipelineError("stage_name_empty")
    pipe = value.get("pipe")
    tx = value.get("transaction_index", value.get("slot_index"))
    if tx is not None:
        tx = _int_value(tx, f"stage_transaction_index_invalid:{name}")
        if tx < 0:
            raise PipelineError(f"stage_transaction_index_invalid:{name}")
    return PipelineStage(name=name, pipe=str(pipe).strip() if pipe else None, transaction_index=tx)


def _parse_pipe_fifo(raw: Any, stage_names: set[str]) -> dict[str, tuple[tuple[str, int], ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise PipelineError("pipe_fifo_invalid")
    result: dict[str, tuple[tuple[str, int], ...]] = {}
    for pipe, value in raw.items():
        entries_raw = value.get("entries", value.get("commands")) if isinstance(value, Mapping) else value
        if not isinstance(entries_raw, Sequence) or isinstance(entries_raw, (str, bytes)):
            raise PipelineError(f"pipe_fifo_invalid:{pipe}")
        entries: list[tuple[str, int]] = []
        for entry in entries_raw:
            if isinstance(entry, Mapping):
                stage = str(entry.get("stage", "")).strip()
                offset = entry.get("tile_offset", entry.get("offset", 0))
            elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) == 2:
                stage, offset = str(entry[0]).strip(), entry[1]
            else:
                raise PipelineError(f"pipe_fifo_entry_invalid:{pipe}")
            if stage not in stage_names:
                raise PipelineError(f"pipe_fifo_stage_unknown:{stage}")
            entries.append((stage, _int_value(offset, f"pipe_fifo_offset_invalid:{pipe}")))
        if not entries:
            raise PipelineError(f"pipe_fifo_empty:{pipe}")
        result[str(pipe)] = tuple(entries)
    return result


def parse_pipeline_graph(spec: Mapping[str, Any], *, source: str = "<mapping>") -> PipelineGraph:
    if not isinstance(spec, Mapping):
        raise PipelineError("graph_invalid")
    name = str(spec.get("name", "pipeline"))
    phases = _int_value(spec.get("phases", 2), "phases_invalid")
    tiles = _int_value(spec.get("tiles_per_period", phases), "tiles_per_period_invalid")
    if phases <= 0 or phases > 64:
        raise PipelineError("phases_invalid")
    if tiles <= 0 or tiles > phases * 64:
        raise PipelineError("tiles_per_period_invalid")
    raw_stages = spec.get("stages")
    if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, (str, bytes)) or not raw_stages:
        raise PipelineError("stages_missing")
    stages = tuple(_stage_from_value(value) for value in raw_stages)
    names = [stage.name for stage in stages]
    if len(set(names)) != len(names):
        raise PipelineError("stage_names_not_unique")
    stage_names = set(names)
    raw_edges = spec.get("edges", ())
    if raw_edges is None:
        raw_edges = ()
    if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
        raise PipelineError("edges_invalid")
    edges: list[PipelineEdge] = []
    for raw in raw_edges:
        if not isinstance(raw, Mapping):
            raise PipelineError("edge_invalid")
        src = str(raw.get("src", "")).strip()
        dst = str(raw.get("dst", "")).strip()
        if src not in stage_names or dst not in stage_names:
            raise PipelineError(f"edge_stage_unknown:{src}->{dst}")
        distance = _int_value(raw.get("distance", raw.get("tile_distance", 0)), "edge_distance_invalid")
        if abs(distance) > 100_000:
            raise PipelineError("edge_distance_out_of_range")
        edges.append(PipelineEdge(
            src=src,
            dst=dst,
            distance=distance,
            kind=str(raw.get("kind", "semaphore")),
            same_slot=str(raw["same_slot"]) if raw.get("same_slot") else None,
        ))
    fifo = _parse_pipe_fifo(spec.get("pipe_fifo"), stage_names)
    loop_length = _int_value(spec.get("loop_length", 1), "loop_length_invalid")
    if loop_length <= 0:
        raise PipelineError("loop_length_invalid")
    for pipe, entries in fifo.items():
        for (src, src_offset), (dst, dst_offset) in zip(entries, entries[1:]):
            edges.append(PipelineEdge(src, dst, dst_offset - src_offset, kind="fifo"))
        first_stage, first_offset = entries[0]
        last_stage, last_offset = entries[-1]
        edges.append(PipelineEdge(last_stage, first_stage, loop_length + first_offset - last_offset, kind="fifo"))
    seen: set[tuple[str, str, int, str]] = set()
    unique_edges: list[PipelineEdge] = []
    for edge in edges:
        key = (edge.src, edge.dst, edge.distance, edge.kind)
        if key not in seen:
            seen.add(key)
            unique_edges.append(edge)
    if not unique_edges:
        raise PipelineError("edges_missing")
    relaxations = spec.get("relaxation_checks", ()) or ()
    if not isinstance(relaxations, Sequence) or isinstance(relaxations, (str, bytes)):
        raise PipelineError("relaxation_checks_invalid")
    tx_loop = spec.get("transactions_per_loop")
    if tx_loop is not None:
        tx_loop = _int_value(tx_loop, "transactions_per_loop_invalid")
        if tx_loop <= 0:
            raise PipelineError("transactions_per_loop_invalid")
    return PipelineGraph(
        name=name,
        phases=phases,
        tiles_per_period=tiles,
        stages=stages,
        edges=tuple(unique_edges),
        pipe_fifo=fifo,
        relaxation_checks=tuple(dict(item) for item in relaxations if isinstance(item, Mapping)),
        transactions_per_loop=tx_loop,
        variant=str(spec["variant"]) if spec.get("variant") else None,
        source=source,
    )


def expand_pipeline_graph(graph: PipelineGraph) -> ExpandedGraph:
    nodes: list[ExpandedNode] = []
    index: dict[str, int] = {}
    for phase in range(graph.phases):
        for stage in graph.stages:
            node_id = f"p{phase}.{stage.name}"
            index[node_id] = len(nodes)
            nodes.append(ExpandedNode(node_id=node_id, stage=stage.name, phase=phase, pipe=stage.pipe))
    expanded_edges: list[ExpandedEdge] = []
    for edge in graph.edges:
        for phase in range(graph.phases):
            raw = phase + edge.distance
            distance = raw // graph.phases
            dst_phase = raw % graph.phases
            expanded_edges.append(ExpandedEdge(
                src=f"p{phase}.{edge.src}",
                dst=f"p{dst_phase}.{edge.dst}",
                distance=distance,
                kind=edge.kind,
                base_src=edge.src,
                base_dst=edge.dst,
                phase=phase,
            ))
    return ExpandedGraph(nodes=tuple(nodes), edges=tuple(expanded_edges), index=index)


def _duration_payload(durations: Any) -> tuple[Mapping[str, Any] | None, str | None, str | None, tuple[str, ...]]:
    if durations is None:
        return None, None, "missing_durations", ()
    if not isinstance(durations, Mapping):
        return None, None, "durations_not_a_mapping", ()
    values: Any = durations.get("durations_ns", durations.get("stage_durations_ns", durations))
    if not isinstance(values, Mapping):
        return None, None, "durations_table_invalid", ()
    provenance = durations.get("provenance", durations.get("source"))
    unit = durations.get("unit", durations.get("duration_kind"))
    extra = tuple(sorted(str(key) for key in values if str(key).startswith("_")))
    return values, (str(unit) if unit is not None else None), (str(provenance) if provenance is not None else None), extra


def _is_percentage_unit(unit: Any) -> bool:
    if unit is None:
        return False
    text = str(unit).strip().lower()
    return (
        "%" in text
        or text in {"percent", "percentage", "pipe_utilization", "pipeutilization", "pipe_utilization_percent"}
        or "pipeutilization" in text
        or "pipe_utilization" in text
    )


def normalize_stage_durations(graph: PipelineGraph, durations: Any) -> DurationCalibration:
    values, unit, provenance, extra = _duration_payload(durations)
    if values is None:
        return DurationCalibration(
            status="uncalibrated",
            durations_ns={},
            uncalibrated_stages=tuple(stage.name for stage in graph.stages),
            reasons=(provenance or "missing_durations",),
        )
    expanded = expand_pipeline_graph(graph)
    reasons: list[str] = []
    if _is_percentage_unit(unit) or _is_percentage_unit(provenance):
        reasons.append("raw_pipe_utilization_is_not_service_duration")
    parsed_nodes: dict[str, float] = {}
    parsed_stages: dict[str, float] = {}
    missing: set[str] = set()
    invalid: set[str] = set()
    known_keys = {stage.name for stage in graph.stages} | {node.node_id for node in expanded.nodes}
    for node in expanded.nodes:
        raw = values.get(node.node_id)
        key = node.node_id
        if raw is None:
            raw = values.get(node.stage)
            key = node.stage
        if raw is None:
            missing.add(node.stage)
            continue
        if isinstance(raw, Mapping):
            raw_unit = raw.get("unit", raw.get("kind"))
            raw_source = raw.get("source")
            if _is_percentage_unit(raw_unit) or _is_percentage_unit(raw_source):
                reasons.append("raw_pipe_utilization_is_not_service_duration")
                invalid.add(key)
                continue
            raw = raw.get("duration_ns", raw.get("value"))
        if isinstance(raw, bool) or not _finite(raw) or float(raw) < 0:
            invalid.add(key)
            continue
        parsed_nodes[node.node_id] = float(raw)
        if key == node.stage:
            parsed_stages[node.stage] = float(raw)
    if missing:
        reasons.append("missing_stage_duration")
    if invalid:
        reasons.append("invalid_stage_duration")
    unknown = tuple(sorted(str(key) for key in values if str(key) not in known_keys and not str(key).startswith("_")))
    if reasons:
        return DurationCalibration(
            status="uncalibrated",
            durations_ns=parsed_stages,
            uncalibrated_stages=tuple(sorted(missing | invalid)),
            reasons=tuple(dict.fromkeys(reasons)),
            provenance=provenance,
            extra_keys=unknown,
            node_durations_ns=parsed_nodes,
        )
    return DurationCalibration(
        status="calibrated",
        durations_ns=parsed_stages,
        uncalibrated_stages=(),
        reasons=(),
        provenance=provenance,
        extra_keys=unknown,
        node_durations_ns=parsed_nodes,
    )
