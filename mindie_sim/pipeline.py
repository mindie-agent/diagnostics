"""Layer 2 intra-kernel periodic microstage dependency solver.

[Role]
  Expand a phase-qualified microstage graph (MTE2/MTE1/M/FIX/V/MTE3 FIFO,
  semaphore, ping-pong slot reuse and buffer last-reader edges) and solve the
  exact minimum feasible super-period with Karp's maximum cycle-mean method.

[Invariants]
  * The engine never embeds a specific operator's C++ issue order.  Every stage,
    edge, FIFO sequence and phase distance is an explicit caller input.
  * An edge ``u -> v`` with super-period distance ``d`` enforces
    ``offset(v) - offset(u) + d*P >= service(u)``.  A positive cycle with total
    distance ``d<=0`` is infeasible; a positive-distance cycle contributes
    ``P >= sum(service)/d``.
  * Missing, null, NaN/infinite, negative or raw PipeUtilization percentage
    durations keep the result ``uncalibrated_durations``: no minimum period or
    tile II is fabricated.  Only topology and symbolic cycle expressions remain.
  * A zero-service graph has period zero; this is arithmetic, not evidence that
    a real kernel has zero cost.  Periodic bounds exclude prologue/drain.

[Timeouts & Locks]
  Pure CPU graph computation with bounded DFS/iteration caps.  No locks,
  sleeps, subprocesses, network, or persistent caches.

[Side-Effect Policy]
  Reads only caller-provided mappings.  Never writes files, starts jobs, or
  changes device/host state.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .pipeline_graph import (
    DurationCalibration,
    ExpandedEdge,
    ExpandedGraph,
    ExpandedNode,
    PipelineEdge,
    PipelineError,
    PipelineGraph,
    PipelineStage,
    _finite,
    _int_value,
    expand_pipeline_graph,
    normalize_stage_durations,
    parse_pipeline_graph,
)

SCHEMA_VERSION = "mindie-sim-pipeline/1"
MAX_CYCLES = 64
MAX_EXPANSIONS = 200_000
DEFAULT_TOLERANCE = 1e-9


def _longest_paths(
    expanded: ExpandedGraph,
    services: Sequence[float],
    period: float,
) -> tuple[list[float], bool]:
    n = expanded.node_count
    dist = [0.0] * n
    eps = 1e-12 * max(1.0, abs(period), max(services, default=0.0))
    converged = False
    for _ in range(n):
        updated = False
        for edge in expanded.edges:
            u = expanded.index[edge.src]
            v = expanded.index[edge.dst]
            weight = services[u] - period * edge.distance
            if dist[v] < dist[u] + weight - eps:
                dist[v] = dist[u] + weight
                updated = True
        if not updated:
            converged = True
            break
    return dist, converged


def _positive_cycle(
    expanded: ExpandedGraph,
    services: Sequence[float],
    period: float,
) -> tuple[bool, list[int], list[int]]:
    n = expanded.node_count
    dist = [0.0] * n
    pred = [-1] * n
    pred_edge = [-1] * n
    eps = 1e-12 * max(1.0, abs(period), max(services, default=0.0))
    last = -1
    for _ in range(n):
        last = -1
        for edge_index, edge in enumerate(expanded.edges):
            u = expanded.index[edge.src]
            v = expanded.index[edge.dst]
            weight = services[u] - period * edge.distance
            if dist[v] < dist[u] + weight - eps:
                dist[v] = dist[u] + weight
                pred[v] = u
                pred_edge[v] = edge_index
                last = v
        if last == -1:
            return False, [], []
    if last == -1:
        return False, [], []
    marker = last
    for _ in range(n):
        if pred[marker] == -1:
            return False, [], []
        marker = pred[marker]
    nodes: list[int] = [marker]
    edges: list[int] = []
    current = marker
    for _ in range(n + 1):
        edge_index = pred_edge[current]
        if edge_index == -1:
            return False, [], []
        edges.append(edge_index)
        current = pred[current]
        if current == marker:
            break
        nodes.append(current)
    if current != marker or not edges:
        return False, [], []
    nodes.append(marker)
    nodes.reverse()
    edges.reverse()
    return True, nodes, edges


def _cycle_stats(
    expanded: ExpandedGraph,
    services: Sequence[float],
    nodes: Sequence[int],
    edge_indices: Sequence[int],
) -> tuple[float, int, float]:
    weight = sum(services[node] for node in nodes[:-1])
    distance = sum(expanded.edges[index].distance for index in edge_indices)
    ratio = weight / distance if distance > 0 else math.inf
    return weight, distance, ratio


def _canonical_cycle(expanded: ExpandedGraph, nodes: Sequence[int]) -> tuple[str, ...]:
    body = list(nodes[:-1])
    if not body:
        return ()
    rotations = [tuple(body[index:] + body[:index]) for index in range(len(body))]
    best = min(rotations, key=lambda item: tuple(expanded.nodes[index].node_id for index in item))
    return tuple(expanded.nodes[index].node_id for index in best)


def _tight_cycles(
    expanded: ExpandedGraph,
    services: Sequence[float],
    offsets: Sequence[float],
    period: float,
    tolerance: float,
) -> list[dict[str, Any]]:
    n = expanded.node_count
    adjacency: dict[int, list[int]] = {index: [] for index in range(n)}
    for edge_index, edge in enumerate(expanded.edges):
        u = expanded.index[edge.src]
        v = expanded.index[edge.dst]
        slack = offsets[v] - offsets[u] - (services[u] - period * edge.distance)
        if slack <= tolerance:
            adjacency[u].append(edge_index)
    for edge_list in adjacency.values():
        edge_list.sort()
    starts = sorted(range(n), key=lambda index: (-services[index], index))
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    expansions = 0

    def walk(start: int, current: int, path: list[int], visited: set[int], edge_path: list[int]) -> None:
        nonlocal expansions
        if len(found) >= MAX_CYCLES or expansions >= MAX_EXPANSIONS:
            return
        for edge_index in adjacency[current]:
            expansions += 1
            edge = expanded.edges[edge_index]
            target = expanded.index[edge.dst]
            if target == start:
                if len(path) < 2:
                    continue
                key = _canonical_cycle(expanded, path + [start])
                if key in seen:
                    continue
                seen.add(key)
                weight, distance, ratio = _cycle_stats(expanded, services, path + [start], edge_path + [edge_index])
                found.append({
                    "nodes": [expanded.nodes[index].node_id for index in path + [start]],
                    "edge_indices": edge_path + [edge_index],
                    "weight_ns": weight,
                    "distance": distance,
                    "cycle_mean_ns": ratio,
                })
            elif target not in visited and len(path) < min(n, 48):
                visited.add(target)
                path.append(target)
                edge_path.append(edge_index)
                walk(start, target, path, visited, edge_path)
                edge_path.pop()
                path.pop()
                visited.remove(target)

    for start in starts:
        walk(start, start, [start], {start}, [])
        if len(found) >= MAX_CYCLES or expansions >= MAX_EXPANSIONS:
            break
    return found


def _format_cycle(expanded: ExpandedGraph, nodes: Sequence[int], distance: int) -> str:
    stage_names = [expanded.nodes[index].stage for index in nodes[:-1]]
    numerator = " + ".join(stage_names)
    if distance > 0:
        return f"P >= ({numerator})/{distance}"
    if distance == 0:
        return f"0 >= {numerator} (zero-distance positive service is infeasible)"
    return f"P <= ({numerator})/{distance} (negative-distance upper bound)"


def _cycle_payload(expanded: ExpandedGraph, services: Sequence[float], cycle: Mapping[str, Any], pipes: Mapping[str, str | None]) -> dict[str, Any]:
    nodes = cycle["nodes"]
    indices = [expanded.index[node] for node in nodes]
    stages = [expanded.nodes[index].stage for index in indices[:-1]]
    unique_stages = list(dict.fromkeys(stages))
    cycle_pipes = sorted({pipes.get(stage) or "unknown" for stage in unique_stages})
    return {
        "nodes": list(nodes),
        "stages": stages,
        "unique_stages": unique_stages,
        "distance": cycle["distance"],
        "service_sum_ns": cycle["weight_ns"],
        "cycle_mean_ns": cycle["cycle_mean_ns"],
        "bound_ns": cycle["cycle_mean_ns"],
        "expression": _format_cycle(expanded, indices, cycle["distance"]),
        "pipes": cycle_pipes,
        "tight": True,
    }


def _solve_expanded(
    expanded: ExpandedGraph,
    graph: PipelineGraph,
    services: Sequence[float],
    *,
    target_us: float | None = None,
    total_tiles: int | None = None,
    reference_mhz: float | None = None,
) -> dict[str, Any]:
    if len(services) != expanded.node_count:
        raise PipelineError("service_vector_length_mismatch")
    n = expanded.node_count
    max_service = max(services, default=0.0)
    if not all(_finite(value) and value >= 0 for value in services):
        raise PipelineError("service_vector_invalid")
    found_zero, zero_nodes, zero_edges = _positive_cycle(expanded, services, 0.0)
    if found_zero:
        _, zero_distance, _ = _cycle_stats(expanded, services, zero_nodes, zero_edges)
        if zero_distance <= 0:
            return {
                "status": "infeasible_nonpositive_distance_cycle",
                "minimum_period_ns": None,
                "minimum_tile_ii_ns": None,
                "reason": "positive service cycle has non-positive super-period distance",
            }
    if max_service == 0.0:
        period = 0.0
        offsets = [0.0] * n
        critical: list[dict[str, Any]] = []
        iterations = 0
        tolerance = 0.0
    else:
        high = max(1.0, n * max_service + 1.0)
        iterations = 0
        while True:
            positive, _, _ = _positive_cycle(expanded, services, high)
            if not positive:
                break
            high *= 2.0
            iterations += 1
            if high > 1e12:
                return {
                    "status": "infeasible_no_finite_period",
                    "minimum_period_ns": None,
                    "minimum_tile_ii_ns": None,
                    "reason": "positive cycle persists above the finite-period bracket",
                }
        low = 0.0
        for _ in range(100):
            middle = (low + high) / 2.0
            positive, _, _ = _positive_cycle(expanded, services, middle)
            if positive:
                low = middle
            else:
                high = middle
            iterations += 1
            if high - low <= DEFAULT_TOLERANCE * max(1.0, high):
                break
        period = high
        offsets, converged = _longest_paths(expanded, services, period * (1.0 + 1e-10))
        if not converged:
            offsets = [0.0] * n
        tolerance = max(1e-6, (high - low) * max(1, n) * 4.0, DEFAULT_TOLERANCE * max(1.0, period))
        critical = _tight_cycles(expanded, services, offsets, period, tolerance)
        for item in critical:
            if item["distance"] > 0:
                item["cycle_mean_ns"] = item["weight_ns"] / item["distance"]
        critical = [item for item in critical if item["distance"] > 0 and math.isfinite(item["cycle_mean_ns"])]
        critical.sort(key=lambda item: item["cycle_mean_ns"], reverse=True)
        critical = critical[:12]
    pipes = {stage.name: stage.pipe for stage in graph.stages}
    critical_payload = [
        _cycle_payload(expanded, services, item, pipes)
        for item in critical
    ]
    if critical_payload:
        period = max(item["cycle_mean_ns"] for item in critical_payload)
    if period > 0:
        offsets, converged = _longest_paths(expanded, services, period * (1.0 + 1e-9))
        if not converged:
            offsets, _ = _longest_paths(expanded, services, period + 1e-6)
    minimum_offset = min(offsets) if offsets else 0.0
    schedule = {
        expanded.nodes[index].node_id: float(offsets[index] - minimum_offset)
        for index in range(n)
    }
    tile_ii = period / graph.tiles_per_period
    payload: dict[str, Any] = {
        "status": "modeled",
        "minimum_period_ns": period,
        "minimum_tile_ii_ns": tile_ii,
        "tiles_per_period": graph.tiles_per_period,
        "critical_cycles": critical_payload,
        "critical_recurrence_cycles": [item["expression"] for item in critical_payload],
        "bottleneck_pipes": sorted({pipe for item in critical_payload for pipe in item["pipes"] if pipe != "unknown"}),
        "solver": {
            "algorithm": "karp_max_cycle_mean",
            "implementation": "distance-weighted parametric longest-path feasibility with exact critical-cycle ratio",
            "distance_weighted": True,
            "iterations": iterations,
            "tolerance_ns": tolerance,
            "node_count": n,
            "edge_count": expanded.edge_count,
        },
        "schedule_offsets_ns": schedule,
    }
    if total_tiles is not None:
        total_tiles = _int_value(total_tiles, "total_tiles_invalid")
        if total_tiles <= 0:
            raise PipelineError("total_tiles_invalid")
        payload["total_tiles"] = total_tiles
        payload["total_time_us"] = tile_ii * total_tiles / 1000.0
    if target_us is not None and total_tiles is not None:
        target = float(target_us)
        if not _finite(target) or target <= 0:
            raise PipelineError("target_us_invalid")
        target_tile_ii = target * 1000.0 / total_tiles
        target_period = target_tile_ii * graph.tiles_per_period
        target_payload: dict[str, Any] = {
            "target_us": target,
            "total_tiles": total_tiles,
            "target_tile_ii_ns": target_tile_ii,
            "target_ns_per_tile": target_tile_ii,
            "target_super_period_ns": target_period,
            "feasible": tile_ii <= target_tile_ii + DEFAULT_TOLERANCE * max(1.0, target_tile_ii),
            "headroom_ns_per_tile": target_tile_ii - tile_ii,
        }
        if reference_mhz is not None and _finite(reference_mhz) and float(reference_mhz) > 0:
            target_payload["allowed_service_cycles_per_tile_at_mhz"] = target_tile_ii * float(reference_mhz) / 1000.0
            target_payload["reference_mhz"] = float(reference_mhz)
        payload["target"] = target_payload
    return payload


def _symbolic_cycles(graph: PipelineGraph, *, max_cycles: int = 32) -> list[dict[str, Any]]:
    adjacency: dict[str, list[tuple[str, int, str]]] = {stage.name: [] for stage in graph.stages}
    for edge in graph.edges:
        adjacency[edge.src].append((edge.dst, edge.distance, edge.kind))
    cycles: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, ...], int]] = set()

    def add(stages: Sequence[str], distance: int, kind: str) -> None:
        key = (tuple(stages), distance)
        if key in seen or len(cycles) >= max_cycles:
            return
        seen.add(key)
        if distance > 0:
            expression = f"P >= ({' + '.join(stages)})/{distance}"
            feasibility = "positive_distance"
        elif distance == 0:
            expression = f"0 >= {' + '.join(stages)} (zero-distance positive service is infeasible)"
            feasibility = "zero_distance_requires_zero_service"
        else:
            expression = f"P <= ({' + '.join(stages)})/{distance} (negative-distance upper bound)"
            feasibility = "negative_distance_upper_bound"
        cycles.append({
            "stages": list(stages),
            "distance": distance,
            "kind": kind,
            "expression": expression,
            "feasibility": feasibility,
        })

    for pipe, entries in graph.pipe_fifo.items():
        stages = [stage for stage, _ in entries]
        closure = 1 + entries[0][1] - entries[-1][1]
        distance = closure + sum(entries[index + 1][1] - entries[index][1] for index in range(len(entries) - 1))
        add(stages, distance, f"pipe_fifo:{pipe}")
    expansions = 0

    def walk(start: str, current: str, path: list[str], visited: set[str], distance_so_far: int) -> None:
        nonlocal expansions
        if len(cycles) >= max_cycles or expansions >= 50_000:
            return
        for target, distance, kind in adjacency[current]:
            expansions += 1
            if target == start:
                add(path, distance_so_far + distance, kind)
            elif target not in visited and len(path) < min(len(graph.stages), 16):
                visited.add(target)
                path.append(target)
                walk(start, target, path, visited, distance_so_far + distance)
                path.pop()
                visited.remove(target)

    for stage in graph.stages:
        walk(stage.name, stage.name, [stage.name], {stage.name}, 0)
        if len(cycles) >= max_cycles:
            break
    return cycles


def _toy_relaxation(name: str, base_distance: int, relaxed_distance: int) -> dict[str, Any]:
    if base_distance <= 0 or relaxed_distance <= base_distance:
        return {"name": name, "strictly_relaxes": False, "reason": "distances_do_not_relax"}
    spec = {
        "name": f"toy-{name}",
        "phases": 1,
        "tiles_per_period": 1,
        "stages": [{"name": "a"}, {"name": "b"}],
        "edges": [
            {"src": "a", "dst": "b", "distance": base_distance, "kind": "toy"},
            {"src": "b", "dst": "a", "distance": 0, "kind": "toy"},
        ],
    }
    base = parse_pipeline_graph(spec)
    base_result = _solve_expanded(expand_pipeline_graph(base), base, [1.0, 1.0])
    relaxed_spec = {
        **spec,
        "edges": [
            {"src": "a", "dst": "b", "distance": relaxed_distance, "kind": "toy"},
            {"src": "b", "dst": "a", "distance": 0, "kind": "toy"},
        ],
    }
    relaxed = parse_pipeline_graph(relaxed_spec)
    relaxed_result = _solve_expanded(expand_pipeline_graph(relaxed), relaxed, [1.0, 1.0])
    base_period = base_result.get("minimum_period_ns")
    relaxed_period = relaxed_result.get("minimum_period_ns")
    return {
        "name": name,
        "strictly_relaxes": bool(
            base_period is not None and relaxed_period is not None and relaxed_period < base_period
        ),
        "note": "unit service weights only; periods are not reported as latency",
    }


def pipeline_self_check(graph: PipelineGraph | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(graph, Mapping):
        graph = parse_pipeline_graph(graph)
    expanded = expand_pipeline_graph(graph)
    failures: list[str] = []
    stage_names = {stage.name for stage in graph.stages}
    expected_node_count = len(graph.stages) * graph.phases
    if expanded.node_count != expected_node_count:
        failures.append("node_count_mismatch")
    if len(stage_names) != len(graph.stages):
        failures.append("duplicate_stage_names")
    expected_edges: set[tuple[str, str, int]] = set()
    for edge in graph.edges:
        for phase in range(graph.phases):
            raw = phase + edge.distance
            expected_edges.add((f"p{phase}.{edge.src}", f"p{raw % graph.phases}.{edge.dst}", raw // graph.phases))
    actual_edges = {(edge.src, edge.dst, edge.distance) for edge in expanded.edges}
    if expected_edges != actual_edges:
        failures.append("distance_expansion_mismatch")
    for edge in graph.edges:
        if edge.src == edge.dst and edge.distance <= 0:
            failures.append(f"zero_distance_self_loop:{edge.src}")
    fifo_checks = []
    for pipe, entries in graph.pipe_fifo.items():
        if not entries:
            failures.append(f"pipe_fifo_empty:{pipe}")
            continue
        closure_distance = 1 + entries[0][1] - entries[-1][1]
        if closure_distance <= 0:
            failures.append(f"fifo_closure_nonpositive:{pipe}")
        fifo_checks.append({
            "pipe": pipe,
            "commands": [stage for stage, _ in entries],
            "closure_distance": closure_distance,
        })
    slot_failures: list[dict[str, Any]] = []
    slot_checks = 0
    same_slot_edges = [edge for edge in graph.edges if edge.same_slot]
    if same_slot_edges:
        if graph.transactions_per_loop is None:
            failures.append("slot_check_missing_transactions_per_loop")
        elif graph.phases % 2 != 0:
            failures.append("slot_check_requires_even_phases")
        else:
            tx = graph.transactions_per_loop
            stage_tx = {stage.name: stage.transaction_index for stage in graph.stages}
            for edge in same_slot_edges:
                src_tx = stage_tx.get(edge.src)
                dst_tx = stage_tx.get(edge.dst)
                if src_tx is None or dst_tx is None:
                    failures.append(f"slot_check_missing_transaction_index:{edge.src}->{edge.dst}")
                    continue
                for phase in range(graph.phases):
                    slot_checks += 1
                    dst_phase = (phase + edge.distance) % graph.phases
                    src_slot = (tx * phase + src_tx) % 2
                    dst_slot = (tx * dst_phase + dst_tx) % 2
                    if src_slot != dst_slot:
                        slot_failures.append({
                            "edge": f"{edge.src}->{edge.dst}",
                            "phase": phase,
                            "slot": edge.same_slot,
                            "source_slot": src_slot,
                            "target_slot": dst_slot,
                        })
            if slot_failures:
                failures.extend(f"slot_identity_mismatch:{item['edge']}:phase{item['phase']}" for item in slot_failures)
    relaxation_results: list[dict[str, Any]] = []
    for check in graph.relaxation_checks:
        name = str(check.get("name", "relaxation"))
        base_distance = check.get("base_distance", check.get("pv1_distance", 1))
        relaxed_distance = check.get("relaxed_distance", check.get("pv2_distance", 2))
        try:
            base_distance = _int_value(base_distance, "relaxation_distance_invalid")
            relaxed_distance = _int_value(relaxed_distance, "relaxation_distance_invalid")
        except PipelineError:
            failures.append(f"relaxation_distance_invalid:{name}")
            continue
        result = _toy_relaxation(name, base_distance, relaxed_distance)
        relaxation_results.append(result)
        if not result["strictly_relaxes"]:
            failures.append(f"relaxation_not_strict:{name}")
    finite_schedule = False
    try:
        unit_result = _solve_expanded(expanded, graph, [1.0] * expanded.node_count)
        finite_schedule = unit_result.get("status") == "modeled" and unit_result.get("minimum_period_ns") is not None
    except PipelineError:
        finite_schedule = False
    if not finite_schedule:
        failures.append("unit_weight_schedule_not_finite")
    relaxation_by_name = {item["name"]: item["strictly_relaxes"] for item in relaxation_results}
    return {
        "status": "ok" if not failures else "violations",
        "ok": not failures,
        "topology_valid": not failures,
        "pv2_strictly_relaxes_pv1": relaxation_by_name.get("pv_ub"),
        "score2_strictly_relaxes_score1": relaxation_by_name.get("score_l0c"),
        "checks": {
            "node_count": expanded.node_count,
            "expected_node_count": expected_node_count,
            "edge_count": expanded.edge_count,
            "stage_count": len(graph.stages),
            "distance_expansion_ok": "distance_expansion_mismatch" not in failures,
            "fifo_checks": fifo_checks,
            "slot_identity": {
                "checks": slot_checks,
                "failures": slot_failures,
                "status": "ok" if not slot_failures else "violations",
            },
            "dimensionless_relaxation": {
                "checks": relaxation_results,
                "note": "Unit service weights are dimensionless and never reported as latency.",
            },
            "finite_schedule": finite_schedule,
        },
        "failures": failures,
    }


def solve_periodic_pipeline(
    graph: PipelineGraph | Mapping[str, Any],
    durations: Any = None,
    *,
    target_us: float | None = None,
    total_tiles: int | None = None,
    reference_mhz: float | None = None,
    run_self_check: bool = True,
    source: str = "<mapping>",
) -> dict[str, Any]:
    if isinstance(graph, Mapping):
        graph = parse_pipeline_graph(graph, source=source)
    expanded = expand_pipeline_graph(graph)
    calibration = normalize_stage_durations(graph, durations)
    self_check = pipeline_self_check(graph) if run_self_check else None
    summary = {
        "name": graph.name,
        "variant": graph.variant,
        "phases": graph.phases,
        "tiles_per_period": graph.tiles_per_period,
        "stage_count": len(graph.stages),
        "node_count": expanded.node_count,
        "edge_count": expanded.edge_count,
        "base_edge_count": len(graph.edges),
        "source": graph.source,
    }
    if not calibration.calibrated:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "uncalibrated_durations",
            "fabricated_prediction": False,
            "graph": summary,
            "minimum_period_ns": None,
            "minimum_tile_ii_ns": None,
            "total_time_us": None,
            "durations": {
                "status": calibration.status,
                "reasons": list(calibration.reasons),
                "uncalibrated_stages": list(calibration.uncalibrated_stages),
                "provenance": calibration.provenance,
                "note": "No numeric period is inferred from missing or raw percentage durations.",
            },
            "self_check": self_check,
            "symbolic_cycles": _symbolic_cycles(graph),
            "critical_cycles": [],
            "critical_recurrence_cycles": [],
        }
    services = [
        calibration.node_durations_ns[node.node_id]
        if node.node_id in calibration.node_durations_ns
        else calibration.durations_ns[node.stage]
        for node in expanded.nodes
    ]
    solved = _solve_expanded(
        expanded,
        graph,
        services,
        target_us=target_us,
        total_tiles=total_tiles,
        reference_mhz=reference_mhz,
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "graph": summary,
        "durations": {
            "status": calibration.status,
            "provenance": calibration.provenance,
            "extra_keys": list(calibration.extra_keys),
        },
        "self_check": self_check,
    }
    payload.update(solved)
    if solved.get("status") != "modeled":
        payload["fabricated_prediction"] = False
    return payload


pipeline_report = solve_periodic_pipeline


def compare_pipeline_variants(
    base: PipelineGraph | Mapping[str, Any],
    variant: PipelineGraph | Mapping[str, Any],
    durations: Any = None,
    *,
    target_us: float | None = None,
    total_tiles: int | None = None,
    reference_mhz: float | None = None,
) -> dict[str, Any]:
    base_graph = parse_pipeline_graph(base) if isinstance(base, Mapping) else base
    variant_graph = parse_pipeline_graph(variant) if isinstance(variant, Mapping) else variant
    base_result = solve_periodic_pipeline(
        base_graph, durations, target_us=target_us, total_tiles=total_tiles,
        reference_mhz=reference_mhz, source=base_graph.source,
    )
    variant_result = solve_periodic_pipeline(
        variant_graph, durations, target_us=target_us, total_tiles=total_tiles,
        reference_mhz=reference_mhz, source=variant_graph.source,
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "modeled" if base_result.get("status") == "modeled" and variant_result.get("status") == "modeled" else "uncalibrated_or_infeasible",
        "base": base_result,
        "variant": variant_result,
    }
    if payload["status"] == "modeled":
        base_period = base_result["minimum_period_ns"]
        variant_period = variant_result["minimum_period_ns"]
        payload["comparison"] = {
            "base_minimum_period_ns": base_period,
            "variant_minimum_period_ns": variant_period,
            "delta_ns": variant_period - base_period,
            "relaxed": variant_period <= base_period + DEFAULT_TOLERANCE * max(1.0, base_period),
            "note": "A smaller period is a theoretical relaxation under the supplied calibrated durations.",
        }
    return payload


__all__ = [
    "SCHEMA_VERSION",
    "MAX_CYCLES",
    "MAX_EXPANSIONS",
    "DEFAULT_TOLERANCE",
    "PipelineError",
    "PipelineStage",
    "PipelineEdge",
    "PipelineGraph",
    "ExpandedNode",
    "ExpandedEdge",
    "ExpandedGraph",
    "DurationCalibration",
    "parse_pipeline_graph",
    "expand_pipeline_graph",
    "normalize_stage_durations",
    "solve_periodic_pipeline",
    "pipeline_report",
    "pipeline_self_check",
    "compare_pipeline_variants",
]
