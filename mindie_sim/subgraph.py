"""Layer 3 producer-consumer subgraph and cross-operator fusion analyzer.

[Role]
  Account removable logical GM round-trip bytes for a candidate fusion, enforce
  hard feasibility gates (collective barriers, L1/UB live-buffer limits,
  non-elidable DCP normalization, route/layout contracts), and compute the
  critical-path saving interval under the concurrent tail ``max(P+C, Q+G)``.

[Invariants]
  * ``removable_gm_bytes`` is logical GM traffic.  It is never divided by HBM
    bandwidth to create a latency claim: L2 residency can absorb logical reads.
  * An AICPU ``HcclLaunch`` duration is never substituted for device
    communication time ``C``; doing so returns an explicit invalid/unmodeled
    result instead of a numeric saving.
  * Hard fusion gates are checked before latency is reported.  A gate failure
    keeps the byte accounting but disables the saving interval.
  * Missing node durations keep latency unmodeled.  No empirical efficiency
    discount or guessed duration is introduced.
  * The saving interval is ``[0, max(tail_before - tail_after, 0)]``; zero is the
    no-guarantee bound when the removed work is already hidden.

[Timeouts & Locks]
  Pure DAG computation with bounded traversal.  No locks, sleeps, subprocesses,
  network, or persistent caches.

[Side-Effect Policy]
  Reads caller-provided mappings only.  Never writes files or changes devices.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Mapping, Sequence

from .hardware import validate_buffer_allocations, DEFAULT_ALIGNMENT_BYTES

SCHEMA_VERSION = "mindie-sim-subgraph/1"
DEFAULT_CAPACITIES_KIB = {"L1": 512, "L0A": 64, "L0B": 64, "L0C": 256, "UB": 248}
_AICPU_LAUNCH_KINDS = {"aicpu_hccl_launch", "aicpu_launch", "hccl_launch", "aicpu"}
_DEVICE_COMM_KINDS = {"device_comm", "hccl", "device", "collective_device"}


class SubgraphError(ValueError):
    """Stable subgraph-input validation failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _duration(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not _finite(value) or float(value) < 0:
        return None
    return float(value)


def _bytes_from_tensor(tensor: Mapping[str, Any] | None) -> float | None:
    if not isinstance(tensor, Mapping):
        return None
    for key, multiplier in (("bytes", 1), ("size_bytes", 1), ("size_kib", 1024), ("size_mib", 1024 * 1024)):
        if tensor.get(key) is not None and _finite(tensor[key]) and float(tensor[key]) >= 0:
            return float(tensor[key]) * multiplier
    return None


def _node_map(dag: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    raw_nodes = dag.get("nodes")
    if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes)) or not raw_nodes:
        raise SubgraphError("nodes_missing")
    nodes: dict[str, dict[str, Any]] = {}
    order: list[dict[str, Any]] = []
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            raise SubgraphError("node_invalid")
        node_id = str(raw.get("id", "")).strip()
        if not node_id:
            raise SubgraphError("node_id_empty")
        if node_id in nodes:
            raise SubgraphError(f"node_id_duplicate:{node_id}")
        node = dict(raw)
        node["id"] = node_id
        nodes[node_id] = node
        order.append(node)
    return nodes, order


def _edge_list(dag: Mapping[str, Any], nodes: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_edges = dag.get("edges", ())
    if raw_edges is None:
        raw_edges = ()
    if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
        raise SubgraphError("edges_invalid")
    edges: list[dict[str, Any]] = []
    for raw in raw_edges:
        if not isinstance(raw, Mapping):
            raise SubgraphError("edge_invalid")
        src = str(raw.get("src", "")).strip()
        dst = str(raw.get("dst", "")).strip()
        if src not in nodes or dst not in nodes:
            raise SubgraphError(f"edge_endpoint_unknown:{src}->{dst}")
        edge = dict(raw)
        edge["src"] = src
        edge["dst"] = dst
        edges.append(edge)
    return edges


def _node_kind(node: Mapping[str, Any]) -> str:
    return str(node.get("duration_kind", node.get("kind", ""))).strip().lower().replace("-", "_")


def _is_aicpu_launch(node: Mapping[str, Any]) -> bool:
    kind = _node_kind(node)
    return kind in _AICPU_LAUNCH_KINDS or "hccllaunch" in kind


def _gate(feasible: bool, code: str, detail: Any = None) -> dict[str, Any]:
    result = {"code": code, "feasible": feasible}
    if detail is not None:
        result["detail"] = detail
    return result


def _resolve_remove_nodes(candidate: Mapping[str, Any], nodes: Mapping[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    remove_nodes = candidate.get("remove_nodes", ())
    if isinstance(remove_nodes, str):
        remove_nodes = [remove_nodes]
    if not isinstance(remove_nodes, Sequence):
        raise SubgraphError("remove_nodes_invalid")
    result: list[str] = []
    for value in remove_nodes:
        node_id = str(value)
        if node_id not in nodes:
            raise SubgraphError(f"remove_node_unknown:{node_id}")
        if node_id not in result:
            result.append(node_id)
    return result, [nodes[node_id] for node_id in result]


def _resolve_remove_edges(
    candidate: Mapping[str, Any],
    edges: Sequence[Mapping[str, Any]],
    producer: str | None,
    consumer: str | None,
    remove_nodes: Sequence[str],
) -> list[dict[str, Any]]:
    raw = candidate.get("remove_edges")
    if raw is None:
        removed_set = set(remove_nodes)
        return [
            dict(edge) for edge in edges
            if (edge["src"] == producer and (edge["dst"] == consumer or edge["dst"] in removed_set))
            or (edge["src"] in removed_set and (edge["dst"] == consumer or edge["dst"] in removed_set))
        ]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise SubgraphError("remove_edges_invalid")
    edge_index = {(edge["src"], edge["dst"]): edge for edge in edges}
    result: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            src, dst = str(item.get("src", "")), str(item.get("dst", ""))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            src, dst = str(item[0]), str(item[1])
        else:
            raise SubgraphError("remove_edge_invalid")
        edge = edge_index.get((src, dst))
        if edge is None:
            raise SubgraphError(f"remove_edge_unknown:{src}->{dst}")
        result.append(edge)
    return result


def _gm_bytes_for_edge(edge: Mapping[str, Any]) -> dict[str, Any]:
    tensor = edge.get("tensor")
    tensor_bytes = _bytes_from_tensor(tensor if isinstance(tensor, Mapping) else None)
    write = edge.get("gm_write_bytes")
    read = edge.get("gm_read_bytes")
    if write is None and read is None:
        if edge.get("gm_roundtrip", True) is False:
            write = tensor_bytes
            read = 0.0
        else:
            write = tensor_bytes
            read = tensor_bytes
    write_value = _duration(write) if write is not None else None
    read_value = _duration(read) if read is not None else None
    if write_value is None and read_value is None:
        return {"src": edge["src"], "dst": edge["dst"], "bytes": None, "write_bytes": None, "read_bytes": None}
    write_value = write_value or 0.0
    read_value = read_value or 0.0
    return {
        "src": edge["src"],
        "dst": edge["dst"],
        "tensor": dict(tensor) if isinstance(tensor, Mapping) else None,
        "write_bytes": write_value,
        "read_bytes": read_value,
        "bytes": write_value + read_value,
    }


def _topological_order(nodes: Mapping[str, Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> list[str]:
    indegree = {node_id: 0 for node_id in nodes}
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        adjacency[edge["src"]].append(edge["dst"])
        indegree[edge["dst"]] += 1
    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for target in adjacency[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(order) != len(nodes):
        raise SubgraphError("subgraph_cycle_detected")
    return order


def _critical_path(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    *,
    removed: set[str] | None = None,
    producer: str | None = None,
    fused_extra_us: float = 0.0,
) -> dict[str, Any]:
    removed = removed or set()
    active = {node_id: node for node_id, node in nodes.items() if node_id not in removed}
    if not active:
        return {"status": "empty_graph", "critical_path_us": None}
    rewired: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()
    for edge in edges:
        src, dst = edge["src"], edge["dst"]
        if src in removed and dst in removed:
            continue
        if src in removed:
            if producer is None or producer in removed:
                continue
            src = producer
        if dst in removed:
            continue
        if src == dst or src not in active or dst not in active:
            continue
        key = (src, dst)
        if key not in seen_edges:
            seen_edges.add(key)
            rewired.append({"src": src, "dst": dst})
    duration: dict[str, float | None] = {}
    for node_id, node in active.items():
        value = _duration(node.get("duration_us"))
        duration[node_id] = value
    if any(value is None for value in duration.values()):
        return {
            "status": "unmodeled_durations",
            "critical_path_us": None,
            "missing_duration_nodes": sorted(node_id for node_id, value in duration.items() if value is None),
        }
    if producer in active and fused_extra_us:
        duration[producer] = (duration[producer] or 0.0) + fused_extra_us
    order = _topological_order(active, rewired)
    longest: dict[str, float] = {}
    previous: dict[str, str | None] = {}
    for node_id in order:
        predecessors = [edge["src"] for edge in rewired if edge["dst"] == node_id]
        if predecessors:
            best = max(predecessors, key=lambda item: longest[item])
            longest[node_id] = longest[best] + float(duration[node_id] or 0.0)
            previous[node_id] = best
        else:
            longest[node_id] = float(duration[node_id] or 0.0)
            previous[node_id] = None
    end = max(longest, key=longest.get)
    path: list[str] = []
    current: str | None = end
    while current is not None and len(path) <= len(active):
        path.append(current)
        current = previous[current]
    path.reverse()
    return {
        "status": "modeled",
        "critical_path_us": longest[end],
        "critical_path_nodes": path,
        "node_durations_us": {node_id: duration[node_id] for node_id in order},
    }


def _contract_gate(
    candidate: Mapping[str, Any],
    remove_edges: Sequence[Mapping[str, Any]],
    nodes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if candidate.get("allow_layout_change"):
        return _gate(True, "layout_scale_contract", {"allowed": True})
    explicit = candidate.get("requires_layout_change")
    if explicit is True:
        return _gate(False, "layout_scale_contract_change", {"reason": "candidate_requires_layout_or_scale_change"})
    expected = candidate.get("consumer_input_contract")
    if isinstance(expected, Mapping):
        for edge in remove_edges:
            tensor = edge.get("tensor")
            if not isinstance(tensor, Mapping):
                continue
            for key in ("dtype", "layout", "scale"):
                if key in expected and key in tensor and str(expected[key]) != str(tensor[key]):
                    return _gate(False, "layout_scale_contract_mismatch", {
                        "src": edge["src"], "dst": edge["dst"], "field": key,
                        "producer": tensor[key], "consumer": expected[key],
                    })
    return _gate(True, "layout_scale_contract")


def _buffer_gate(
    candidate: Mapping[str, Any],
    dag: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    remove_nodes: Sequence[str],
) -> dict[str, Any]:
    allocations = candidate.get("live_buffers")
    if allocations is None:
        allocations = []
        for node_id in remove_nodes:
            node = nodes[node_id]
            for allocation in node.get("live_buffers", ()) or ():
                allocations.append(allocation)
    if not allocations:
        return _gate(True, "live_buffer_limit", {"checked": False, "reason": "no_live_buffers_supplied"})
    capacities = candidate.get("capacities_kib", dag.get("capacities_kib", DEFAULT_CAPACITIES_KIB))
    if not isinstance(capacities, Mapping):
        return _gate(False, "live_buffer_limit", {"reason": "capacities_invalid"})
    check = validate_buffer_allocations(
        allocations,
        capacities,
        alignment_bytes=int(candidate.get("alignment_bytes", DEFAULT_ALIGNMENT_BYTES)),
        ping_pong=candidate.get("ping_pong"),
        pack_missing_offsets=True,
    )
    return _gate(
        check["ok"],
        "live_buffer_limit",
        {
            "checked": True,
            "capacity_source": "candidate_or_dag" if "capacities_kib" in candidate or "capacities_kib" in dag else "default_a5_reference",
            "status": check["status"],
            "diagnostics": check["diagnostics"],
        },
    )


def _dcp_gate(candidate: Mapping[str, Any], removed_nodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    elide = bool(candidate.get("elide_normalization")) or bool(candidate.get("elide_full_reduction"))
    for node in removed_nodes:
        text = " ".join(
            str(node.get(key, "")) for key in ("op", "kind", "name", "normalization", "semantic")
        ).lower()
        if ("dcp" in text or "merge" in text) and (
            "normalization" in text or "lse" in text or "full_reduction" in text or "reduce" in text
        ):
            if elide:
                return _gate(False, "non_elidable_full_reduction_normalization", {
                    "node": node["id"],
                    "normalization": node.get("normalization"),
                    "reason": "DCP full-reduction/LSE normalization must remain in the consumer",
                })
            return _gate(True, "non_elidable_full_reduction_normalization", {
                "node": node["id"],
                "normalization_preserved": True,
            })
    return _gate(True, "non_elidable_full_reduction_normalization", {"checked": False})


def _collective_gate(
    candidate: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    remove_nodes: Sequence[str],
    remove_edges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    allowed = bool(candidate.get("allow_collective_barrier"))
    merged = set(remove_nodes)
    if candidate.get("producer"):
        merged.add(str(candidate["producer"]))
    if candidate.get("consumer"):
        merged.add(str(candidate["consumer"]))
    for node_id in merged:
        node = nodes.get(node_id, {})
        if bool(node.get("collective")) or bool(node.get("collective_barrier")):
            if allowed:
                return _gate(True, "collective_barrier_requires_global_sync", {
                    "node": node_id, "barrier_preserved": True,
                })
            return _gate(False, "collective_barrier_requires_global_sync", {"node": node_id})
    for edge in remove_edges:
        if bool(edge.get("collective_barrier")) or bool(edge.get("collective")):
            if allowed:
                return _gate(True, "collective_barrier_requires_global_sync", {
                    "edge": f"{edge['src']}->{edge['dst']}", "barrier_preserved": True,
                })
            return _gate(False, "collective_barrier_requires_global_sync", {"edge": f"{edge['src']}->{edge['dst']}"})
    return _gate(True, "collective_barrier_requires_global_sync", {"allow_collective_barrier": allowed})


def _route_gate(candidate: Mapping[str, Any], removed_nodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    preserve = bool(candidate.get("preserve_route", False))
    for node in removed_nodes:
        text = " ".join(str(node.get(key, "")) for key in ("op", "kind", "name", "route")).lower()
        if "zeroexpertrouter" in text or "zero_expert_router" in text or bool(node.get("route_fallback")):
            if not preserve:
                return _gate(False, "route_fallback_contract", {"node": node["id"]})
            return _gate(True, "route_fallback_contract", {"node": node["id"], "preserved": True})
    return _gate(True, "route_fallback_contract", {"checked": False})


def _overlap_analysis(
    dag: Mapping[str, Any],
    candidate: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    overlap = candidate.get("overlap")
    if not isinstance(overlap, Mapping):
        return {"status": "not_supplied", "reason": "overlap_topology_missing"}
    def resolve(key: str, explicit_key: str) -> tuple[float | None, str | None, Mapping[str, Any] | None]:
        if overlap.get(explicit_key) is not None:
            value = _duration(overlap[explicit_key])
            return value, None, None
        node_id = overlap.get(key)
        if node_id is None:
            return None, f"missing_{key}", None
        node = nodes.get(str(node_id))
        if node is None:
            return None, f"unknown_{key}:{node_id}", None
        return _duration(node.get("duration_us")), None, node
    pack, pack_error, _ = resolve("pack", "pack_us")
    comm, comm_error, comm_node = resolve("comm", "comm_us")
    current, current_error, _ = resolve("current", "current_us")
    gate, gate_error, _ = resolve("gate", "gate_us")
    errors = [value for value in (pack_error, comm_error, current_error, gate_error) if value]
    if errors:
        return {"status": "unmodeled_durations", "reasons": errors}
    if comm_node is not None and _is_aicpu_launch(comm_node):
        return {
            "status": "invalid_communication_duration",
            "reason": "aicpu_hccl_launch_is_not_device_communication",
            "communication_node": comm_node.get("id"),
            "communication_kind": _node_kind(comm_node),
            "min_saving_us": None,
            "max_saving_us": None,
        }
    assert pack is not None and comm is not None and current is not None and gate is not None
    pack_comm = pack + comm
    current_gate = current + gate
    tail_before = max(pack_comm, current_gate)
    tail_after = max(comm, current_gate)
    max_saving = max(0.0, tail_before - tail_after)
    return {
        "status": "modeled",
        "formula": "max(P + C, Q + G)",
        "pack_us": pack,
        "communication_us": comm,
        "current_us": current,
        "gate_us": gate,
        "pack_plus_comm_us": pack_comm,
        "current_plus_gate_us": current_gate,
        "tail_before_us": tail_before,
        "tail_after_eliding_pack_us": tail_after,
        "min_saving_us": 0.0,
        "max_saving_us": max_saving,
        "fully_elided_saving_us": max_saving,
        "critical_branch_before": "pack_plus_comm" if pack_comm >= current_gate else "current_plus_gate",
        "communication_source": "device_duration" if comm_node is not None else "explicit",
        "note": "AICPU HcclLaunch is rejected as C; zero is the no-guarantee bound.",
    }


def analyze_subgraph_fusion(
    dag: Mapping[str, Any],
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(dag, Mapping):
        raise SubgraphError("dag_invalid")
    nodes, order = _node_map(dag)
    edges = _edge_list(dag, nodes)
    candidate = dict(candidate or dag.get("candidate_fusion") or {})
    if not isinstance(candidate, Mapping):
        raise SubgraphError("candidate_invalid")
    producer = str(candidate.get("producer", "")).strip() or None
    consumer = str(candidate.get("consumer", "")).strip() or None
    if producer is not None and producer not in nodes:
        raise SubgraphError(f"producer_unknown:{producer}")
    if consumer is not None and consumer not in nodes:
        raise SubgraphError(f"consumer_unknown:{consumer}")
    remove_nodes, removed_node_objects = _resolve_remove_nodes(candidate, nodes)
    remove_edges = _resolve_remove_edges(candidate, edges, producer, consumer, remove_nodes)
    removed_edges_bytes = [_gm_bytes_for_edge(edge) for edge in remove_edges]
    removable_bytes = sum(item["bytes"] or 0.0 for item in removed_edges_bytes)
    missing_bytes = [f"{item['src']}->{item['dst']}" for item in removed_edges_bytes if item["bytes"] is None]
    gates = [
        _collective_gate(candidate, nodes, remove_nodes, remove_edges),
        _buffer_gate(candidate, dag, nodes, remove_nodes),
        _dcp_gate(candidate, removed_node_objects),
        _route_gate(candidate, removed_node_objects),
        _contract_gate(candidate, remove_edges, nodes),
    ]
    failures = [gate["code"] for gate in gates if not gate["feasible"]]
    latency_before = _critical_path(nodes, edges)
    latency_after = _critical_path(
        nodes,
        edges,
        removed=set(remove_nodes),
        producer=producer,
        fused_extra_us=float(candidate.get("fused_extra_us", 0.0) or 0.0),
    )
    savings = None
    if latency_before.get("status") == "modeled" and latency_after.get("status") == "modeled":
        savings = max(0.0, latency_before["critical_path_us"] - latency_after["critical_path_us"])
    overlap = _overlap_analysis(dag, candidate, nodes)
    capacities = candidate.get("capacities_kib", dag.get("capacities_kib"))
    l2_cache_mib = candidate.get("l2_cache_mib", dag.get("l2_cache_mib"))
    working_set_mib = candidate.get("working_set_mib", dag.get("working_set_mib"))
    residency = None
    if _finite(l2_cache_mib) and _finite(working_set_mib):
        residency = float(working_set_mib) <= float(l2_cache_mib)
    status = "modeled"
    if failures:
        status = "infeasible"
    elif missing_bytes:
        status = "unmodeled_gm_bytes"
    elif latency_before.get("status") != "modeled" or latency_after.get("status") != "modeled":
        status = "unmodeled_latency"
    if overlap.get("status") == "invalid_communication_duration":
        status = "invalid_communication_duration"
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "fusion": {
            "name": candidate.get("name", "candidate"),
            "producer": producer,
            "consumer": consumer,
            "remove_nodes": remove_nodes,
            "remove_edges": [{"src": edge["src"], "dst": edge["dst"]} for edge in remove_edges],
        },
        "removable_gm_bytes": removable_bytes if not missing_bytes else None,
        "removable_gm_read_bytes": sum(item["read_bytes"] or 0.0 for item in removed_edges_bytes),
        "removable_gm_write_bytes": sum(item["write_bytes"] or 0.0 for item in removed_edges_bytes),
        "removable_gm_bytes_by_edge": removed_edges_bytes,
        "removed_kernel_launches": len(remove_nodes),
        "l2_cache_caveat": {
            "logical_gm_bytes": removable_bytes if not missing_bytes else None,
            "hbm_bytes_lower_bound": None,
            "hbm_time_lower_bound_us": None,
            "l2_cache_mib": float(l2_cache_mib) if _finite(l2_cache_mib) else None,
            "working_set_mib": float(working_set_mib) if _finite(working_set_mib) else None,
            "l2_cache_residency_possible": residency,
            "note": "Logical GM reads are not HBM traffic; do not divide by HBM bandwidth when L2 can retain the working set.",
        },
        "feasibility": {
            "feasible": not failures,
            "gates": gates,
            "failures": failures,
        },
        "latency": {
            "before": latency_before,
            "after": latency_after,
            "critical_path_saving_us": savings,
            "min_saving_us": 0.0 if savings is not None else None,
            "max_saving_us": savings,
        },
        "overlap": overlap,
        "assumptions": [
            "GM byte counts are logical write+read requests, not HBM transactions",
            "L2 residency is evaluated when a working set is supplied; otherwise it stays unknown",
            "critical-path savings are theoretical and do not replace a matched A/B measurement",
            "AICPU HcclLaunch is not device communication completion time",
            "buffer capacities default to an A5 reference only when the caller omits platform capacities",
        ],
    }
    if capacities is not None:
        payload["buffer_capacities_kib"] = dict(capacities) if isinstance(capacities, Mapping) else capacities
    return payload


__all__ = [
    "SCHEMA_VERSION",
    "SubgraphError",
    "DEFAULT_CAPACITIES_KIB",
    "analyze_subgraph_fusion",
]
