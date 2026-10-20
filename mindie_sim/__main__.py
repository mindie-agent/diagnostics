"""Command-line entry point for ``python -m mindie_sim`` and ``mindie-sim``.

[Role]
  Dispatch the four stateless subcommands ``hardware``, ``pipeline``,
  ``subgraph`` and ``net``; load explicit JSON inputs; print or write the
  structured JSON report.

[Invariants]
  * The CLI never fabricates a missing duration, capacity or provider value.
  * Known validation failures return a structured error with a stable reason
    and a non-zero exit code; they are not printed as successful empty reports.
  * Output is one JSON document; diagnostics go to the same document so a
    caller can distinguish modeled, uncalibrated and infeasible states.

[Timeouts & Locks]
  File reads are bounded by the caller's path.  Provider construction may run
  bounded git probes; no network, service, lock or sleep is started here.

[Side-Effect Policy]
  Reads the explicit input paths and writes only the explicit ``--output`` path.
  Never changes device state, installs packages, or mutates source checkouts.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import __version__
from .hardware import (
    BUILTIN_SKU_CATALOG,
    HardwareError,
    default_link_ceilings,
    hardware_report,
    is_ambiguous_family_token,
    load_npu_smi_snapshot,
    parse_cannsim_link_ceilings,
    parse_npu_smi_topo,
    parse_pcie_link_info,
    parse_platform_ini,
)
from .net import NetError, net_report
from .pipeline import PipelineError, compare_pipeline_variants, solve_periodic_pipeline
from .subgraph import SubgraphError, analyze_subgraph_fusion

ERROR_SCHEMA = "mindie-sim-error/1"


def _load_json(path: str | Path) -> Any:
    source = Path(path).expanduser()
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"input_unreadable:{source}:{type(exc).__name__}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"input_json_invalid:{source}") from exc


def _write(payload: Mapping[str, Any], output: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output:
        destination = Path(output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


def _mhz_range(text: str | None) -> list[float] | None:
    if not text:
        return None
    parts = [part for part in text.replace(",", ":").split(":") if part.strip()]
    try:
        values = sorted(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"sustained_mhz_invalid:{text}") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"sustained_mhz_invalid:{text}")
    if len(values) == 1:
        values = [values[0], values[0]]
    return values[:2]


def _find_platform_ini(soc: str) -> Path:
    direct = Path(soc).expanduser()
    if direct.is_file():
        return direct
    env_dir = os.environ.get("MINDIE_SIM_PLATFORM_CONFIG_DIR")
    candidates: list[str] = []
    if env_dir:
        candidates.append(str(Path(env_dir).expanduser() / f"{soc}.ini"))
    for pattern in (
        f"/usr/local/Ascend/*/*/data/platform_config/{soc}.ini",
        f"/usr/local/Ascend/*/data/platform_config/{soc}.ini",
        f"/usr/local/Ascend/*/*/*/data/platform_config/{soc}.ini",
    ):
        candidates.extend(sorted(glob.glob(pattern)))
    for candidate in candidates:
        if Path(candidate).is_file():
            return Path(candidate)
    raise HardwareError(f"platform_ini_not_found_for_soc:{soc}")


def _profile_merge(dag: Mapping[str, Any], report: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    merged = dict(dag)
    nodes = [dict(node) for node in dag.get("nodes", ())]
    by_id = {str(node.get("id")): node for node in nodes}
    applied: list[str] = []
    unmatched: list[str] = []
    mapping = report.get("node_durations_us")
    if isinstance(mapping, Mapping):
        for node_id, duration in mapping.items():
            node = by_id.get(str(node_id))
            if node is None:
                unmatched.append(str(node_id))
                continue
            if node.get("duration_us") is None and isinstance(duration, (int, float)):
                node["duration_us"] = float(duration)
                applied.append(str(node_id))
    for kernel in report.get("kernels", ()) or ():
        if not isinstance(kernel, Mapping):
            continue
        node_id = kernel.get("node", kernel.get("id", kernel.get("name")))
        node = by_id.get(str(node_id))
        if node is None:
            unmatched.append(str(node_id))
            continue
        duration = kernel.get("duration_us")
        if duration is None and kernel.get("duration_ms") is not None:
            duration = float(kernel["duration_ms"]) * 1000.0
        if node.get("duration_us") is None and isinstance(duration, (int, float)):
            node["duration_us"] = float(duration)
            applied.append(str(node_id))
    merged["nodes"] = nodes
    return merged, {"applied_nodes": applied, "unmatched_nodes": unmatched}


def _hardware(args: argparse.Namespace) -> dict[str, Any]:
    ini: Path | None = None
    if args.ini:
        ini = Path(args.ini).expanduser()
    elif args.soc and not is_ambiguous_family_token(args.soc):
        try:
            ini = _find_platform_ini(args.soc)
        except HardwareError:
            if args.soc not in BUILTIN_SKU_CATALOG and not args.sku_profile:
                raise
    if ini is None and not args.soc and not args.sku_profile:
        raise HardwareError("platform_ini_or_soc_required")
    config = parse_platform_ini(ini) if ini is not None else None
    snapshot = load_npu_smi_snapshot(args.npu_smi_file) if args.npu_smi_file else None
    topo_snapshot = parse_npu_smi_topo(args.topo_file) if args.topo_file else None
    pcie_snapshot = parse_pcie_link_info(args.pcie_link_info) if args.pcie_link_info else None
    cannsim = parse_cannsim_link_ceilings(args.cannsim_constants) if args.cannsim_constants else default_link_ceilings()
    workload = _load_json(args.workload) if args.workload else None
    buffer_plan = _load_json(args.buffers) if args.buffers else None
    transfer_plan = _load_json(args.transfers) if args.transfers else None
    simulator_ub = None if args.simulator_ub_kib is None or args.simulator_ub_kib <= 0 else float(args.simulator_ub_kib)
    payload = hardware_report(
        config,
        soc=args.soc,
        sku_profile=args.sku_profile,
        snapshot=snapshot,
        topo_snapshot=topo_snapshot,
        pcie_snapshot=pcie_snapshot,
        verified_runtime_hbm_mib=args.verified_runtime_hbm_mib,
        simulator_ub_kib=simulator_ub,
        allocatable_ub_kib=args.allocatable_ub_kib,
        sustained_mhz_range=_mhz_range(args.sustained_mhz) or (1250.0, 1300.0),
        buffer_plan=buffer_plan,
        workload=workload,
        transfer_plan=transfer_plan,
        cannsim_links=cannsim,
    )
    payload["command"] = "hardware"
    return payload


def _pipeline(args: argparse.Namespace) -> dict[str, Any]:
    graph = _load_json(args.graph)
    durations = _load_json(args.durations) if args.durations else None
    if args.compare_variant:
        payload = compare_pipeline_variants(
            graph,
            _load_json(args.compare_variant),
            durations,
            target_us=args.target_us,
            total_tiles=args.total_tiles,
            reference_mhz=args.reference_mhz,
        )
    else:
        payload = solve_periodic_pipeline(
            graph,
            durations,
            target_us=args.target_us,
            total_tiles=args.total_tiles,
            reference_mhz=args.reference_mhz,
            run_self_check=True,
        )
    payload["command"] = "pipeline"
    return payload


def _subgraph(args: argparse.Namespace) -> dict[str, Any]:
    dag = _load_json(args.dag)
    merge_status = None
    if args.profile_report:
        dag, merge_status = _profile_merge(dag, _load_json(args.profile_report))
    candidate = _load_json(args.candidate_fusion) if args.candidate_fusion else None
    payload = analyze_subgraph_fusion(dag, candidate)
    payload["command"] = "subgraph"
    if merge_status is not None:
        payload["profile_merge"] = merge_status
    return payload


def _net(args: argparse.Namespace) -> dict[str, Any]:
    model_spec = _load_json(args.model_spec)
    if not isinstance(model_spec, dict):
        raise NetError("model_spec_invalid")
    if args.parallel:
        parallel = _load_json(args.parallel)
        if not isinstance(parallel, Mapping):
            raise NetError("parallel_invalid")
        model_spec = {**model_spec, "parallel": dict(parallel)}
    payload = net_report(
        model_spec,
        model_spec.get("parallel"),
        measured_device_ms=args.measured_ms,
        provider_source=args.provider_source,
        device_name=args.device,
        expected_commit=args.expected_commit,
    )
    payload["command"] = "net"
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mindie-sim",
        description="Ascend NPU four-layer simulation and boundary solver.",
    )
    parser.add_argument("--version", action="version", version=f"mindie-sim {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hardware = subparsers.add_parser("hardware", help="Layer 1 hardware, buffers and thermal bounds")
    hardware.add_argument("--soc", default=None, help="SoC/platform name or direct .ini path")
    hardware.add_argument("--ini", default=None, help="explicit CANN platform .ini")
    hardware.add_argument("--sku-profile", default=None, help="external JSON SKU capability profile")
    hardware.add_argument("--npu-smi-file", default=None, help="npu-smi snapshot file or directory")
    hardware.add_argument("--topo-file", default=None, help="npu-smi info -t topo file")
    hardware.add_argument("--pcie-link-info", default=None, help="lspci LnkCap/LnkSta file or text")
    hardware.add_argument("--cannsim-constants", default=None, help="explicit cannsim link constants")
    hardware.add_argument("--buffers", default=None, help="buffer allocation plan JSON")
    hardware.add_argument("--workload", default=None, help="workload/thermal JSON")
    hardware.add_argument("--transfers", default=None, help="D2D/H2D/D2H/RH2D/RD2H transfer & collective plan JSON")
    hardware.add_argument("--sustained-mhz", default="1250:1300", help="sustained thermal range, e.g. 1250:1300")
    hardware.add_argument("--simulator-ub-kib", type=float, default=256.0, help="simulator UB reference in KiB; 0 disables")
    hardware.add_argument("--allocatable-ub-kib", type=float, default=None, help="explicit allocatable UB capacity in KiB")
    hardware.add_argument("--verified-runtime-hbm-mib", type=float, default=None, help="explicit verified runtime HBM capacity")
    hardware.add_argument("--output", default=None)
    hardware.set_defaults(handler=_hardware)

    pipeline = subparsers.add_parser("pipeline", help="Layer 2 periodic microstage pipeline solver")
    pipeline.add_argument("--graph", required=True, help="microstage graph JSON")
    pipeline.add_argument("--durations", default=None, help="calibrated stage durations JSON")
    pipeline.add_argument("--compare-variant", default=None, help="alternative graph JSON")
    pipeline.add_argument("--self-check", action="store_true", help="run topological and relaxation self-checks")
    pipeline.add_argument("--target-us", type=float, default=None)
    pipeline.add_argument("--total-tiles", type=int, default=None)
    pipeline.add_argument("--reference-mhz", type=float, default=None)
    pipeline.add_argument("--output", default=None)
    pipeline.set_defaults(handler=_pipeline)

    subgraph = subparsers.add_parser("subgraph", help="Layer 3 subgraph fusion analyzer")
    subgraph.add_argument("--dag", required=True, help="producer-consumer DAG JSON")
    subgraph.add_argument("--candidate-fusion", default=None, help="candidate fusion JSON")
    subgraph.add_argument("--profile-report", default=None, help="optional measured node durations JSON")
    subgraph.add_argument("--output", default=None)
    subgraph.set_defaults(handler=_subgraph)

    net = subparsers.add_parser("net", help="Layer 4 parallel and whole-network decomposition")
    net.add_argument("--model-spec", required=True, help="model/step spec JSON")
    net.add_argument("--parallel", default=None, help="explicit DP/TP/EP/PP/DCP JSON")
    net.add_argument("--provider-source", default=None, help="pinned msmodeling checkout")
    net.add_argument("--device", default=None, help="explicit upstream DeviceProfile name")
    net.add_argument("--expected-commit", default=None)
    net.add_argument("--measured-ms", type=float, default=None)
    net.add_argument("--output", default=None)
    net.set_defaults(handler=_net)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = args.handler(args)
    except (HardwareError, PipelineError, SubgraphError, NetError, ValueError, OSError, TypeError) as exc:
        reason = getattr(exc, "reason", None) or str(exc)
        _write({
            "schema_version": ERROR_SCHEMA,
            "status": "error",
            "command": getattr(args, "command", None),
            "reason": reason,
        }, getattr(args, "output", None))
        return 1
    _write(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
