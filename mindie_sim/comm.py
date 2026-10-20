"""Full-path interconnect and host-device transfer modeling (D2D / H2D / D2H / RH2D / RD2H).

[Role]
  Parse ``npu-smi info -t topo`` interconnect matrices and ``lspci`` PCIe link
  negotiation status (``LnkCap`` vs ``LnkSta``); represent the 7 transfer
  channels (``d2d_intra_package``, ``d2d_intra_node``, ``d2d_inter_node``,
  ``h2d``, ``d2h``, ``rh2d``, ``rd2h``); and solve point-to-point, host-staged
  and hierarchical collective communication bounds.

[Invariants]
  * Theoretical link peaks (e.g. PCIe Gen5 x16 ``LnkCap 32GT/s``) and runtime
    negotiated/calibrated bandwidths (e.g. ``LnkSta 16GT/s (downgraded)``) are
    kept strictly separate; a PCIe downgrade produces an explicit diagnostic.
  * ``h2d`` and ``d2h`` distinguish ``pinned`` (DMA direct) from ``pageable``
    (extra Host DRAM staging copy ceiling) and account for full-duplex or
    shared Root-Complex / NUMA contention.
  * ``rh2d`` and ``rd2h`` distinguish ``direct_rdma`` (RNIC <-> RoCE <-> NPU HBM)
    from ``host_staged`` (Network <-> Local Host DRAM <-> Local PCIe H2D/D2H).
    When ``transport_mode`` is unspecified, both bounds are reported with
    ``transport_mode_unverified``.

[Timeouts & Locks]
  Pure deterministic parsing and arithmetic.  No locks, sleeps, or subprocesses.

[Side-Effect Policy]
  Reads only explicit paths or strings supplied by the caller.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hardware_common import HardwareError, _finite, _float_value

TRANSFER_CHANNELS = (
    "d2d_intra_package",
    "d2d_intra_node",
    "d2d_inter_node",
    "h2d",
    "d2h",
    "rh2d",
    "rd2h",
)

# Raw per-lane unidirectional GB/s by PCIe GT/s (including 128b/130b encoding).
_PCIE_PER_LANE_RAW_GBPS: dict[float, float] = {
    2.5: 0.25,
    5.0: 0.50,
    8.0: 0.9846,
    16.0: 1.9692,
    32.0: 3.9385,
    64.0: 7.56,
}
_PCIE_PAYLOAD_EFFICIENCY = 0.78


@dataclass(frozen=True)
class TransferChannelSpec:
    channel: str
    protocol: str
    uni_peak_bw_gbps: float | None
    bi_peak_bw_gbps: float | None
    calibrated_uni_bw_gbps: float | None = None
    calibrated_bi_bw_gbps: float | None = None
    pageable_uni_bw_gbps: float | None = None
    alpha_us: float = 1.5
    transport_mode: str = "direct"
    notes: str | None = None

    @property
    def effective_uni_bw_gbps(self) -> float | None:
        return self.calibrated_uni_bw_gbps if self.calibrated_uni_bw_gbps is not None else self.uni_peak_bw_gbps

    @property
    def effective_bi_bw_gbps(self) -> float | None:
        return self.calibrated_bi_bw_gbps if self.calibrated_bi_bw_gbps is not None else self.bi_peak_bw_gbps

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "protocol": self.protocol,
            "uni_peak_bw_gbps": self.uni_peak_bw_gbps,
            "bi_peak_bw_gbps": self.bi_peak_bw_gbps,
            "calibrated_uni_bw_gbps": self.calibrated_uni_bw_gbps,
            "calibrated_bi_bw_gbps": self.calibrated_bi_bw_gbps,
            "effective_uni_bw_gbps": self.effective_uni_bw_gbps,
            "effective_bi_bw_gbps": self.effective_bi_bw_gbps,
            "pageable_uni_bw_gbps": self.pageable_uni_bw_gbps,
            "alpha_us": self.alpha_us,
            "transport_mode": self.transport_mode,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class NpuTopologySnapshot:
    endpoints: tuple[str, ...]
    matrix: Mapping[str, Mapping[str, str]]
    link_type_counts: Mapping[str, int]
    cpu_affinity: Mapping[str, str] = field(default_factory=dict)
    numa_groups: tuple[tuple[str, ...], ...] = ()
    source: str = "<topo>"

    @property
    def has_sio_intra_package(self) -> bool:
        return self.link_type_counts.get("SIO", 0) > 0

    @property
    def intra_node_fabric(self) -> str:
        for fabric in ("HCCS_SW", "HCCS", "PIX", "PXB", "PHB", "SYS"):
            if self.link_type_counts.get(fabric, 0) > 0:
                return fabric
        return "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "endpoints": list(self.endpoints),
            "has_sio_intra_package": self.has_sio_intra_package,
            "intra_node_fabric": self.intra_node_fabric,
            "link_type_counts": dict(sorted(self.link_type_counts.items())),
            "cpu_affinity": dict(sorted(self.cpu_affinity.items())),
            "numa_groups": [list(group) for group in self.numa_groups],
        }


@dataclass(frozen=True)
class PcieLinkSnapshot:
    cap_speed_gtps: float
    cap_width: int
    sta_speed_gtps: float
    sta_width: int
    downgraded: bool
    cap_raw_uni_bw_gbps: float
    sta_raw_uni_bw_gbps: float
    effective_pinned_uni_bw_gbps: float
    source: str = "<lspci>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "cap_speed_gtps": self.cap_speed_gtps,
            "cap_width": self.cap_width,
            "sta_speed_gtps": self.sta_speed_gtps,
            "sta_width": self.sta_width,
            "downgraded": self.downgraded,
            "cap_raw_uni_bw_gbps": round(self.cap_raw_uni_bw_gbps, 3),
            "sta_raw_uni_bw_gbps": round(self.sta_raw_uni_bw_gbps, 3),
            "effective_pinned_uni_bw_gbps": round(self.effective_pinned_uni_bw_gbps, 3),
        }


_HEADER_TOKEN_RE = re.compile(r"(NPU\d+|Phy-ID\d+)")
_LNK_RE = re.compile(r"Lnk(Cap|Sta):.*?Speed\s+([0-9.]+)GT/s(\s*\(downgraded\))?,\s*Width\s+x(\d+)", re.IGNORECASE)


def parse_npu_smi_topo(path_or_text: str | Path, *, source: str = "<npu-smi-topo>") -> NpuTopologySnapshot:
    candidate = Path(path_or_text) if isinstance(path_or_text, (str, Path)) and "\n" not in str(path_or_text) else None
    if candidate is not None and candidate.exists() and candidate.is_file():
        text = candidate.read_text(encoding="utf-8-sig", errors="replace")
        source = str(candidate)
    else:
        text = str(path_or_text)
    endpoints: list[str] = []
    matrix: dict[str, dict[str, str]] = {}
    counts: dict[str, int] = {}
    affinity: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("Legend:"):
            if stripped.startswith("Legend:"):
                break
            continue
        if not endpoints:
            found = _HEADER_TOKEN_RE.findall(stripped)
            if len(found) >= 2:
                endpoints = found
            continue
        parts = stripped.split()
        if not parts or parts[0] not in endpoints:
            continue
        row_id = parts[0]
        cells = parts[1 : 1 + len(endpoints)]
        if len(cells) < len(endpoints):
            continue
        row_map: dict[str, str] = {}
        for col_id, link in zip(endpoints, cells):
            row_map[col_id] = link
            if row_id != col_id and link != "X":
                counts[link] = counts.get(link, 0) + 1
        matrix[row_id] = row_map
        if len(parts) > 1 + len(endpoints):
            affinity[row_id] = " ".join(parts[1 + len(endpoints) :])
    if not endpoints or not matrix:
        raise HardwareError("npu_smi_topo_empty")
    by_affinity: dict[str, list[str]] = {}
    for ep, cpus in affinity.items():
        by_affinity.setdefault(cpus, []).append(ep)
    numa_groups = tuple(tuple(group) for group in by_affinity.values() if len(group) > 1)
    return NpuTopologySnapshot(
        endpoints=tuple(endpoints),
        matrix=matrix,
        link_type_counts=counts,
        cpu_affinity=affinity,
        numa_groups=numa_groups,
        source=source,
    )


def parse_pcie_link_info(path_or_text: str | Path, *, source: str = "<lspci>") -> PcieLinkSnapshot:
    candidate = Path(path_or_text) if isinstance(path_or_text, (str, Path)) and "\n" not in str(path_or_text) else None
    if candidate is not None and candidate.exists() and candidate.is_file():
        text = candidate.read_text(encoding="utf-8-sig", errors="replace")
        source = str(candidate)
    else:
        text = str(path_or_text)
    cap: tuple[float, int] | None = None
    sta: tuple[float, int, bool] | None = None
    for match in _LNK_RE.finditer(text):
        kind = match.group(1).lower()
        speed = float(match.group(2))
        downgraded_flag = bool(match.group(3))
        width = int(match.group(4))
        if kind == "cap":
            cap = (speed, width)
        elif kind == "sta":
            sta = (speed, width, downgraded_flag)
    if sta is None and cap is None:
        raise HardwareError("pcie_link_info_missing")
    cap_speed, cap_width = cap if cap is not None else (sta[0], sta[1])  # type: ignore[index]
    sta_speed, sta_width, explicit_down = sta if sta is not None else (cap_speed, cap_width, False)
    downgraded = explicit_down or (sta_speed < cap_speed) or (sta_width < cap_width)
    cap_raw = _PCIE_PER_LANE_RAW_GBPS.get(cap_speed, cap_speed * 0.12) * cap_width
    sta_raw = _PCIE_PER_LANE_RAW_GBPS.get(sta_speed, sta_speed * 0.12) * sta_width
    eff = sta_raw * _PCIE_PAYLOAD_EFFICIENCY
    return PcieLinkSnapshot(
        cap_speed_gtps=cap_speed,
        cap_width=cap_width,
        sta_speed_gtps=sta_speed,
        sta_width=sta_width,
        downgraded=downgraded,
        cap_raw_uni_bw_gbps=cap_raw,
        sta_raw_uni_bw_gbps=sta_raw,
        effective_pinned_uni_bw_gbps=eff,
        source=source,
    )


def default_transfer_channels(
    arch_family: str,
    *,
    dies_per_package: int = 1,
    intra_package_die_bw_gbps: float | None = None,
    intra_node_hccs_bw_gbps: float | None = None,
    inter_node_roce_bw_gbps: float | None = None,
    pcie_snapshot: PcieLinkSnapshot | None = None,
    topo_snapshot: NpuTopologySnapshot | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, TransferChannelSpec]:
    fam = arch_family.upper()
    sio_bw = intra_package_die_bw_gbps if intra_package_die_bw_gbps is not None else (
        190.0 if (dies_per_package > 1 or (topo_snapshot and topo_snapshot.has_sio_intra_package)) else None
    )
    hccs_uni = intra_node_hccs_bw_gbps if intra_node_hccs_bw_gbps is not None else (
        392.0 if fam == "A5" else 196.0 if fam == "A3" else 168.0
    )
    roce_uni = inter_node_roce_bw_gbps if inter_node_roce_bw_gbps is not None else (
        50.0 if fam == "A5" else 25.0
    )
    pcie_peak_uni = 50.0 if fam in {"A2", "A3"} else 64.0
    pcie_cal_uni = round(pcie_snapshot.effective_pinned_uni_bw_gbps, 3) if pcie_snapshot is not None else None
    pcie_cal_bi = round(pcie_cal_uni * 1.75, 3) if pcie_cal_uni is not None else None
    pageable_uni = 14.0 if fam in {"A2", "A3"} else 20.0
    intra_proto = topo_snapshot.intra_node_fabric if (topo_snapshot and topo_snapshot.intra_node_fabric != "UNKNOWN") else (
        "UB_MESH" if fam == "A5" else "HCCS_SW" if fam == "A3" else "HCCS"
    )
    channels: dict[str, TransferChannelSpec] = {
        "d2d_intra_package": TransferChannelSpec(
            channel="d2d_intra_package",
            protocol="SIO" if sio_bw is not None else "NONE_SINGLE_DIE",
            uni_peak_bw_gbps=sio_bw,
            bi_peak_bw_gbps=(sio_bw * 1.85) if sio_bw is not None else None,
            alpha_us=0.8 if sio_bw is not None else 0.0,
            transport_mode="sio_die_to_die" if sio_bw is not None else "unavailable_single_die",
        ),
        "d2d_intra_node": TransferChannelSpec(
            channel="d2d_intra_node",
            protocol=intra_proto,
            uni_peak_bw_gbps=hccs_uni,
            bi_peak_bw_gbps=round(hccs_uni * 1.85, 2),
            alpha_us=1.8 if fam == "A5" else 2.5,
            transport_mode="direct_fabric",
        ),
        "d2d_inter_node": TransferChannelSpec(
            channel="d2d_inter_node",
            protocol="RoCEv2_400G" if roce_uni >= 45.0 else "RoCEv2_200G",
            uni_peak_bw_gbps=roce_uni,
            bi_peak_bw_gbps=round(roce_uni * 1.8, 2),
            alpha_us=5.0,
            transport_mode="roce_rdma",
        ),
        "h2d": TransferChannelSpec(
            channel="h2d",
            protocol="PCIe_Downgraded" if (pcie_snapshot and pcie_snapshot.downgraded) else "PCIe",
            uni_peak_bw_gbps=pcie_peak_uni,
            bi_peak_bw_gbps=round(pcie_peak_uni * 1.75, 2),
            calibrated_uni_bw_gbps=pcie_cal_uni,
            calibrated_bi_bw_gbps=pcie_cal_bi,
            pageable_uni_bw_gbps=pageable_uni,
            alpha_us=2.0,
            transport_mode="pcie_dma",
        ),
        "d2h": TransferChannelSpec(
            channel="d2h",
            protocol="PCIe_Downgraded" if (pcie_snapshot and pcie_snapshot.downgraded) else "PCIe",
            uni_peak_bw_gbps=pcie_peak_uni,
            bi_peak_bw_gbps=round(pcie_peak_uni * 1.75, 2),
            calibrated_uni_bw_gbps=pcie_cal_uni,
            calibrated_bi_bw_gbps=pcie_cal_bi,
            pageable_uni_bw_gbps=pageable_uni,
            alpha_us=2.0,
            transport_mode="pcie_dma",
        ),
        "rh2d": TransferChannelSpec(
            channel="rh2d",
            protocol="RemoteHost_to_NPU",
            uni_peak_bw_gbps=min(roce_uni, pcie_cal_uni or pcie_peak_uni),
            bi_peak_bw_gbps=round(min(roce_uni, pcie_cal_uni or pcie_peak_uni) * 1.6, 2),
            alpha_us=6.5,
            transport_mode="auto",
            notes="Supports direct_rdma (RNIC->RoCE->NPU) or host_staged (Network->LocalHost->PCIe H2D)",
        ),
        "rd2h": TransferChannelSpec(
            channel="rd2h",
            protocol="NPU_to_RemoteHost",
            uni_peak_bw_gbps=min(roce_uni, pcie_cal_uni or pcie_peak_uni),
            bi_peak_bw_gbps=round(min(roce_uni, pcie_cal_uni or pcie_peak_uni) * 1.6, 2),
            alpha_us=6.5,
            transport_mode="auto",
            notes="Supports direct_rdma (NPU->RoCE->RNIC) or host_staged (PCIe D2H->LocalHost->Network)",
        ),
    }
    if isinstance(overrides, Mapping):
        for name, raw in overrides.items():
            if name not in channels or not isinstance(raw, Mapping):
                continue
            base = channels[name]
            channels[name] = TransferChannelSpec(
                channel=name,
                protocol=str(raw.get("protocol", base.protocol)),
                uni_peak_bw_gbps=float(raw["uni_peak_bw_gbps"]) if raw.get("uni_peak_bw_gbps") is not None else base.uni_peak_bw_gbps,
                bi_peak_bw_gbps=float(raw["bi_peak_bw_gbps"]) if raw.get("bi_peak_bw_gbps") is not None else base.bi_peak_bw_gbps,
                calibrated_uni_bw_gbps=float(raw["calibrated_uni_bw_gbps"]) if raw.get("calibrated_uni_bw_gbps") is not None else base.calibrated_uni_bw_gbps,
                calibrated_bi_bw_gbps=float(raw["calibrated_bi_bw_gbps"]) if raw.get("calibrated_bi_bw_gbps") is not None else base.calibrated_bi_bw_gbps,
                pageable_uni_bw_gbps=float(raw["pageable_uni_bw_gbps"]) if raw.get("pageable_uni_bw_gbps") is not None else base.pageable_uni_bw_gbps,
                alpha_us=float(raw.get("alpha_us", base.alpha_us)),
                transport_mode=str(raw.get("transport_mode", base.transport_mode)),
                notes=raw.get("notes", base.notes),
            )
    return channels


def solve_transfer_bound(
    channel: str,
    size_bytes: float,
    channels: Mapping[str, TransferChannelSpec],
    *,
    memory_type: str = "pinned",
    transport_mode: str | None = None,
    full_duplex: bool = False,
    shared_root_complex_npus: int = 1,
    pipeline_chunks: int = 4,
) -> dict[str, Any]:
    key = channel.strip().lower()
    if key not in channels:
        raise HardwareError(f"transfer_channel_unknown:{channel}")
    bytes_val = _float_value(size_bytes, "transfer_bytes_invalid")
    spec = channels[key]
    contention = max(1, int(shared_root_complex_npus))
    chunks = max(1, int(pipeline_chunks))

    if key in {"rh2d", "rd2h"}:
        net_spec = channels.get("d2d_inter_node", spec)
        local_dma_spec = channels.get("h2d" if key == "rh2d" else "d2h", spec)
        net_bw = (net_spec.effective_bi_bw_gbps / 2.0 if full_duplex and net_spec.effective_bi_bw_gbps else net_spec.effective_uni_bw_gbps) or 25.0
        local_bw_base = (
            local_dma_spec.pageable_uni_bw_gbps
            if memory_type == "pageable" and local_dma_spec.pageable_uni_bw_gbps
            else (local_dma_spec.effective_bi_bw_gbps / 2.0 if full_duplex and local_dma_spec.effective_bi_bw_gbps else local_dma_spec.effective_uni_bw_gbps)
        ) or 24.0
        local_bw = local_bw_base / contention
        direct_bw = min(net_bw, (local_dma_spec.effective_uni_bw_gbps or net_bw) / contention)
        direct_us = spec.alpha_us + (bytes_val / (direct_bw * 1e3))
        staged_alpha = net_spec.alpha_us + local_dma_spec.alpha_us
        fast_bw = max(net_bw, local_bw)
        slow_bw = min(net_bw, local_bw)
        staged_us = staged_alpha + (bytes_val / (fast_bw * 1e3)) / chunks + (bytes_val / (slow_bw * 1e3))
        mode = (transport_mode or spec.transport_mode or "auto").strip().lower()
        flags: list[str] = []
        if mode == "direct_rdma":
            selected_us: float | None = direct_us
        elif mode == "host_staged":
            selected_us = staged_us
        else:
            selected_us = None
            flags.append("transport_mode_unverified")
        return {
            "channel": key,
            "size_bytes": bytes_val,
            "transport_mode": mode,
            "memory_type": memory_type,
            "direct_rdma_lower_bound_us": direct_us,
            "host_staged_lower_bound_us": staged_us,
            "lower_bound_us": selected_us,
            "lower_bound_interval_us": [direct_us, staged_us],
            "effective_direct_bw_gbps": direct_bw,
            "effective_staged_bottleneck_bw_gbps": slow_bw,
            "pipeline_chunks": chunks,
            "shared_root_complex_npus": contention,
            "flags": flags,
        }

    base_bw = spec.effective_bi_bw_gbps / 2.0 if (full_duplex and spec.effective_bi_bw_gbps) else spec.effective_uni_bw_gbps
    if base_bw is None or base_bw <= 0:
        return {
            "channel": key,
            "status": "unavailable_on_sku",
            "size_bytes": bytes_val,
            "lower_bound_us": None,
        }
    flags = []
    if key in {"h2d", "d2h"} and memory_type == "pageable" and spec.pageable_uni_bw_gbps:
        base_bw = min(base_bw, spec.pageable_uni_bw_gbps)
        flags.append("pageable_host_memory_ceiling_applied")
    if key in {"h2d", "d2h"} and contention > 1:
        base_bw = base_bw / contention
        flags.append("shared_pcie_root_complex_contention")
    duration_us = spec.alpha_us + (bytes_val / (base_bw * 1e3))
    return {
        "channel": key,
        "status": "modeled",
        "size_bytes": bytes_val,
        "memory_type": memory_type,
        "full_duplex": full_duplex,
        "effective_bw_gbps": base_bw,
        "alpha_us": spec.alpha_us,
        "lower_bound_us": duration_us,
        "flags": flags,
    }


def solve_collective_bound(
    collective: str,
    size_bytes: float,
    channels: Mapping[str, TransferChannelSpec],
    *,
    total_ranks: int,
    ranks_per_node: int = 8,
    dies_per_package: int = 1,
    algorithm: str = "hierarchical",
) -> dict[str, Any]:
    op = collective.strip().lower().replace("-", "_")
    bytes_val = _float_value(size_bytes, "collective_bytes_invalid")
    n_total = max(1, int(total_ranks))
    if n_total == 1:
        return {"collective": op, "total_ranks": 1, "lower_bound_us": 0.0, "tiers": []}
    n_pkg = min(n_total, max(1, int(dies_per_package)))
    n_node = min(n_total, max(n_pkg, int(ranks_per_node)))
    n_inter = max(1, math.ceil(n_total / n_node))
    n_intra_hops = max(1, n_node // n_pkg)

    def factor(deg: int) -> float:
        if deg <= 1:
            return 0.0
        if op == "all_reduce":
            return 2.0 * (deg - 1) / deg
        if op in {"all_gather", "reduce_scatter", "all_to_all"}:
            return (deg - 1) / deg
        return 1.0

    tiers: list[dict[str, Any]] = []
    total_us = 0.0
    if algorithm == "hierarchical" and n_pkg > 1 and channels["d2d_intra_package"].effective_uni_bw_gbps:
        vol = bytes_val * factor(n_pkg)
        step = solve_transfer_bound("d2d_intra_package", vol, channels, full_duplex=True)
        tiers.append({"tier": "d2d_intra_package", "degree": n_pkg, "volume_bytes": vol, "duration_us": step["lower_bound_us"]})
        total_us += float(step["lower_bound_us"] or 0.0)
    if n_intra_hops > 1:
        vol = bytes_val * (factor(n_intra_hops) if algorithm == "hierarchical" else factor(n_node))
        step = solve_transfer_bound("d2d_intra_node", vol, channels, full_duplex=True)
        tiers.append({"tier": "d2d_intra_node", "degree": n_intra_hops, "volume_bytes": vol, "duration_us": step["lower_bound_us"]})
        total_us += float(step["lower_bound_us"] or 0.0)
    if n_inter > 1:
        vol = (bytes_val / n_node if op == "all_to_all" else bytes_val / max(1, n_pkg)) * factor(n_inter)
        step = solve_transfer_bound("d2d_inter_node", vol, channels, full_duplex=True)
        tiers.append({"tier": "d2d_inter_node", "degree": n_inter, "volume_bytes": vol, "duration_us": step["lower_bound_us"]})
        total_us += float(step["lower_bound_us"] or 0.0)
    return {
        "collective": op,
        "algorithm": algorithm,
        "size_bytes": bytes_val,
        "total_ranks": n_total,
        "ranks_per_node": n_node,
        "dies_per_package": n_pkg,
        "lower_bound_us": total_us,
        "tiers": tiers,
    }


def evaluate_transfer_plan(
    plan: Mapping[str, Any],
    channels: Mapping[str, TransferChannelSpec],
    *,
    dies_per_package: int = 1,
) -> dict[str, Any]:
    transfers_raw = plan.get("transfers", ())
    collectives_raw = plan.get("collectives", ())
    transfers = [
        solve_transfer_bound(
            str(item["channel"]),
            float(item.get("size_bytes", item.get("bytes", 0.0))),
            channels,
            memory_type=str(item.get("memory_type", "pinned")),
            transport_mode=item.get("transport_mode"),
            full_duplex=bool(item.get("full_duplex", False)),
            shared_root_complex_npus=int(item.get("shared_root_complex_npus", 1)),
            pipeline_chunks=int(item.get("pipeline_chunks", 4)),
        )
        for item in transfers_raw
        if isinstance(item, Mapping) and item.get("channel")
    ]
    collectives = [
        solve_collective_bound(
            str(item["collective"]),
            float(item.get("size_bytes", item.get("bytes", 0.0))),
            channels,
            total_ranks=int(item.get("total_ranks", 8)),
            ranks_per_node=int(item.get("ranks_per_node", 8)),
            dies_per_package=int(item.get("dies_per_package", dies_per_package)),
            algorithm=str(item.get("algorithm", "hierarchical")),
        )
        for item in collectives_raw
        if isinstance(item, Mapping) and item.get("collective")
    ]
    return {"transfers": transfers, "collectives": collectives}


__all__ = [
    "TRANSFER_CHANNELS",
    "TransferChannelSpec",
    "NpuTopologySnapshot",
    "PcieLinkSnapshot",
    "parse_npu_smi_topo",
    "parse_pcie_link_info",
    "default_transfer_channels",
    "solve_transfer_bound",
    "solve_collective_bound",
    "evaluate_transfer_plan",
]
