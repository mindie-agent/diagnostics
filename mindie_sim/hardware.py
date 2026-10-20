"""Layer 1 hardware, on-chip buffer, and thermal-bound analyzer.

[Role]
  Parse CANN platform INI files, cannsim link ceilings, and npu-smi snapshots;
  keep platform metadata strictly separate from verified runtime capacity; audit
  L1/L0A/L0B/L0C/UB layouts, alignment, ping-pong slots and region overlap; and
  compute reference lower bounds from thermal clock ranges and PMU cycles.

[Invariants]
  * ``platform_config_hbm_mib`` and ``simulator_ub_kib`` are metadata.  They are
    never copied into ``verified_runtime_hbm_mib`` or ``allocatable_ub_kib``.
  * A runtime capacity is reported only from npu-smi evidence; missing evidence
    yields ``verified_runtime_hbm_mib=None`` and ``runtime_unverified``.
  * Buffer sizes/offsets are rounded only by the requested 32B/256B alignment;
    overlap, overflow and misalignment are diagnostics, never silently fixed.
  * Thermal reference times use caller-supplied peaks/frequencies and are lower
    bounds, not measured sustained performance.  Short-burst clocks never
    replace sustained clocks.
  * The (W-C) sensitivity table is reproducible arithmetic, not a promise that
    the gap is removable.

[Timeouts & Locks]
  Pure parsing/computation.  No locks, sleeps, subprocesses, network, or caches.

[Side-Effect Policy]
  Reads only explicit paths supplied by the caller.  Never writes files, changes
  clocks, changes device state, or invokes npu-smi itself.
"""
from __future__ import annotations

import configparser
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .buffers import BufferAllocation, check_mx_scale_layout, validate_buffer_allocations
from .comm import (
    TRANSFER_CHANNELS,
    NpuTopologySnapshot,
    PcieLinkSnapshot,
    TransferChannelSpec,
    default_transfer_channels,
    evaluate_transfer_plan,
    parse_npu_smi_topo,
    parse_pcie_link_info,
    solve_collective_bound,
    solve_transfer_bound,
)
from .hardware_common import (
    A5_CANN_SIM_LINK_CEILINGS_BPC,
    A5_REFERENCE_PEAKS,
    DEFAULT_ALIGNMENT_BYTES,
    PING_PONG_ALIGNMENT_BYTES,
    SHORT_BURST_MHZ_RANGE,
    SUSTAINED_MHZ_RANGE,
    HardwareError,
    _canonical_link_key,
    _float_value,
    _int_value,
    _kib,
    _mib,
    align_up,
    default_link_ceilings,
)
from .sku_profile import (
    AMBIGUOUS_SKU_REASON,
    BUILTIN_SKU_CATALOG,
    ClockAndTopologySpec,
    CubeSpec,
    DeviceCapabilityProfile,
    MemoryHierarchySpec,
    VectorSpec,
    infer_arch_family,
    is_ambiguous_family_token,
    load_sku_profile,
    resolve_device_capability_profile,
)
from .thermal import (
    cube_reference_cycles,
    cycles_to_us,
    link_reference_bounds,
    reference_time_us,
    sensitivity_matrix,
    thermal_bounds,
)

SCHEMA_VERSION = "mindie-sim-hardware/1"


@dataclass(frozen=True)
class PlatformConfig:
    soc_version: str
    short_soc_version: str
    npu_arch: str | None
    ai_core_cnt: int
    cube_core_cnt: int
    vector_core_cnt: int
    ai_cpu_cnt: int | None
    memory_size_bytes: int
    l2_size_bytes: int
    cube_freq_mhz: float
    l0_a_bytes: int
    l0_b_bytes: int
    l0_c_bytes: int
    l1_bytes: int
    ub_bytes: int
    link_rates_bpc: Mapping[str, float] = field(default_factory=dict)
    source: str = "<text>"

    @property
    def platform_config_hbm_mib(self) -> float:
        return _mib(self.memory_size_bytes)

    @property
    def platform_config_l2_mib(self) -> float:
        return _mib(self.l2_size_bytes)

    @property
    def l0_a_kib(self) -> float:
        return _kib(self.l0_a_bytes)

    @property
    def l0_b_kib(self) -> float:
        return _kib(self.l0_b_bytes)

    @property
    def l0_c_kib(self) -> float:
        return _kib(self.l0_c_bytes)

    @property
    def l1_kib(self) -> float:
        return _kib(self.l1_bytes)

    @property
    def platform_allocatable_ub_kib(self) -> float:
        return _kib(self.ub_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "soc_version": self.soc_version,
            "short_soc_version": self.short_soc_version,
            "npu_arch": self.npu_arch,
            "ai_core_cnt": self.ai_core_cnt,
            "cube_core_cnt": self.cube_core_cnt,
            "vector_core_cnt": self.vector_core_cnt,
            "ai_cpu_cnt": self.ai_cpu_cnt,
            "platform_config_hbm_mib": self.platform_config_hbm_mib,
            "platform_config_l2_mib": self.platform_config_l2_mib,
            "cube_freq_mhz": self.cube_freq_mhz,
            "l1_kib": self.l1_kib,
            "l0_a_kib": self.l0_a_kib,
            "l0_b_kib": self.l0_b_kib,
            "l0_c_kib": self.l0_c_kib,
            "platform_allocatable_ub_kib": self.platform_allocatable_ub_kib,
            "link_rates_bpc": dict(sorted(self.link_rates_bpc.items())),
            "source": self.source,
        }


def parse_platform_ini(path: str | Path) -> PlatformConfig:
    source = Path(path).expanduser()
    try:
        text = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise HardwareError(f"platform_ini_unreadable:{type(exc).__name__}") from exc
    return parse_platform_ini_text(text, source=str(source))


def parse_platform_ini_text(text: str, *, source: str = "<text>") -> PlatformConfig:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read_string(text.lstrip("\ufeff"))
    except configparser.Error as exc:
        raise HardwareError(f"platform_ini_invalid:{type(exc).__name__}") from exc

    def get(section: str, key: str, default: Any = None) -> Any:
        if not parser.has_section(section) or not parser.has_option(section, key):
            return default
        return parser.get(section, key).strip()

    def required_int(section: str, key: str) -> int:
        raw = get(section, key)
        if raw is None:
            raise HardwareError(f"platform_ini_missing:{section}.{key}")
        try:
            return _int_value(int(raw, 0), f"platform_ini_invalid:{section}.{key}")
        except ValueError as exc:
            raise HardwareError(f"platform_ini_invalid:{section}.{key}") from exc

    def required_float(section: str, key: str) -> float:
        raw = get(section, key)
        if raw is None:
            raise HardwareError(f"platform_ini_missing:{section}.{key}")
        try:
            return _float_value(float(raw), f"platform_ini_invalid:{section}.{key}")
        except ValueError as exc:
            raise HardwareError(f"platform_ini_invalid:{section}.{key}") from exc

    version = "version"
    soc = get(version, "soc_version", "")
    short = get(version, "short_soc_version", soc)
    if not soc:
        raise HardwareError("platform_ini_missing:version.soc_version")
    ai_cpu_raw = get("SoCInfo", "ai_cpu_cnt")
    ai_cpu = int(ai_cpu_raw, 0) if ai_cpu_raw is not None else None
    rates: dict[str, float] = {}
    for key in (
        "ddr_rate", "l2_rate", "l1_to_l0_a_rate", "l1_to_l0_b_rate",
        "l1_to_ub_rate", "l0_c_to_ub_rate", "ub_to_l2_rate", "ub_to_ddr_rate",
        "ub_to_l1_rate",
    ):
        raw = get("AICoreMemoryRates", key)
        if raw is not None:
            try:
                rates[key] = float(raw)
            except ValueError as exc:
                raise HardwareError(f"platform_ini_invalid:AICoreMemoryRates.{key}") from exc
    return PlatformConfig(
        soc_version=soc,
        short_soc_version=short,
        npu_arch=get(version, "NpuArch"),
        ai_core_cnt=required_int("SoCInfo", "ai_core_cnt"),
        cube_core_cnt=required_int("SoCInfo", "cube_core_cnt"),
        vector_core_cnt=required_int("SoCInfo", "vector_core_cnt"),
        ai_cpu_cnt=ai_cpu,
        memory_size_bytes=required_int("SoCInfo", "memory_size"),
        l2_size_bytes=required_int("SoCInfo", "l2_size"),
        cube_freq_mhz=required_float("AICoreSpec", "cube_freq"),
        l0_a_bytes=required_int("AICoreSpec", "l0_a_size"),
        l0_b_bytes=required_int("AICoreSpec", "l0_b_size"),
        l0_c_bytes=required_int("AICoreSpec", "l0_c_size"),
        l1_bytes=required_int("AICoreSpec", "l1_size"),
        ub_bytes=required_int("AICoreSpec", "ub_size"),
        link_rates_bpc=rates,
        source=source,
    )


@dataclass(frozen=True)
class NpuSmiDevice:
    npu_id: int
    chip_id: int
    phy_id: int | None
    name: str
    health: str | None
    power_w: float | None
    temp_c: float | None
    aicore_pct: float | None
    memory_used_mib: int | None
    memory_total_mib: int | None
    hbm_used_mib: int | None
    hbm_total_mib: int | None
    bus_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NpuSmiChip:
    npu_id: int
    chip_id: int
    hbm_usage_rate_pct: float | None
    aicore_freq_mhz: float | None
    aicore_current_freq_mhz: float | None
    aicore_count: int | None
    temp_c: float | None
    power_w: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NpuSmiSnapshot:
    devices: tuple[NpuSmiDevice, ...]
    chips: tuple[NpuSmiChip, ...]
    board: Mapping[str, str]
    source: str = "<text>"

    @property
    def chip_count(self) -> int:
        raw = self.board.get("Chip Count", "")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return max((device.chip_id for device in self.devices), default=-1) + 1

    @property
    def dual_die(self) -> bool:
        return self.chip_count > 1

    @property
    def phy_ids(self) -> tuple[int, ...]:
        return tuple(sorted({d.phy_id for d in self.devices if d.phy_id is not None}))

    @property
    def hbm_totals_mib(self) -> tuple[int, ...]:
        return tuple(sorted({d.hbm_total_mib for d in self.devices if d.hbm_total_mib}))

    @property
    def min_hbm_mib(self) -> float | None:
        values = self.hbm_totals_mib
        return float(min(values)) if values else None

    @property
    def max_hbm_mib(self) -> float | None:
        values = self.hbm_totals_mib
        return float(max(values)) if values else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "chip_count": self.chip_count,
            "dual_die": self.dual_die,
            "phy_ids": list(self.phy_ids),
            "verified_runtime_hbm_mib": self.min_hbm_mib,
            "verified_runtime_hbm_mib_max": self.max_hbm_mib,
            "devices": [device.to_dict() for device in self.devices],
            "chips": [chip.to_dict() for chip in self.chips],
            "board": dict(sorted(self.board.items())),
        }


_BUS_RE = re.compile(r"^[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.")
_PAIR_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def _number(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text.strip())
    except ValueError:
        return None


def parse_npu_smi_info(text: str, *, source: str = "<npu-smi-info>") -> NpuSmiSnapshot:
    devices: list[NpuSmiDevice] = []
    pending: tuple[int, str, str | None, float | None, float | None] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 3:
            continue
        first = cells[0].split()
        if not first or not first[0].isdigit():
            continue
        if first[0].lower() in {"npu", "chip"}:
            continue
        if len(first) >= 2 and not first[1].isdigit():
            stats = cells[2].split()
            pending = (
                int(first[0]),
                first[1],
                cells[1] or None,
                _number(stats[0]) if stats else None,
                _number(stats[1]) if len(stats) > 1 else None,
            )
            continue
        bus_index = next((i for i, cell in enumerate(cells) if _BUS_RE.match(cell)), None)
        if bus_index is None or pending is None:
            continue
        chip_id = int(first[0])
        phy_id = int(first[1]) if len(first) > 1 and first[1].isdigit() else None
        stats = cells[bus_index + 1].split() if len(cells) > bus_index + 1 else []
        pairs = _PAIR_RE.findall(cells[bus_index + 1]) if len(cells) > bus_index + 1 else []
        memory = (int(pairs[0][0]), int(pairs[0][1])) if pairs else (None, None)
        hbm = (int(pairs[1][0]), int(pairs[1][1])) if len(pairs) > 1 else (None, None)
        devices.append(
            NpuSmiDevice(
                npu_id=pending[0],
                chip_id=chip_id,
                phy_id=phy_id,
                name=pending[1],
                health=pending[2],
                power_w=pending[3],
                temp_c=pending[4],
                aicore_pct=_number(stats[0]) if stats else None,
                memory_used_mib=memory[0],
                memory_total_mib=memory[1],
                hbm_used_mib=hbm[0],
                hbm_total_mib=hbm[1],
                bus_id=cells[bus_index],
            )
        )
    if not devices:
        raise HardwareError("npu_smi_info_no_devices")
    return NpuSmiSnapshot(devices=tuple(devices), chips=(), board={}, source=source)


def parse_npu_smi_common(text: str, *, source: str = "<npu-smi-common>") -> tuple[NpuSmiChip, ...]:
    chips: list[NpuSmiChip] = []
    current: dict[str, str] | None = None
    npu_id = 0
    for line in text.splitlines():
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = (part.strip() for part in stripped.split(":", 1))
        if key == "NPU ID":
            try:
                npu_id = int(value)
            except ValueError:
                npu_id = 0
            continue
        if key == "Chip ID":
            if current is not None:
                chips.append(_chip_from_fields(npu_id, current))
            current = {"Chip ID": value}
            continue
        if current is not None:
            current[key] = value
    if current is not None:
        chips.append(_chip_from_fields(npu_id, current))
    if not chips:
        raise HardwareError("npu_smi_common_no_chips")
    return tuple(chips)


def _chip_from_fields(npu_id: int, fields: Mapping[str, str]) -> NpuSmiChip:
    def value(key: str) -> float | None:
        return _number(fields.get(key))

    def integer(key: str) -> int | None:
        raw = fields.get(key)
        try:
            return int(raw) if raw is not None else None
        except ValueError:
            return None

    return NpuSmiChip(
        npu_id=npu_id,
        chip_id=integer("Chip ID") or 0,
        hbm_usage_rate_pct=value("HBM Usage Rate(%)"),
        aicore_freq_mhz=value("Aicore Freq(MHZ)"),
        aicore_current_freq_mhz=value("Aicore curFreq(MHZ)"),
        aicore_count=integer("Aicore Count"),
        temp_c=value("Temperature(C)"),
        power_w=value("NPU Real-time Power(W)"),
    )


def parse_npu_smi_board(text: str) -> dict[str, str]:
    board: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key and value:
            board[key] = value
    if not board:
        raise HardwareError("npu_smi_board_empty")
    return board


def parse_npu_smi_snapshot(
    info_text: str,
    *,
    common_text: str | None = None,
    board_text: str | None = None,
    source: str = "<npu-smi>",
) -> NpuSmiSnapshot:
    info = parse_npu_smi_info(info_text, source=source)
    chips = parse_npu_smi_common(common_text) if common_text else ()
    board = parse_npu_smi_board(board_text) if board_text else {}
    return NpuSmiSnapshot(devices=info.devices, chips=chips, board=board, source=source)


def load_npu_smi_snapshot(path: str | Path) -> NpuSmiSnapshot:
    root = Path(path).expanduser()
    if not root.exists():
        raise HardwareError("npu_smi_path_missing")
    if root.is_dir():
        texts = {p.name: p.read_text(encoding="utf-8-sig", errors="replace") for p in root.glob("*.txt")}
        info = next((text for name, text in texts.items() if "npu-smi" in text and _BUS_RE.search(text)), None)
        common = next((text for name, text in texts.items() if "Chip ID" in text and "Aicore Freq" in text), None)
        board = next((text for name, text in texts.items() if "Product Name" in text and "Chip Count" in text), None)
        if info is None:
            raise HardwareError("npu_smi_info_missing")
        return parse_npu_smi_snapshot(info, common_text=common, board_text=board, source=str(root))
    text = root.read_text(encoding="utf-8-sig", errors="replace")
    if "Product Name" in text and "Chip Count" in text and "npu-smi" not in text:
        return NpuSmiSnapshot(devices=(), chips=(), board=parse_npu_smi_board(text), source=str(root))
    if "Chip ID" in text and "Aicore Freq" in text and "npu-smi" not in text:
        return NpuSmiSnapshot(devices=(), chips=parse_npu_smi_common(text), board={}, source=str(root))
    return parse_npu_smi_info(text, source=str(root))


def cross_check_capacities(
    config: PlatformConfig,
    snapshot: NpuSmiSnapshot | None = None,
    *,
    verified_runtime_hbm_mib: float | None = None,
    simulator_ub_kib: float | None = None,
    sustained_mhz_range: Sequence[float] | None = None,
) -> dict[str, Any]:
    if snapshot is not None and snapshot.devices:
        runtime_hbm = snapshot.min_hbm_mib
        runtime_max = snapshot.max_hbm_mib
    else:
        runtime_hbm = float(verified_runtime_hbm_mib) if verified_runtime_hbm_mib is not None else None
        runtime_max = runtime_hbm
    flags: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    hbm_discrepancy = None
    if runtime_hbm is not None:
        hbm_discrepancy = config.platform_config_hbm_mib - runtime_hbm
        if abs(hbm_discrepancy) > 0.5:
            flags.append("platform_config_hbm_mib_differs_from_runtime")
            diagnostics.append({
                "code": "platform_config_hbm_mib_differs_from_runtime",
                "platform_config_hbm_mib": config.platform_config_hbm_mib,
                "verified_runtime_hbm_mib": runtime_hbm,
                "difference_mib": hbm_discrepancy,
                "authority": "verified_runtime_capacity",
            })
    else:
        flags.append("runtime_hbm_unverified")
        diagnostics.append({"code": "runtime_hbm_unverified", "detail": "no npu-smi runtime capacity evidence"})
    ub_discrepancy = None
    if simulator_ub_kib is not None:
        ub_discrepancy = float(simulator_ub_kib) - config.platform_allocatable_ub_kib
        if abs(ub_discrepancy) > 0.5:
            flags.append("simulator_ub_kib_differs_from_allocatable")
            diagnostics.append({
                "code": "simulator_ub_kib_differs_from_allocatable",
                "simulator_ub_kib": float(simulator_ub_kib),
                "platform_allocatable_ub_kib": config.platform_allocatable_ub_kib,
                "difference_kib": ub_discrepancy,
            })
    if snapshot is not None and snapshot.chips:
        runtime_counts = sorted({chip.aicore_count for chip in snapshot.chips if chip.aicore_count})
        if runtime_counts and any(value != config.ai_core_cnt for value in runtime_counts):
            flags.append("runtime_aicore_count_differs_from_platform_config")
            diagnostics.append({
                "code": "runtime_aicore_count_differs_from_platform_config",
                "platform_config_ai_core_cnt": config.ai_core_cnt,
                "runtime_aicore_count": runtime_counts,
            })
        runtime_clocks = sorted({chip.aicore_freq_mhz for chip in snapshot.chips if chip.aicore_freq_mhz})
        if runtime_clocks and any(abs(value - config.cube_freq_mhz) > 0.5 for value in runtime_clocks):
            flags.append("runtime_configured_clock_differs_from_platform_config")
            diagnostics.append({
                "code": "runtime_configured_clock_differs_from_platform_config",
                "platform_config_cube_freq_mhz": config.cube_freq_mhz,
                "runtime_aicore_freq_mhz": runtime_clocks,
                "runtime_aicore_current_freq_mhz": sorted(
                    {chip.aicore_current_freq_mhz for chip in snapshot.chips if chip.aicore_current_freq_mhz}
                ),
            })
    if sustained_mhz_range is not None:
        low, high = sorted(float(v) for v in sustained_mhz_range)
        if high < config.cube_freq_mhz:
            flags.append("platform_clock_is_not_sustained_clock")
            diagnostics.append({
                "code": "platform_clock_is_not_sustained_clock",
                "platform_config_cube_freq_mhz": config.cube_freq_mhz,
                "sustained_mhz_range": [low, high],
            })
    return {
        "soc_version": config.soc_version,
        "short_soc_version": config.short_soc_version,
        "platform_config_hbm_mib": config.platform_config_hbm_mib,
        "verified_runtime_hbm_mib": runtime_hbm,
        "verified_runtime_hbm_mib_max": runtime_max,
        "hbm_discrepancy_mib": hbm_discrepancy,
        "platform_allocatable_ub_kib": config.platform_allocatable_ub_kib,
        "simulator_ub_kib": float(simulator_ub_kib) if simulator_ub_kib is not None else None,
        "allocatable_ub_kib": config.platform_allocatable_ub_kib,
        "ub_discrepancy_kib": ub_discrepancy,
        "runtime_verified": runtime_hbm is not None,
        "capacity_authority": "verified_runtime" if runtime_hbm is not None else "platform_config_metadata",
        "flags": flags,
        "diagnostics": diagnostics,
        "note": "Platform INI values are metadata. Runtime HBM must come from npu-smi.",
    }


def parse_cannsim_link_ceilings(path_or_text: str | Path) -> dict[str, Any]:
    candidate = Path(path_or_text) if isinstance(path_or_text, (str, Path)) else None
    text: str
    source: str
    if candidate is not None and candidate.exists() and candidate.is_file():
        text = candidate.read_text(encoding="utf-8-sig", errors="replace")
        source = str(candidate)
    else:
        text = str(path_or_text)
        source = "<text>"
    ceilings: dict[str, float] = {}
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise HardwareError("cannsim_link_json_invalid") from exc
        items = data.items() if isinstance(data, Mapping) else ()
        for key, value in items:
            ceilings[_canonical_link_key(str(key))] = _float_value(value, "cannsim_link_invalid")
    else:
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            try:
                numeric = float(value)
            except ValueError:
                continue
            ceilings[_canonical_link_key(key)] = _float_value(numeric, "cannsim_link_invalid")
    if not ceilings:
        raise HardwareError("cannsim_link_constants_empty")
    return {
        "ceilings_bpc": dict(sorted(ceilings.items())),
        "source": source,
        "verified_bandwidth": False,
        "note": "Simulator link ceilings are B/cycle references, not measured bandwidth.",
    }


def hardware_report(
    config: PlatformConfig | None = None,
    *,
    soc: str | None = None,
    sku_profile: str | Path | Mapping[str, Any] | None = None,
    snapshot: NpuSmiSnapshot | None = None,
    topo_snapshot: NpuTopologySnapshot | None = None,
    pcie_snapshot: PcieLinkSnapshot | None = None,
    verified_runtime_hbm_mib: float | None = None,
    simulator_ub_kib: float | None = 256.0,
    allocatable_ub_kib: float | None = None,
    sustained_mhz_range: Sequence[float] = SUSTAINED_MHZ_RANGE,
    buffer_plan: Mapping[str, Any] | None = None,
    workload: Mapping[str, Any] | None = None,
    transfer_plan: Mapping[str, Any] | None = None,
    cannsim_links: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    profile = resolve_device_capability_profile(
        config=config,
        soc=soc,
        sku_profile=sku_profile,
        snapshot=snapshot,
        topo_snapshot=topo_snapshot,
        pcie_snapshot=pcie_snapshot,
        verified_runtime_hbm_mib=verified_runtime_hbm_mib,
        simulator_ub_kib=simulator_ub_kib,
        allocatable_ub_kib=allocatable_ub_kib,
        sustained_mhz_range=sustained_mhz_range,
        cannsim_links=cannsim_links,
    )
    if profile.status == "ambiguous_sku":
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "ambiguous_sku",
            "unmodeled_reason": profile.unmodeled_reason,
            "device_capability_profile": profile.to_dict(),
            "soc": None,
            "capacity_cross_check": None,
            "cannsim_links": dict(cannsim_links) if cannsim_links else default_link_ceilings(),
        }
        if snapshot is not None:
            payload["npu_smi"] = snapshot.to_dict()
        if buffer_plan:
            capacities = buffer_plan.get("capacities_kib") or {
                "L1": profile.cube_spec.l1_kib,
                "L0A": profile.cube_spec.l0a_kib,
                "L0B": profile.cube_spec.l0b_kib,
                "L0C": profile.cube_spec.l0c_kib,
                "UB": profile.vector_spec.allocatable_ub_kib,
            }
            payload["buffers"] = validate_buffer_allocations(
                buffer_plan.get("allocations", ()),
                capacities,
                alignment_bytes=int(buffer_plan.get("alignment_bytes", DEFAULT_ALIGNMENT_BYTES)),
                ping_pong=buffer_plan.get("ping_pong"),
            )
        if workload:
            payload["thermal"] = {
                "status": "unmodeled",
                "unmodeled_reason": profile.unmodeled_reason,
                "ideal_hardware_lower_bound_us": None,
            }
        return payload

    if config is None:
        config = PlatformConfig(
            soc_version=profile.clock_and_topology_spec.soc_version or (soc or "custom_sku"),
            short_soc_version=profile.clock_and_topology_spec.short_soc_version or (soc or "custom_sku"),
            npu_arch=profile.clock_and_topology_spec.npu_arch,
            ai_core_cnt=profile.cube_spec.aic_count_per_die or 0,
            cube_core_cnt=profile.cube_spec.aic_count_per_die or 0,
            vector_core_cnt=profile.vector_spec.aiv_count_per_die or 0,
            ai_cpu_cnt=None,
            memory_size_bytes=int((profile.memory_hierarchy_spec.platform_config_hbm_mib or 0.0) * 1024 * 1024),
            l2_size_bytes=int((profile.memory_hierarchy_spec.l2_capacity_mib or 0.0) * 1024 * 1024),
            cube_freq_mhz=float(profile.clock_and_topology_spec.configured_clock_mhz or 0.0),
            l0_a_bytes=int(profile.cube_spec.l0a_kib * 1024),
            l0_b_bytes=int(profile.cube_spec.l0b_kib * 1024),
            l0_c_bytes=int(profile.cube_spec.l0c_kib * 1024),
            l1_bytes=int(profile.cube_spec.l1_kib * 1024),
            ub_bytes=int(profile.vector_spec.allocatable_ub_kib * 1024),
            link_rates_bpc=profile.memory_hierarchy_spec.link_ceilings_bpc,
            source=",".join(profile.resolution_sources) or "<sku-profile>",
        )
    cross = cross_check_capacities(
        config,
        snapshot,
        verified_runtime_hbm_mib=verified_runtime_hbm_mib,
        simulator_ub_kib=simulator_ub_kib,
        sustained_mhz_range=sustained_mhz_range,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "modeled" if cross["runtime_verified"] else "runtime_unverified",
        "unmodeled_reason": None,
        "device_capability_profile": profile.to_dict(),
        "soc": config.to_dict(),
        "capacity_cross_check": cross,
        "cannsim_links": dict(cannsim_links) if cannsim_links else default_link_ceilings(),
    }
    if snapshot is not None:
        payload["npu_smi"] = snapshot.to_dict()
    if topo_snapshot is not None:
        payload["topology"] = topo_snapshot.to_dict()
    if pcie_snapshot is not None:
        payload["pcie_link"] = pcie_snapshot.to_dict()
    if transfer_plan:
        payload["transfers"] = evaluate_transfer_plan(
            transfer_plan,
            profile.memory_hierarchy_spec.transfer_channels,
            dies_per_package=profile.clock_and_topology_spec.dies_per_package,
        )
    if buffer_plan:
        capacities = buffer_plan.get("capacities_kib") or {
            "L1": profile.cube_spec.l1_kib,
            "L0A": profile.cube_spec.l0a_kib,
            "L0B": profile.cube_spec.l0b_kib,
            "L0C": profile.cube_spec.l0c_kib,
            "UB": profile.vector_spec.allocatable_ub_kib,
        }
        payload["buffers"] = validate_buffer_allocations(
            buffer_plan.get("allocations", ()),
            capacities,
            alignment_bytes=int(buffer_plan.get("alignment_bytes", DEFAULT_ALIGNMENT_BYTES)),
            ping_pong=buffer_plan.get("ping_pong"),
        )
    if workload:
        thermal = thermal_bounds(
            fp8_flops=float(workload.get("flops_fp8", 206158430208.0)),
            bf16_flops=float(workload.get("flops_bf16", 12884901888.0)),
            sustained_mhz_range=sustained_mhz_range,
            target_us=float(workload.get("target_us", 350.0)),
            wall_cycles=float(workload.get("wall_cycles", 566344.125)),
            counted_cycles=float(workload.get("counted_cycles", 476981.3125)),
            vector_cycles=float(workload.get("vector_cycles", 456026.906)),
            fix_cycles=float(workload.get("fix_cycles", 437384.344)),
        )
        traffic = workload.get("link_traffic_bytes_per_aic")
        if traffic:
            links_source = cannsim_links or default_link_ceilings()
            ceilings = links_source.get("ceilings_bpc", links_source) if isinstance(links_source, Mapping) else {}
            thermal["link_bounds"] = link_reference_bounds(
                traffic,
                ceilings_bpc=ceilings,
                mhz=float(workload.get("link_reference_mhz", 1280.0)),
            )
        payload["thermal"] = thermal
    return payload


# Public aliases used by the CLI and tests.
parse_ini = parse_platform_ini
cross_check = cross_check_capacities
parse_cannsim = parse_cannsim_link_ceilings
__all__ = [
    "SCHEMA_VERSION", "AMBIGUOUS_SKU_REASON", "BUILTIN_SKU_CATALOG", "HardwareError",
    "PlatformConfig", "CubeSpec", "VectorSpec", "MemoryHierarchySpec", "ClockAndTopologySpec",
    "DeviceCapabilityProfile", "NpuSmiDevice", "NpuSmiChip", "NpuSmiSnapshot", "BufferAllocation",
    "A5_CANN_SIM_LINK_CEILINGS_BPC", "A5_REFERENCE_PEAKS", "SUSTAINED_MHZ_RANGE",
    "SHORT_BURST_MHZ_RANGE", "DEFAULT_ALIGNMENT_BYTES", "PING_PONG_ALIGNMENT_BYTES",
    "align_up", "infer_arch_family", "is_ambiguous_family_token", "load_sku_profile",
    "resolve_device_capability_profile", "parse_platform_ini", "parse_platform_ini_text",
    "parse_ini", "parse_npu_smi_info", "parse_npu_smi_common", "parse_npu_smi_board",
    "parse_npu_smi_snapshot", "load_npu_smi_snapshot", "cross_check_capacities", "cross_check",
    "parse_cannsim_link_ceilings", "parse_cannsim", "default_link_ceilings",
    "validate_buffer_allocations", "check_mx_scale_layout", "cube_reference_cycles",
    "cycles_to_us", "reference_time_us", "sensitivity_matrix", "thermal_bounds",
    "link_reference_bounds", "hardware_report", "TRANSFER_CHANNELS", "TransferChannelSpec",
    "NpuTopologySnapshot", "PcieLinkSnapshot", "parse_npu_smi_topo", "parse_pcie_link_info",
    "default_transfer_channels", "solve_transfer_bound", "solve_collective_bound", "evaluate_transfer_plan",
]
