"""Layer 1 thermal clock ranges, reference lower bounds and link ceilings.

[Role]
  Convert declared FP8/BF16 work into reference Cube cycles, express reference
  lower bounds at sustained and short-burst clocks, build the (W-C) sensitivity
  matrix, and compute per-link byte/cycle lower bounds.

[Invariants]
  * Reference cycles use caller/simulator peaks and are lower bounds, not
    board-certified measurements or predictions of a whole kernel.
  * Sustained clocks and short-burst clocks remain separate ranges; a boost
    observation never becomes a sustained guarantee.
  * ``T(r,f) = [C + (1-r)*(W-C)]/f_MHz`` is reproducible arithmetic; ``r`` is a
    hypothetical removed fraction, not an identified bubble.
  * Link bounds are individual ceilings and are not summed across shared ports.

[Timeouts & Locks]
  Pure computation.  No I/O, locks, sleeps, subprocesses or caches.

[Side-Effect Policy]
  No filesystem, network, device or global-state mutation.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .hardware_common import (
    A5_CANN_SIM_LINK_CEILINGS_BPC,
    A5_REFERENCE_PEAKS,
    SHORT_BURST_MHZ_RANGE,
    SUSTAINED_MHZ_RANGE,
    HardwareError,
    _canonical_link_key,
    _finite,
    _float_value,
)


def cube_reference_cycles(
    flops_fp8: float,
    flops_bf16: float,
    *,
    peak_fp8_flops_per_s: float = A5_REFERENCE_PEAKS["fp8_flops_per_s"],
    peak_bf16_flops_per_s: float = A5_REFERENCE_PEAKS["bf16_flops_per_s"],
    reference_mhz: float = A5_REFERENCE_PEAKS["reference_mhz"],
) -> float:
    if not _finite(flops_fp8) or not _finite(flops_bf16) or flops_fp8 < 0 or flops_bf16 < 0:
        raise HardwareError("cube_reference_flops_invalid")
    if not _finite(peak_fp8_flops_per_s) or not _finite(peak_bf16_flops_per_s):
        raise HardwareError("cube_reference_peak_invalid")
    if peak_fp8_flops_per_s <= 0 or peak_bf16_flops_per_s <= 0 or reference_mhz <= 0:
        raise HardwareError("cube_reference_peak_invalid")
    seconds = flops_fp8 / float(peak_fp8_flops_per_s) + flops_bf16 / float(peak_bf16_flops_per_s)
    return seconds * float(reference_mhz) * 1e6


def cycles_to_us(cycles: float, mhz: float) -> float:
    if not _finite(cycles) or not _finite(mhz) or mhz <= 0:
        raise HardwareError("cycle_time_invalid")
    return float(cycles) / float(mhz)


def reference_time_us(cycles: float, mhz: float) -> float:
    return cycles_to_us(cycles, mhz)


def sensitivity_matrix(
    wall_cycles: float,
    counted_cycles: float,
    *,
    frequencies_mhz: Sequence[float] = (1250.0, 1280.0, 1300.0),
    r_values: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    wall = _float_value(wall_cycles, "wall_cycles_invalid")
    counted = _float_value(counted_cycles, "counted_cycles_invalid")
    if counted > wall:
        raise HardwareError("counted_cycles_exceed_wall_cycles")
    frequencies = sorted({float(value) for value in frequencies_mhz if _finite(value) and float(value) > 0})
    fractions = [float(value) for value in r_values]
    if not frequencies or any(not _finite(value) or value < 0 or value > 1 for value in fractions):
        raise HardwareError("sensitivity_matrix_arguments_invalid")
    gap = wall - counted
    table: list[dict[str, Any]] = []
    for fraction in fractions:
        remaining = counted + (1.0 - fraction) * gap
        times = {f"{frequency:g}": remaining / frequency for frequency in frequencies}
        table.append({
            "r": fraction,
            "removed_gap_cycles": fraction * gap,
            "remaining_cycles": remaining,
            "time_us": times,
        })
    return {
        "wall_cycles": wall,
        "counted_cycles": counted,
        "gap_cycles": gap,
        "frequencies_mhz": frequencies,
        "r_values": fractions,
        "table": table,
        "formula": "T(r,f) = [C + (1-r)*(W-C)] / f_MHz",
        "assumptions": [
            "W and C are aggregate PMU cycle counts, not per-instruction service times",
            "r is a hypothetical removable fraction, not an identified bubble",
            "frequency changes are assumed not to change the counted cycles",
        ],
    }


def thermal_bounds(
    *,
    fp8_flops: float = 206158430208.0,
    bf16_flops: float = 12884901888.0,
    sustained_mhz_range: Sequence[float] = SUSTAINED_MHZ_RANGE,
    short_burst_mhz_range: Sequence[float] = SHORT_BURST_MHZ_RANGE,
    extra_frequencies_mhz: Sequence[float] = (1280.0,),
    target_us: float = 350.0,
    wall_cycles: float = 566344.125,
    counted_cycles: float = 476981.3125,
    vector_cycles: float = 456026.906,
    fix_cycles: float = 437384.344,
    link_traffic_bytes_per_aic: Mapping[str, int | float] | None = None,
    link_reference_mhz: float = 1280.0,
    link_ceilings_bpc: Mapping[str, int | float] | None = None,
) -> dict[str, Any]:
    sustained = sorted(float(value) for value in sustained_mhz_range)
    burst = sorted(float(value) for value in short_burst_mhz_range)
    if len(sustained) != 2 or len(burst) != 2 or sustained[0] <= 0 or burst[0] <= 0:
        raise HardwareError("thermal_clock_range_invalid")
    if sustained[0] > sustained[1] or burst[0] > burst[1]:
        raise HardwareError("thermal_clock_range_invalid")
    frequencies = sorted({*sustained, *burst, *(float(value) for value in extra_frequencies_mhz if _finite(value) and float(value) > 0)})
    cycles = cube_reference_cycles(fp8_flops, bf16_flops)
    lower_bounds = {f"{frequency:g}": reference_time_us(cycles, frequency) for frequency in frequencies}
    target = _float_value(target_us, "thermal_target_invalid")
    if target <= 0:
        raise HardwareError("thermal_target_invalid")
    sensitivity = sensitivity_matrix(wall_cycles, counted_cycles)
    required = {
        "reference_cube_mhz": cycles / target,
        "counted_mac_mhz": float(counted_cycles) / target,
        "vector_mhz": float(vector_cycles) / target,
        "fix_mhz": float(fix_cycles) / target,
        "wall_mhz": float(wall_cycles) / target,
    }
    result = {
        "status": "modeled",
        "reference_cube_cycles_per_aic": cycles,
        "reference_mhz_for_cycles": A5_REFERENCE_PEAKS["reference_mhz"],
        "frequencies_mhz": frequencies,
        "reference_lower_bounds_us": lower_bounds,
        "sustained_mhz_range": sustained,
        "sustained_reference_us": [reference_time_us(cycles, sustained[0]), reference_time_us(cycles, sustained[1])],
        "short_burst_mhz_range": burst,
        "short_burst_reference_us": [reference_time_us(cycles, burst[0]), reference_time_us(cycles, burst[1])],
        "target_us": target,
        "required_mhz_for_target": required,
        "sensitivity": sensitivity,
        "assumptions": [
            "cube reference cycles use generic FP8/BF16 simulator peaks at 1650 MHz",
            "sustained thermal range is supplied evidence, not inferred from the 1650 MHz configured clock",
            "short-burst clocks are never extrapolated to sustained performance",
            "the reference lower bound excludes launch, prologue/drain and resource contention",
        ],
    }
    if link_traffic_bytes_per_aic:
        result["link_bounds"] = link_reference_bounds(
            link_traffic_bytes_per_aic,
            ceilings_bpc=link_ceilings_bpc,
            mhz=link_reference_mhz,
        )
    return result


def link_reference_bounds(
    traffic_bytes: Mapping[str, int | float],
    *,
    ceilings_bpc: Mapping[str, int | float] | None = None,
    mhz: float = 1280.0,
) -> dict[str, Any]:
    ceilings = dict(A5_CANN_SIM_LINK_CEILINGS_BPC) if ceilings_bpc is None else {
        _canonical_link_key(str(key)): float(value) for key, value in ceilings_bpc.items()
    }
    if not _finite(mhz) or mhz <= 0:
        raise HardwareError("link_reference_mhz_invalid")
    links: list[dict[str, Any]] = []
    for key, raw_bytes in sorted(traffic_bytes.items()):
        canonical = _canonical_link_key(str(key))
        bpc = ceilings.get(canonical)
        if bpc is None:
            raise HardwareError(f"link_ceiling_missing:{canonical}")
        byte_count = _float_value(raw_bytes, f"link_traffic_invalid:{canonical}")
        links.append({
            "link": canonical,
            "bytes_per_aic": byte_count,
            "bytes_per_cycle": bpc,
            "minimum_us": byte_count / (bpc * mhz),
        })
    if not links:
        raise HardwareError("link_traffic_empty")
    return {
        "at_mhz": float(mhz),
        "links": links,
        "max_us": max(item["minimum_us"] for item in links),
        "sum_us_not_additive": True,
        "note": "Individual link ceilings; do not sum links sharing ports or contention.",
    }


