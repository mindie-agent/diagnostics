"""Shared constants and validation primitives for the Layer 1 hardware package.

[Role]
  Provide the error type, unit constants, alignment helper and A5 simulator
  reference constants used by ``hardware``, ``buffers`` and ``thermal``.

[Invariants]
  * Constants here are references or units, never verified board measurements.
  * Alignment is a pure integer ceiling operation; no silent clamping.
  * ``HardwareError.reason`` is stable and caller-safe.

[Timeouts & Locks]
  Pure constants and functions.  No I/O, locks, sleeps or subprocesses.

[Side-Effect Policy]
  No filesystem, network, device or global-state mutation.
"""
from __future__ import annotations

import math
from typing import Any

MIB = 1024 * 1024
KIB = 1024
DEFAULT_ALIGNMENT_BYTES = 32
PING_PONG_ALIGNMENT_BYTES = 256
SUSTAINED_MHZ_RANGE = (1250.0, 1300.0)
SHORT_BURST_MHZ_RANGE = (1640.0, 1650.0)
A5_REFERENCE_PEAKS = {
    "fp8_flops_per_s": 864e12,
    "bf16_flops_per_s": 432e12,
    "reference_mhz": 1650.0,
    "source": "generic cannsim roofline; not a board-certified peak",
}
A5_CANN_SIM_LINK_CEILINGS_BPC = {
    "gm_l2_to_l1": 128,
    "l1_to_l0a": 256,
    "l1_to_l0b": 256,
    "l1_to_ub": 128,
    "l0c_to_ub": 128,
    "ub_to_l2": 128,
    "ub_to_ddr": 128,
    "ub_to_l1": 128,
    "simd_to_ub": 256,
    "l0c_to_fixp": 512,
    "l0a_or_l0b_to_cube": 512,
    "cube_to_l0c": 1024,
}
_LINK_ALIASES = {
    "GM_L2_TO_L1": "gm_l2_to_l1",
    "L1_TO_L0A": "l1_to_l0a",
    "L1_TO_L0B": "l1_to_l0b",
    "L1_TO_UB": "l1_to_ub",
    "L0C_TO_UB": "l0c_to_ub",
    "L0C_TO_UB0": "l0c_to_ub",
    "L0C_TO_UB1": "l0c_to_ub",
    "UB_TO_L2": "ub_to_l2",
    "UB_TO_DDR": "ub_to_ddr",
    "UB_TO_L1": "ub_to_l1",
    "SIMD_TO_UB": "simd_to_ub",
    "L0C_TO_FIXP": "l0c_to_fixp",
    "L0A_OR_L0B_TO_CUBE": "l0a_or_l0b_to_cube",
    "CUBE_TO_L0C": "cube_to_l0c",
}


class HardwareError(ValueError):
    """Stable hardware-input validation failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _int_value(value: Any, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HardwareError(reason)
    return value


def _float_value(value: Any, reason: str) -> float:
    if not _finite(value) or float(value) < 0:
        raise HardwareError(reason)
    return float(value)


def align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return int(value)
    return (int(value) + alignment - 1) // alignment * alignment


def _mib(value: int | float) -> float:
    return float(value) / MIB


def _kib(value: int | float) -> float:
    return float(value) / KIB


def _canonical_link_key(key: str) -> str:
    normalized = key.strip().upper()
    for suffix in ("_BYTES_PER_CYCLE", "_BPC"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return _LINK_ALIASES.get(normalized, normalized.lower())


def default_link_ceilings() -> dict[str, Any]:
    return {
        "ceilings_bpc": dict(A5_CANN_SIM_LINK_CEILINGS_BPC),
        "source": "A5_HARDWARE_CONSTRAINTS.md / cannsim constants",
        "verified_bandwidth": False,
        "note": "Simulator link ceilings are B/cycle references, not measured bandwidth.",
    }


__all__ = [
    "MIB",
    "KIB",
    "DEFAULT_ALIGNMENT_BYTES",
    "PING_PONG_ALIGNMENT_BYTES",
    "SUSTAINED_MHZ_RANGE",
    "SHORT_BURST_MHZ_RANGE",
    "A5_REFERENCE_PEAKS",
    "A5_CANN_SIM_LINK_CEILINGS_BPC",
    "HardwareError",
    "_finite",
    "_int_value",
    "_float_value",
    "align_up",
    "_mib",
    "_kib",
    "_canonical_link_key",
    "default_link_ceilings",
]
