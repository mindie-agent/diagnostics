"""Memory-unit arithmetic shared by attribution and peak comparison.

[Role]
  Convert explicit memory values to bytes and validate finite numeric inputs.

[Invariants]
  * Unit spelling is explicit; GB is decimal 1e9 and GiB is binary 1024^3.
  * No magnitude-based unit guessing.

[Timeouts & Locks]
  Pure arithmetic.

[Side-Effect Policy]
  No I/O or mutation.
"""

from __future__ import annotations

import math
from typing import Any

UNIT_FACTORS = {
    "B": 1.0, "BYTE": 1.0, "BYTES": 1.0,
    "KB": 1_000.0, "MB": 1_000_000.0, "GB": 1_000_000_000.0, "TB": 1_000_000_000_000.0,
    "KIB": 1024.0, "MIB": 1024.0 ** 2, "GIB": 1024.0 ** 3, "TIB": 1024.0 ** 4,
}

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

def to_bytes(value: Any, unit: str) -> float | None:
    """Convert a finite numeric memory value to bytes with an explicit unit."""
    if value is None:
        return None
    if not _is_number(value):
        raise ValueError(f"memory value must be a finite number, got {value!r}")
    key = str(unit).strip().upper().replace(" ", "")
    if key not in UNIT_FACTORS:
        raise ValueError(f"unsupported memory unit {unit!r}")
    return float(value) * UNIT_FACTORS[key]

__all__ = ["UNIT_FACTORS", "_is_number", "to_bytes"]
