"""Layer 1 on-chip buffer allocation, alignment and overlap checks.

[Role]
  Validate L1/L0A/L0B/L0C/UB allocations, 32B/256B alignment rounding,
  ping-pong slot sizing, region overlap and the MLAPO L1B MX-scale offset.

[Invariants]
  * Sizes and offsets are rounded only by the requested alignment; overlap,
    overflow and misalignment remain visible diagnostics.
  * A ping-pong plan is two independently aligned slots; slot sizing never
    silently consumes the adjacent region.
  * MX scales must start after the full K*baseN weight tile; raising baseN
    without relocating the scale region is a collision, not a warning.

[Timeouts & Locks]
  Pure integer arithmetic.  No I/O, locks, sleeps, subprocesses or caches.

[Side-Effect Policy]
  Reads caller mappings only; no filesystem, network or device mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .hardware_common import (
    DEFAULT_ALIGNMENT_BYTES,
    KIB,
    PING_PONG_ALIGNMENT_BYTES,
    HardwareError,
    _int_value,
    align_up,
)

_BUFFER_ALIASES = {
    "L1": "L1",
    "L1A": "L1",
    "L1B": "L1",
    "L0A": "L0A",
    "L0B": "L0B",
    "L0C": "L0C",
    "UB": "UB",
}


@dataclass(frozen=True)
class BufferAllocation:
    name: str
    buffer: str
    offset_bytes: int
    size_bytes: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BufferAllocation":
        buffer_name = str(value.get("buffer") or value.get("kind") or "")
        canonical = _BUFFER_ALIASES.get(buffer_name.strip().upper())
        if canonical is None:
            raise HardwareError(f"buffer_unknown:{buffer_name}")
        offset = value.get("offset_bytes")
        if offset is None and value.get("offset_kib") is not None:
            offset = int(round(float(value["offset_kib"]) * KIB))
        if offset is None:
            offset = 0
        size = value.get("size_bytes")
        if size is None and value.get("size_kib") is not None:
            size = int(round(float(value["size_kib"]) * KIB))
        if size is None:
            raise HardwareError("buffer_allocation_missing_size")
        return cls(
            name=str(value.get("name", buffer_name)),
            buffer=canonical,
            offset_bytes=_int_value(int(offset), "buffer_offset_invalid"),
            size_bytes=_int_value(int(size), "buffer_size_invalid"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "buffer": self.buffer,
            "offset_bytes": self.offset_bytes,
            "size_bytes": self.size_bytes,
        }


def validate_buffer_allocations(
    allocations: Iterable[Mapping[str, Any] | BufferAllocation],
    capacities_kib: Mapping[str, int | float],
    *,
    alignment_bytes: int = DEFAULT_ALIGNMENT_BYTES,
    ping_pong: Iterable[Mapping[str, Any]] | None = None,
    pack_missing_offsets: bool = False,
) -> dict[str, Any]:
    capacities = {name.strip().upper(): int(round(float(value) * KIB)) for name, value in capacities_kib.items()}
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    by_buffer: dict[str, list[dict[str, Any]]] = {}
    cursor: dict[str, int] = {}
    for raw in allocations:
        explicit_offset = isinstance(raw, BufferAllocation) or (
            isinstance(raw, Mapping)
            and (raw.get("offset_bytes") is not None or raw.get("offset_kib") is not None)
        )
        allocation = raw if isinstance(raw, BufferAllocation) else BufferAllocation.from_mapping(raw)
        if pack_missing_offsets and not explicit_offset:
            allocation = BufferAllocation(
                name=allocation.name,
                buffer=allocation.buffer,
                offset_bytes=cursor.get(allocation.buffer, 0),
                size_bytes=allocation.size_bytes,
            )
        if allocation.buffer not in capacities:
            diagnostics.append({"code": "buffer_capacity_unknown", "buffer": allocation.buffer, "name": allocation.name})
            continue
        aligned_offset = align_up(allocation.offset_bytes, alignment_bytes)
        aligned_size = align_up(allocation.size_bytes, alignment_bytes)
        end = aligned_offset + aligned_size
        capacity = capacities[allocation.buffer]
        record = {
            "name": allocation.name,
            "buffer": allocation.buffer,
            "offset_bytes": allocation.offset_bytes,
            "size_bytes": allocation.size_bytes,
            "aligned_offset_bytes": aligned_offset,
            "aligned_size_bytes": aligned_size,
            "aligned_end_bytes": end,
            "capacity_bytes": capacity,
            "overflow_bytes": max(0, end - capacity),
            "waste_bytes": (aligned_offset - allocation.offset_bytes) + (aligned_size - allocation.size_bytes),
            "offset_aligned": allocation.offset_bytes % alignment_bytes == 0,
            "size_aligned": allocation.size_bytes % alignment_bytes == 0,
        }
        if not record["offset_aligned"]:
            diagnostics.append({"code": "buffer_offset_misaligned", "name": allocation.name, "offset_bytes": allocation.offset_bytes, "alignment_bytes": alignment_bytes})
        if not record["size_aligned"]:
            diagnostics.append({"code": "buffer_size_misaligned", "name": allocation.name, "size_bytes": allocation.size_bytes, "alignment_bytes": alignment_bytes})
        if record["overflow_bytes"]:
            diagnostics.append({"code": "buffer_overflow", "name": allocation.name, "buffer": allocation.buffer, "overflow_bytes": record["overflow_bytes"]})
        records.append(record)
        by_buffer.setdefault(allocation.buffer, []).append(record)
        cursor[allocation.buffer] = max(cursor.get(allocation.buffer, 0), end)

    overlaps: list[dict[str, Any]] = []
    for buffer_name, items in by_buffer.items():
        ordered = sorted(items, key=lambda item: (item["aligned_offset_bytes"], item["name"]))
        for previous, current in zip(ordered, ordered[1:]):
            overlap = previous["aligned_end_bytes"] - current["aligned_offset_bytes"]
            if overlap > 0:
                overlaps.append({
                    "buffer": buffer_name,
                    "first": previous["name"],
                    "second": current["name"],
                    "overlap_bytes": overlap,
                })
    for item in overlaps:
        diagnostics.append({"code": "buffer_region_overlap", **item})

    ping_records: list[dict[str, Any]] = []
    for raw in ping_pong or ():
        buffer_name = _BUFFER_ALIASES.get(str(raw.get("buffer", "")).strip().upper())
        if buffer_name is None or buffer_name not in capacities:
            diagnostics.append({"code": "ping_pong_buffer_unknown", "buffer": raw.get("buffer")})
            continue
        size = raw.get("slot_size_bytes")
        if size is None and raw.get("slot_size_kib") is not None:
            size = int(round(float(raw["slot_size_kib"]) * KIB))
        slots = int(raw.get("slots", 2))
        if size is None or slots < 1:
            diagnostics.append({"code": "ping_pong_invalid", "buffer": buffer_name})
            continue
        base = raw.get("base_offset_bytes")
        if base is None and raw.get("base_offset_kib") is not None:
            base = int(round(float(raw["base_offset_kib"]) * KIB))
        base = align_up(int(base or 0), PING_PONG_ALIGNMENT_BYTES)
        slot_aligned = align_up(int(size), PING_PONG_ALIGNMENT_BYTES)
        total = slot_aligned * slots
        total_end = base + total
        overflow = max(0, total_end - capacities[buffer_name])
        record = {
            "buffer": buffer_name,
            "base_offset_bytes": base,
            "slot_size_bytes": int(size),
            "slot_aligned_bytes": slot_aligned,
            "slots": slots,
            "total_bytes": total,
            "total_end_bytes": total_end,
            "capacity_bytes": capacities[buffer_name],
            "overflow_bytes": overflow,
            "fits": overflow == 0,
        }
        if overflow:
            diagnostics.append({"code": "ping_pong_overflow", **record})
        ping_records.append(record)
    return {
        "status": "ok" if not diagnostics else "violations",
        "ok": not diagnostics,
        "alignment_bytes": alignment_bytes,
        "capacities_kib": {name: value / KIB for name, value in sorted(capacities.items())},
        "allocations": records,
        "overlaps": overlaps,
        "ping_pong": ping_records,
        "diagnostics": diagnostics,
    }


def check_mx_scale_layout(
    base_n: int,
    *,
    k: int = 512,
    dtype_bytes: int = 1,
    scale_offset_kib: float = 64.0,
    scale_size_bytes: int = 4096,
    slot_size_kib: float = 128.0,
    alignment_bytes: int = PING_PONG_ALIGNMENT_BYTES,
) -> dict[str, Any]:
    base_n = _int_value(int(base_n), "mx_scale_base_n_invalid")
    k = _int_value(int(k), "mx_scale_k_invalid")
    dtype_bytes = _int_value(int(dtype_bytes), "mx_scale_dtype_bytes_invalid")
    if base_n <= 0 or k <= 0 or dtype_bytes <= 0:
        raise HardwareError("mx_scale_shape_invalid")
    weight_bytes = align_up(k * base_n * dtype_bytes, alignment_bytes)
    scale_offset_bytes = int(round(float(scale_offset_kib) * KIB))
    scale_bytes = align_up(int(scale_size_bytes), alignment_bytes)
    slot_bytes = int(round(float(slot_size_kib) * KIB))
    check = validate_buffer_allocations(
        [
            {"name": "L1B_weight_tile", "buffer": "L1", "offset_bytes": 0, "size_bytes": weight_bytes},
            {"name": "L1B_mx_scale", "buffer": "L1", "offset_bytes": scale_offset_bytes, "size_bytes": scale_bytes},
        ],
        {"L1": slot_size_kib},
        alignment_bytes=alignment_bytes,
    )
    overlap_bytes = sum(item["overlap_bytes"] for item in check["overlaps"])
    total_required = max(weight_bytes, scale_offset_bytes + scale_bytes)
    recommended = align_up(weight_bytes, alignment_bytes) // KIB
    result = {
        "base_n": base_n,
        "k": k,
        "dtype_bytes": dtype_bytes,
        "weight_tile_bytes": weight_bytes,
        "weight_tile_kib": weight_bytes / KIB,
        "scale_offset_kib": float(scale_offset_kib),
        "scale_size_bytes": scale_bytes,
        "slot_size_kib": float(slot_size_kib),
        "overlap_bytes": overlap_bytes,
        "slot_overflow_bytes": max(0, total_required - slot_bytes),
        "recommended_scale_offset_kib": float(recommended),
        "status": "ok" if overlap_bytes == 0 and total_required <= slot_bytes else "violation",
        "allocation_check": check,
        "note": "MX scale region must start after the full K*baseN weight tile.",
    }
    if overlap_bytes:
        result["diagnostic"] = "mx_scale_overwritten_by_weight_tile"
    elif total_required > slot_bytes:
        result["diagnostic"] = "mx_scale_layout_exceeds_l1b_slot"
    return result


