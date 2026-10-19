"""P2 framework-log memory fact parsing.

[Role]
  Extract documented vLLM/Ascend startup-log memory facts for P2 attribution,
  preserving the log's explicit GB/GiB units and Available-vs-Current KV
  semantics.

[Invariants]
  * Available KV cache memory is a budget; Current KV cache memory is the
    actual allocation and is parsed separately.
  * GB is decimal 1e9 and GiB is binary 1024^3; units are never guessed from
    magnitude.
  * Only explicit log facts become values; missing lines stay absent.
  * Raw matched text and source lines are retained for precision/audit.

[Timeouts & Locks]
  Pure string parsing over supplied bounded log text.

[Side-Effect Policy]
  No I/O or mutation.
"""

from __future__ import annotations

import re
from typing import Any

_WEIGHT_RE = re.compile(r"Loading model weights took\s+([\d.]+)\s*(GB|GiB)")
_KV_AVAIL_RE = re.compile(r"Available KV cache memory:\s+([\d.]+)\s*(GB|GiB)")
_KV_CURRENT_RE = re.compile(r"Current KV cache memory:\s+([\d.]+)\s*(GB|GiB)")
_KV_TOKENS_RE = re.compile(r"(?:GPU|NPU) KV cache size:\s+([\d,]+)\s*tokens")
_GRAPH_RE = re.compile(r"Graph capturing finished in\s+(\d+)\s*secs,\s*took\s+([\d.]+)\s*(GB|GiB)")
_GPU_UTIL_RE = re.compile(r"gpu_memory_utilization['\"]?\s*[:=]\s*([\d.]+)")
_STARTUP_RE = re.compile(r"Free memory on device\s*\(([\d.]+)/([\d.]+)\s*(GB|GiB)\)")
_DESIRED_RE = re.compile(r"Desired GPU memory utilization is\s*\(([\d.]+),\s*([\d.]+)\s*(GB|GiB)\)")
_USAGE_RE = re.compile(
    r"Actual usage:\s*([\d.]+)\s*(GB|GiB) for weights,\s*"
    r"([\d.]+)\s*(GB|GiB) for peak activation,\s*"
    r"([\d.]+)\s*(GB|GiB) for non-torch memory,\s*"
    r"([\d.]+)\s*(GB|GiB) for NPU graph memory"
)
_KV_MEMORY_RE = re.compile(r"--kv-cache-memory=(\d+)`?\s*\(([\d.]+)\s*(GB|GiB)\)")
_WARMUP_RE = re.compile(
    r"After warmup: torch reserved memory\s+([\d.]+)\s*(GB|GiB),\s*"
    r"torch allocated memory\s+([\d.]+)\s*(GB|GiB)"
)


def _suffix(unit: str) -> str:
    return "gib" if unit == "GiB" else "gb"


def _set(info: dict[str, Any], key: str, value: float, unit: str, raw: str) -> None:
    info[key] = value
    info.setdefault("units", {})[key] = unit
    info.setdefault("raw", {})[key] = f"{raw} {unit}"


def parse_vllm_logs(log_text: str) -> dict[str, Any]:
    """Extract explicit P2 memory facts, preserving units and raw text."""
    info: dict[str, Any] = {}
    graphs: list[tuple[float, str]] = []
    source_lines: list[str] = []

    def note(line: str) -> None:
        stripped = line.strip()
        if stripped and stripped not in source_lines:
            source_lines.append(stripped)

    for line in str(log_text).splitlines():
        match = _WEIGHT_RE.search(line)
        if match:
            unit = match.group(2)
            _set(info, f"weights_{_suffix(unit)}", float(match.group(1)), unit, match.group(1))
            info["weights_unit"] = unit
            note(line)
        match = _KV_AVAIL_RE.search(line)
        if match:
            unit = match.group(2)
            _set(info, f"kv_cache_available_{_suffix(unit)}", float(match.group(1)), unit, match.group(1))
            info["kv_cache_available_is_budget"] = True
            note(line)
        match = _KV_CURRENT_RE.search(line)
        if match:
            unit = match.group(2)
            _set(info, f"kv_cache_current_{_suffix(unit)}", float(match.group(1)), unit, match.group(1))
            info["kv_cache_current_is_actual"] = True
            note(line)
        match = _KV_TOKENS_RE.search(line)
        if match:
            info["kv_cache_tokens"] = int(match.group(1).replace(",", ""))
            note(line)
        match = _GRAPH_RE.search(line)
        if match:
            graphs.append((float(match.group(2)), match.group(3)))
            note(line)
        match = _GPU_UTIL_RE.search(line)
        if match:
            info["gpu_memory_utilization"] = float(match.group(1))
            note(line)
        match = _STARTUP_RE.search(line)
        if match:
            unit = match.group(3)
            suffix = _suffix(unit)
            _set(info, f"startup_free_{suffix}", float(match.group(1)), unit, match.group(1))
            _set(info, f"startup_total_{suffix}", float(match.group(2)), unit, match.group(2))
            note(line)
        match = _DESIRED_RE.search(line)
        if match:
            info["desired_gpu_memory_utilization"] = float(match.group(1))
            unit = match.group(3)
            _set(info, f"desired_budget_{_suffix(unit)}", float(match.group(2)), unit, match.group(2))
            note(line)
        match = _USAGE_RE.search(line)
        if match:
            unit_1, unit_2, unit_3, unit_4 = match.group(2), match.group(4), match.group(6), match.group(8)
            if unit_1 == unit_2 == unit_3 == unit_4:
                suffix = _suffix(unit_1)
                _set(info, f"weights_usage_{suffix}", float(match.group(1)), unit_1, match.group(1))
                _set(info, f"peak_activation_{suffix}", float(match.group(3)), unit_2, match.group(3))
                _set(info, f"non_torch_memory_{suffix}", float(match.group(5)), unit_3, match.group(5))
                _set(info, f"npu_graph_memory_{suffix}", float(match.group(7)), unit_4, match.group(7))
            note(line)
        for match in _KV_MEMORY_RE.finditer(line):
            info.setdefault("kv_cache_memory_recommendations", []).append({
                "bytes": int(match.group(1)),
                "value": float(match.group(2)),
                "unit": match.group(3),
            })
            note(line)
        match = _WARMUP_RE.search(line)
        if match:
            unit_1, unit_2 = match.group(2), match.group(4)
            _set(info, f"torch_reserved_{_suffix(unit_1)}", float(match.group(1)), unit_1, match.group(1))
            _set(info, f"torch_allocated_{_suffix(unit_2)}", float(match.group(3)), unit_2, match.group(3))
            note(line)

    if graphs:
        max_value, max_unit = max(graphs, key=lambda item: item[0])
        _set(info, f"graph_capture_{_suffix(max_unit)}", max_value, max_unit, str(max_value))
        info["graph_capture_count"] = len(graphs)
    recommendations = info.get("kv_cache_memory_recommendations")
    if isinstance(recommendations, list) and recommendations:
        info["kv_cache_memory_fit_bytes"] = recommendations[0]["bytes"]
        info["kv_cache_memory_fit_value"] = recommendations[0]["value"]
        info["kv_cache_memory_fit_unit"] = recommendations[0]["unit"]
        if len(recommendations) > 1:
            info["kv_cache_memory_full_bytes"] = recommendations[1]["bytes"]
            info["kv_cache_memory_full_value"] = recommendations[1]["value"]
            info["kv_cache_memory_full_unit"] = recommendations[1]["unit"]
    if source_lines:
        info["source_lines"] = source_lines
    return info


__all__ = ["parse_vllm_logs"]
