"""Acceptance-contract validators for matched profile and C2 evidence.

[Role]
Shared pure validation for caller-explicit windows, measurement timing
summaries, and manifest SHA provenance used by the standalone acceptance driver.

[Invariants]
- Only finite numeric windows with positive duration are accepted.
- Timing count/total/min/max/mean must agree with each other.
- Provenance is accepted only when every declared file exists and its SHA256
  matches exactly; missing provenance is a failure, not an empty success.

[Timeouts & Locks]
Pure in-memory/file-hash validation; no network, subprocess, lock, or sleep.

[Side-Effect Policy]
Reads only caller-named files; no writes or source mutation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


def _valid_window(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Mapping):
        return None
    start, end = value.get("start_ts"), value.get("end_ts")
    if not isinstance(start, (int, float)) or isinstance(start, bool):
        return None
    if not isinstance(end, (int, float)) or isinstance(end, bool):
        return None
    if float(end) <= float(start):
        return None
    return float(start), float(end)


def _close(left: Any, right: Any, *, rel: float = 1e-6) -> bool:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    scale = max(abs(float(left)), abs(float(right)), 1.0)
    return abs(float(left) - float(right)) <= rel * scale


def _valid_timing(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    count = value.get("count")
    total, minimum, maximum, mean = (
        value.get("total_ms"), value.get("min_ms"),
        value.get("max_ms"), value.get("mean_ms"),
    )
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        return False
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool)
               for item in (total, minimum, maximum, mean)):
        return False
    if float(total) <= 0 or float(minimum) < 0 or float(maximum) < float(minimum):
        return False
    if not float(minimum) <= float(mean) <= float(maximum):
        return False
    return _close(mean, float(total) / count)


def _valid_measurement(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and _valid_window(value.get("measurement_window")) is not None
        and isinstance(value.get("duration_ms"), (int, float))
        and not isinstance(value.get("duration_ms"), bool)
        and float(value["duration_ms"]) > 0
        and isinstance(value.get("sample_count"), int)
        and not isinstance(value.get("sample_count"), bool)
        and value["sample_count"] > 0
        and _valid_timing(value.get("timing"))
        and value["timing"]["count"] == value["sample_count"]
    )


def validate_module_table(module_table: Mapping[str, Any] | None) -> tuple[bool, list[str]]:
    if not isinstance(module_table, Mapping) or not module_table.get("modules"):
        return False, ["module_table_missing"]
    reasons = []
    if module_table.get("status") != "comparable":
        reasons.append("module_table_not_comparable")
    for row in module_table["modules"]:
        module_id = row.get("module_id")
        if row.get("status") != "compared":
            reasons.append(f"module_not_compared:{module_id}")
        if row.get("baseline_mapping_coverage") != 1.0:
            reasons.append(f"baseline_mapping_incomplete:{module_id}")
        if row.get("candidate_mapping_coverage") != 1.0:
            reasons.append(f"candidate_mapping_incomplete:{module_id}")
    return not reasons, reasons


def _verify_provenance(manifest: Mapping[str, Any]) -> tuple[bool, list[str]]:
    provenance = manifest.get("provenance")
    files = provenance.get("source_files") if isinstance(provenance, Mapping) else None
    if not isinstance(files, Mapping) or not files:
        return False, ["provenance_source_files_missing"]
    failures: list[str] = []
    for name, item in files.items():
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            failures.append(f"provenance_entry_invalid:{name}")
            continue
        path = Path(item["path"])
        if not path.is_file():
            failures.append(f"provenance_file_missing:{name}")
        elif not isinstance(item.get("sha256"), str) or _hash_bytes(path) != item["sha256"]:
            failures.append(f"provenance_hash_mismatch:{name}")
    return not failures, failures


def _hash_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
