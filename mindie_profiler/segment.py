"""Explicit rank discovery and step/time-window selection.

[Role]
  Discover rank directories and their DB/CSV sources, parse caller-supplied
  step windows and plain time windows, and resolve explicit rank filters.

[Invariants]
  * No step is inferred from kernel order, timestamps, or a CSV header.  A
    step exists only when the caller supplies ``STEP=START:END`` (or a steps
    file); an unlabelled window stays unlabelled.
  * Units are explicit at the CLI/API boundary and converted once to integer
    nanoseconds.  Report values are milliseconds.
  * Duplicate/ambiguous rank ids are rejected rather than silently paired.

[Timeouts & Locks]
  Discovery is a bounded filesystem walk (no symlink following); no locks.

[Side-Effect Policy]
  Read-only directory listing.  No files are created or modified.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence

from .events import ProfilerUsageError, Window

_DB_GLOB = "ascend_pytorch_profiler*.db"
_OUTPUT_DIR = "ASCEND_PROFILER_OUTPUT"
_RANK_PATTERNS = (
    re.compile(r"(dp\d+_pp\d+_tp\d+_dcp\d+_ep\d+_rank\d+)", re.IGNORECASE),
    re.compile(r"(rank[_-]?\d+)", re.IGNORECASE),
    re.compile(r"(device[_-]?\d+)", re.IGNORECASE),
)
_INDEX_RE = re.compile(r"(?:rank|device)[_-]?(\d+)", re.IGNORECASE)
_SAFE_RE = re.compile(r"[^A-Za-z0-9_]+")


@dataclass(frozen=True, slots=True)
class RankSource:
    rank_id: str
    rank_index: int | None
    rank_dir: str
    db_path: str | None
    csv_path: str | None
    kind: str | None
    notes: tuple[str, ...] = ()


def infer_rank_id(rank_dir: Path, ordinal: int = 0) -> str:
    text = rank_dir.name
    for pattern in _RANK_PATTERNS:
        match = pattern.search(text)
        if match:
            return _SAFE_RE.sub("_", match.group(1)).strip("_").lower()
    safe = _SAFE_RE.sub("_", text).strip("_").lower()
    return safe or f"rank_{ordinal}"


def infer_rank_index(rank_dir_text: str) -> int | None:
    matches = _INDEX_RE.findall(rank_dir_text)
    return int(matches[-1]) if matches else None


def _find_db(rank_dir: Path) -> tuple[Path | None, tuple[str, ...]]:
    found: list[Path] = []
    for base in (rank_dir, rank_dir / _OUTPUT_DIR):
        found.extend(base.glob(_DB_GLOB))
    notes: list[str] = []
    if not found:
        return None, ()
    found = sorted(set(found), key=lambda item: (-item.stat().st_mtime, item.name))
    if len(found) > 1:
        notes.append(f"{len(found)} profiler dbs in rank dir; using newest by mtime: {found[0].name}")
    return found[0], tuple(notes)


def _find_csv(rank_dir: Path) -> Path | None:
    for candidate in (rank_dir / "kernel_details.csv", rank_dir / _OUTPUT_DIR / "kernel_details.csv"):
        if candidate.is_file():
            return candidate
    return None


def _rank_dir_for_file(path: Path) -> Path:
    if path.parent.name == _OUTPUT_DIR:
        return path.parent.parent
    return path.parent


def _source_for_dir(
    rank_dir: Path,
    *,
    ordinal: int,
    source: str,
    explicit_file: Path | None = None,
) -> RankSource | None:
    rank_id = infer_rank_id(rank_dir, ordinal)
    rank_index = infer_rank_index(rank_dir.name)
    db_path, notes = _find_db(rank_dir)
    csv_path = _find_csv(rank_dir)
    if explicit_file is not None:
        if explicit_file.suffix == ".db":
            db_path, csv_path = explicit_file, None
        else:
            csv_path, db_path = explicit_file, None
    if source == "db":
        kind = "db" if db_path else None
    elif source == "csv":
        kind = "csv" if csv_path else None
    else:
        kind = "db" if db_path else ("csv" if csv_path else None)
    if kind is None:
        return None
    return RankSource(
        rank_id=rank_id,
        rank_index=rank_index,
        rank_dir=str(rank_dir.resolve()),
        db_path=str(db_path.resolve()) if db_path else None,
        csv_path=str(csv_path.resolve()) if csv_path else None,
        kind=kind,
        notes=notes,
    )


def discover_rank_sources(
    root: str | Path,
    *,
    source: str = "auto",
    max_dirs: int = 4096,
) -> list[RankSource]:
    if source not in {"auto", "db", "csv"}:
        raise ProfilerUsageError(f"unknown source mode {source!r}; expected auto/db/csv")
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise ProfilerUsageError(f"profiling root not found: {root_path}")
    if root_path.is_file():
        rank_dir = _rank_dir_for_file(root_path)
        item = _source_for_dir(rank_dir, ordinal=0, source=source, explicit_file=root_path)
        if item is None:
            raise ProfilerUsageError(f"no usable {source} source at {root_path}")
        return [item]
    root_path = root_path.resolve()
    if root_path.name == _OUTPUT_DIR:
        root_path = root_path.parent
    own_db, _ = _find_db(root_path)
    own_csv = _find_csv(root_path)
    if own_db or own_csv:
        item = _source_for_dir(root_path, ordinal=0, source=source)
        if item is None:
            raise ProfilerUsageError(f"no usable {source} source under {root_path}")
        return [item]

    discovered: list[Path] = []
    seen: set[Path] = set()
    walked = 0
    for current, dirnames, filenames in os.walk(root_path, followlinks=False):
        walked += 1
        if walked > max_dirs:
            raise ProfilerUsageError(f"rank discovery exceeded {max_dirs} directories under {root_path}")
        current_path = Path(current)
        has_db = any(name.startswith("ascend_pytorch_profiler") and name.endswith(".db") for name in filenames)
        has_csv = "kernel_details.csv" in filenames
        has_output = _OUTPUT_DIR in dirnames
        if has_db or has_csv:
            if current_path not in seen:
                discovered.append(current_path)
                seen.add(current_path)
            dirnames[:] = []
            continue
        if has_output:
            output_path = current_path / _OUTPUT_DIR
            output_names = set()
            try:
                output_names = {entry.name for entry in output_path.iterdir()}
            except OSError:
                pass
            if (
                "kernel_details.csv" in output_names
                or any(name.startswith("ascend_pytorch_profiler") and name.endswith(".db") for name in output_names)
            ):
                if current_path not in seen:
                    discovered.append(current_path)
                    seen.add(current_path)
                dirnames[:] = []
    items = [
        item
        for ordinal, rank_dir in enumerate(sorted(discovered, key=str))
        if (item := _source_for_dir(rank_dir, ordinal=ordinal, source=source)) is not None
    ]
    if not items:
        raise ProfilerUsageError(
            f"no kernel_details.csv or {_DB_GLOB} found under {root_path} for source={source}"
        )
    ids: dict[str, str] = {}
    for item in items:
        if item.rank_id in ids and ids[item.rank_id] != item.rank_dir:
            raise ProfilerUsageError(
                f"duplicate rank id {item.rank_id!r}: {ids[item.rank_id]} and {item.rank_dir}; "
                "pass a rank directory or a unique root"
            )
        ids[item.rank_id] = item.rank_dir
    return items


def resolve_ranks(sources: Sequence[RankSource], selected: Iterable[str] | None) -> list[RankSource]:
    if selected is None:
        return list(sources)
    wanted = [str(value).strip() for value in selected if str(value).strip()]
    if not wanted:
        return list(sources)
    resolved: list[RankSource] = []
    for token in wanted:
        exact = [item for item in sources if item.rank_id == token]
        if exact:
            resolved.extend(exact)
            continue
        token_match = re.fullmatch(r"(?:rank|device)[_-]?(\d+)", token, flags=re.IGNORECASE)
        if token_match:
            token = token_match.group(1)
        index_matches = [
            item for item in sources if item.rank_index is not None and str(item.rank_index) == token
        ]
        if len(index_matches) == 1:
            resolved.append(index_matches[0])
            continue
        if len(index_matches) > 1:
            raise ProfilerUsageError(
                f"rank {token!r} is ambiguous; choose one of {[item.rank_id for item in index_matches]}"
            )
        raise ProfilerUsageError(
            f"rank {token!r} not found; available={[item.rank_id for item in sources]}"
        )
    unique: dict[str, RankSource] = {item.rank_id: item for item in resolved}
    return list(unique.values())


def _to_ns(value: object, unit: str) -> int:
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ProfilerUsageError(f"non-numeric time bound {value!r}") from exc
    if unit == "ns":
        multiplier = Decimal(1)
    elif unit == "ms":
        multiplier = Decimal(1_000_000)
    else:
        multiplier = Decimal(1000)
    return int((number * multiplier).to_integral_value())


def _pair(text: str, unit: str) -> tuple[int, int]:
    part = text.strip()
    if ":" not in part:
        raise ProfilerUsageError(f"window {text!r} must be START:END")
    left, right = part.split(":", 1)
    return _to_ns(left, unit), _to_ns(right, unit)


def parse_step_windows(specs: Iterable[str], *, unit: str = "us") -> list[Window]:
    windows: list[Window] = []
    seen_steps: set[int] = set()
    for spec in specs:
        text = str(spec).strip()
        if not text:
            continue
        if "=" not in text:
            raise ProfilerUsageError(
                f"step window {text!r} must be STEP=START:END (microseconds by default)"
            )
        step_text, range_text = text.split("=", 1)
        try:
            step = int(step_text)
        except ValueError as exc:
            raise ProfilerUsageError(f"step window {text!r} has a non-integer step") from exc
        if step in seen_steps:
            raise ProfilerUsageError(f"step {step} supplied more than once; merge the window explicitly")
        seen_steps.add(step)
        start_ns, end_ns = _pair(range_text, unit)
        windows.append(Window(start_ns=start_ns, end_ns=end_ns, step=step, label=f"step_{step}"))
    return windows


def parse_windows(specs: Iterable[str], *, unit: str = "us") -> list[Window]:
    windows: list[Window] = []
    for index, spec in enumerate(specs):
        text = str(spec).strip()
        if not text:
            continue
        start_ns, end_ns = _pair(text, unit)
        windows.append(Window(start_ns=start_ns, end_ns=end_ns, step=None, label=f"window_{index}"))
    return windows


def load_steps_file(path: str | Path, *, unit: str = "us") -> list[Window]:
    source = Path(path)
    try:
        doc: Any = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProfilerUsageError(f"steps file not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ProfilerUsageError(f"{source.name}: invalid JSON: {exc}") from exc
    records = doc.get("steps") if isinstance(doc, dict) else doc
    if not isinstance(records, list) or not records:
        raise ProfilerUsageError(f"{source.name}: expected a non-empty 'steps' list")
    file_unit = str(doc.get("time_unit", unit)) if isinstance(doc, dict) else unit
    windows: list[Window] = []
    seen: set[int] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ProfilerUsageError(f"{source.name}: every step must be an object")
        try:
            step = int(record["step"])
            if "start_us" in record or "end_us" in record:
                start = record["start_us"]
                end = record["end_us"]
                record_unit = "us"
            else:
                start = record["start"]
                end = record["end"]
                record_unit = file_unit
        except (KeyError, TypeError, ValueError) as exc:
            raise ProfilerUsageError(f"{source.name}: step needs step/start_us/end_us: {record!r}") from exc
        if step in seen:
            raise ProfilerUsageError(f"{source.name}: duplicate step {step}")
        seen.add(step)
        windows.append(
            Window(
                start_ns=_to_ns(start, record_unit),
                end_ns=_to_ns(end, record_unit),
                step=step,
                label=f"step_{step}",
            )
        )
    return windows


def validate_selection(windows: Sequence[Window], *, full_range: bool) -> None:
    if not windows and not full_range:
        raise ProfilerUsageError(
            "an explicit --step-window, --window, or --full-range is required; "
            "kernel detail rows carry no step id, so this tool will not infer one"
        )
    if windows and full_range:
        raise ProfilerUsageError("--full-range cannot be combined with explicit windows")
