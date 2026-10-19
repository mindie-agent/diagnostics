"""Command-line entry for comparing two exported profile runs.

[Role]
Read C7 summaries or analyze two local roots with their own explicit step
windows, then produce the same comparison receipt as the Python API.
[Invariants]
Each run owns its timestamps and scene metadata. A missing workload identity
or incompatible measurement remains not_comparable in the receipt.
[Timeouts & Locks]
The analyzer applies its per-query budgets. No lock or network request is used.
[Side-Effect Policy]
Read source artifacts in place; atomically write only the requested output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .analyze import analyze_profile
from .compare import compare_profiles
from .module_trace import build_module_table
from .events import ProfilerError, ProfilerUsageError
from .report import write_report
from .segment import load_steps_file


def _json_object(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProfilerUsageError(f"{path}: expected a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mindie-profiler compare",
        description="Compare C7 summaries or two roots with separate step and scene files.",
        allow_abbrev=False,
    )
    for side in ("baseline", "candidate"):
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument(f"--{side}-summary", help="existing C7 JSON summary")
        source.add_argument(f"--{side}-root", help="profiling root, DB, or kernel_details.csv")
        parser.add_argument(f"--{side}-steps-file", help="explicit step timestamps for this run")
        parser.add_argument(f"--{side}-scene-file", help="scene metadata, including input_signature")
        parser.add_argument(f"--{side}-structure-file", help="actual operator/module structure JSON")
    parser.add_argument("--rank", action="append", default=None, help="rank filter for both roots")
    parser.add_argument("--source", choices=("auto", "db", "csv"), default="auto")
    parser.add_argument("--unit", choices=("us", "ns", "ms"), default="us")
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    parser.add_argument("--query-timeout-s", type=float, default=20.0)
    parser.add_argument("--csv-timeout-s", type=float, default=30.0)
    parser.add_argument("--output", help="atomically write comparison JSON")
    return parser


def _structure_object(path: str) -> dict:
    value = _json_object(path)
    structure = value.get("structure") if isinstance(value.get("structure"), dict) else value
    if not isinstance(structure, dict):
        raise ProfilerUsageError(f"{path}: expected a structure object")
    return structure


def _read_side(args: argparse.Namespace, side: str) -> dict:
    summary = getattr(args, f"{side}_summary")
    steps = getattr(args, f"{side}_steps_file")
    scene = getattr(args, f"{side}_scene_file")
    structure_file = getattr(args, f"{side}_structure_file")
    if summary:
        if steps or scene:
            raise ProfilerUsageError(f"--{side}-summary already contains its step and scene evidence")
        report = _json_object(summary)
    else:
        if not steps:
            raise ProfilerUsageError(f"--{side}-root requires --{side}-steps-file")
        report = analyze_profile(
            getattr(args, f"{side}_root"),
            ranks=args.rank,
            windows=load_steps_file(steps, unit=args.unit),
            source=args.source,
            scene=_json_object(scene) if scene else None,
            max_rows=args.max_rows,
            query_timeout_s=args.query_timeout_s,
            csv_timeout_s=args.csv_timeout_s,
        )
    if structure_file:
        report["structure"] = _structure_object(structure_file)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        baseline = _read_side(args, "baseline")
        candidate = _read_side(args, "candidate")
        result = compare_profiles(baseline, candidate, require_binding=True)
        baseline_structure = baseline.get("structure")
        candidate_structure = candidate.get("structure")
        if isinstance(baseline_structure, dict) and isinstance(candidate_structure, dict):
            module_table = build_module_table(baseline_structure, candidate_structure)
            if module_table["modules"]:
                result.setdefault("tables", {})["modules"] = module_table
                result["module_scope"] = module_table["scope"]
        if args.output:
            write_report(result, args.output)
            print(json.dumps({"output": args.output, "status": result["status"]}))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ProfilerError, OSError, ValueError) as exc:
        print(f"mindie-profiler compare: {exc}", file=sys.stderr)
        return 2
    return 0 if result.get("comparable") else 3
