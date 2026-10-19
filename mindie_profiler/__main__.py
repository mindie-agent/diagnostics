"""CLI for the compact MindIE profiler core.

[Role]
  Parse explicit step/time windows, rank filters, source mode, and optional
  scene metadata, then run the analysis and print or atomically write C7 JSON.

[Invariants]
  * No command infers a step from kernel rows; absent windows are a usage
    error unless --full-range is explicit.
  * The source is analyzed in place; nothing is copied off this machine.
  * Errors are reported with a non-zero exit code and actionable message.

[Timeouts & Locks]
  Delegates bounded read budgets to ``analyze_profile``.

[Side-Effect Policy]
  Reads sources; writes only the requested --output JSON.  No network.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from . import __version__
from .analyze import analyze_profile
from .events import ProfilerError, ProfilerUsageError
from .segment import load_steps_file, parse_step_windows, parse_windows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mindie_profiler",
        description="Explicit-step Ascend profiler DB/CSV accounting with C7 JSON output.",
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"mindie_profiler/{__version__}")
    parser.add_argument("--root", required=True, help="profiling root, rank dir, DB, or kernel_details.csv")
    parser.add_argument("--rank", action="append", default=None, help="rank id or numeric rank index (repeatable)")
    parser.add_argument(
        "--step-window",
        "--step",
        dest="step_window",
        action="append",
        default=[],
        metavar="STEP=START:END",
        help="explicit step window in --unit (repeatable; kernel rows have no step id)",
    )
    parser.add_argument(
        "--window",
        "--time-window",
        dest="window",
        action="append",
        default=[],
        metavar="START:END",
        help="unlabelled explicit time window in --unit (repeatable)",
    )
    parser.add_argument("--steps-file", default=None, help="JSON steps file with explicit step windows")
    parser.add_argument("--full-range", action="store_true", help="explicitly analyze each source's full captured range")
    parser.add_argument("--unit", choices=("us", "ns", "ms"), default="us", help="unit for window bounds (default us)")
    parser.add_argument("--source", choices=("auto", "db", "csv"), default="auto", help="source preference (default db when present)")
    parser.add_argument("--output", default=None, help="write C7 JSON here (atomic); otherwise print to stdout")
    parser.add_argument("--top", type=int, default=8, help="number of hotspot kernels (default 8)")
    parser.add_argument("--max-rows", type=int, default=1_000_000, help="per-window row budget (default 1,000,000)")
    parser.add_argument("--query-timeout-s", type=float, default=20.0, help="sqlite query deadline (default 20s)")
    parser.add_argument("--csv-timeout-s", type=float, default=30.0, help="CSV scan deadline (default 30s)")
    parser.add_argument("--no-hash", action="store_true", help="skip sha256 (reported as a limitation)")
    parser.add_argument("--mode", choices=("inference", "training", "rl"), default=None, help="explicit workload mode (not inferred)")
    parser.add_argument("--framework", default=None, help="scene framework, e.g. vllm-ascend (not inferred)")
    parser.add_argument("--phase", default=None, help="scene phase, e.g. prefill/decode (not inferred)")
    parser.add_argument("--tp", type=int, default=None, help="tensor parallel size (not inferred)")
    parser.add_argument("--dp", type=int, default=None, help="data parallel size (not inferred)")
    parser.add_argument("--graph-mode", default=None, help="graph mode, e.g. piecewise_graph (not inferred)")
    parser.add_argument("--clock-domain", default=None, help="clock domain, e.g. syscnt_aligned (not inferred)")
    parser.add_argument("--input-signature", default=None, help="identity of the profiled workload and inputs")
    return parser


def _collect_windows(args: argparse.Namespace):
    windows = []
    windows.extend(parse_step_windows(args.step_window, unit=args.unit))
    windows.extend(parse_windows(args.window, unit=args.unit))
    if args.steps_file:
        windows.extend(load_steps_file(args.steps_file, unit=args.unit))
    seen: set[int] = set()
    for window in windows:
        if window.step is None:
            continue
        if window.step in seen:
            raise ProfilerUsageError(f"step {window.step} supplied more than once across CLI/file")
        seen.add(window.step)
    return windows


def _scene(args: argparse.Namespace) -> dict:
    scene = {
        "mode": args.mode,
        "framework": args.framework,
        "phase": args.phase,
        "tp": args.tp,
        "dp": args.dp,
        "graph_mode": args.graph_mode,
        "clock_domain": args.clock_domain,
        "input_signature": args.input_signature,
    }
    return {key: value for key, value in scene.items() if value is not None}


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "compare":
        from .cli_compare import main as compare_main

        return compare_main(raw[1:])
    if raw and raw[0] == "analyze":
        raw = raw[1:]
    parser = build_parser()
    args = parser.parse_args(raw)
    try:
        windows = _collect_windows(args)
        report = analyze_profile(
            args.root,
            ranks=args.rank,
            windows=windows,
            full_range=args.full_range,
            source=args.source,
            top_n=args.top,
            hash_sources=not args.no_hash,
            max_rows=args.max_rows,
            query_timeout_s=args.query_timeout_s,
            csv_timeout_s=args.csv_timeout_s,
            scene=_scene(args),
            output=args.output,
        )
    except ProfilerError as exc:
        print(f"mindie_profiler: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"mindie_profiler: filesystem error: {exc}", file=sys.stderr)
        return 2
    if args.output:
        summary = {
            "output": args.output,
            "schema_version": report["schema_version"],
            "ranks": report["step_accounting"]["rank_scope"],
            "steady_steps": report["step_accounting"]["steady_steps"],
            "query_elapsed_ms": report["source_identity"]["query_elapsed_ms"],
            "limitations": [item["code"] for item in report["limitations"]],
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
