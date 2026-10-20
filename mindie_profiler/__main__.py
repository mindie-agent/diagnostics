"""Progressive CLI for the compact MindIE profiler core.

[Role]
  Expose five explicit subcommands — ``inventory``, ``analyze``,
  ``correlate``, ``compare``, and ``sql`` — plus a legacy no-subcommand
  ``analyze`` form.  Parse explicit windows, ranks, source preferences, and
  bounded query limits, then print or atomically write JSON.

[Invariants]
  * No command infers a step from kernel rows; absent windows are a usage
    error unless ``--full-range`` is explicit.
  * ``analyze`` computes content SHA256 only with ``--hash``; the default
    identity is path+size_bytes+mtime_ns+schema_version.
  * ``sql`` is read-only and bounded; a limit overflow is an error, never a
    silent truncation.
  * Errors are reported with a non-zero exit code and an actionable message.

[Timeouts & Locks]
  Delegates bounded budgets to ``analyze_profile``, ``inventory_profile``,
  ``correlate_profile``, and ``run_sql``.

[Side-Effect Policy]
  Reads sources; writes only the requested ``--output`` JSON.  No network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .analyze import analyze_profile
from .events import ProfilerError, ProfilerUsageError
from .segment import load_steps_file, parse_step_windows, parse_windows

SUBCOMMANDS = ("inventory", "analyze", "correlate", "compare", "sql")


def _add_analyze_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", required=True, help="profiling root, rank dir, DB, or kernel_details.csv")
    parser.add_argument("--rank", action="append", default=None, help="rank id or numeric rank index (repeatable)")
    parser.add_argument(
        "--step-window", "--step", dest="step_window", action="append", default=[],
        metavar="STEP=START:END", help="explicit step window in --unit (repeatable)",
    )
    parser.add_argument(
        "--window", "--time-window", dest="window", action="append", default=[],
        metavar="START:END", help="unlabelled explicit time window in --unit (repeatable)",
    )
    parser.add_argument("--steps-file", default=None, help="JSON steps file with explicit step windows")
    parser.add_argument("--full-range", action="store_true", help="explicitly analyze each source's full captured range")
    parser.add_argument("--unit", choices=("us", "ns", "ms"), default="us", help="unit for window bounds (default us)")
    parser.add_argument("--source", choices=("auto", "db", "csv"), default="auto", help="source preference")
    parser.add_argument("--output", default=None, help="write JSON here (atomic); otherwise print to stdout")
    parser.add_argument("--top", type=int, default=8, help="number of hotspot rows (default 8)")
    parser.add_argument(
        "--group-by", choices=("variant", "kernel"), default="variant",
        help="hotspot grouping: shape/dtype/format variant (default) or kernel name",
    )
    parser.add_argument("--max-rows", type=int, default=1_000_000, help="per-window row budget")
    parser.add_argument("--query-timeout-s", type=float, default=20.0, help="sqlite query deadline")
    parser.add_argument("--csv-timeout-s", type=float, default=30.0, help="CSV scan deadline")
    parser.add_argument("--hash", action="store_true", help="compute content SHA256 for source identity")
    parser.add_argument("--no-hash", dest="hash", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--request-wall-ms", type=float, default=None, help="external request wall time (measured)")
    parser.add_argument("--mode", choices=("inference", "training", "rl"), default=None, help="explicit workload mode")
    parser.add_argument("--framework", default=None, help="scene framework, e.g. vllm-ascend")
    parser.add_argument("--phase", default=None, help="scene phase, e.g. prefill/decode")
    parser.add_argument("--tp", type=int, default=None, help="tensor parallel size")
    parser.add_argument("--dp", type=int, default=None, help="data parallel size")
    parser.add_argument("--graph-mode", default=None, help="graph mode, e.g. piecewise_graph")
    parser.add_argument("--clock-domain", default=None, help="clock domain, e.g. syscnt_aligned")
    parser.add_argument("--input-signature", default=None, help="identity of the profiled workload and inputs")


def build_parser() -> argparse.ArgumentParser:
    """Legacy analyze parser (kept for import compatibility)."""

    parser = argparse.ArgumentParser(
        prog="python -m mindie_profiler analyze",
        description="Explicit-step Ascend profiler accounting with C7 JSON output.",
        allow_abbrev=False,
    )
    _add_analyze_arguments(parser)
    return parser


def _inventory_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mindie_profiler inventory",
        description="List ranks, source candidates, tables, native step/MSTX windows, and capture span.",
        allow_abbrev=False,
    )
    parser.add_argument("--root", required=True, help="profiling root, rank dir, or source file")
    parser.add_argument("--output", default=None, help="write JSON here (atomic); otherwise print to stdout")
    parser.add_argument("--max-dirs", type=int, default=4096, help="bounded directory walk (default 4096)")
    parser.add_argument("--timeout-s", type=float, default=5.0, help="per-DB inventory deadline")
    return parser


def _correlate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mindie_profiler correlate",
        description="Join Host/Runtime stacks to device TASKs and inspect device gaps.",
        allow_abbrev=False,
    )
    parser.add_argument("--root", required=True, help="profiling root or rank dir with a DB")
    parser.add_argument("--rank", action="append", default=None, help="rank id or numeric rank index")
    parser.add_argument("--window", default=None, metavar="START:END", help="explicit correlation window in --unit")
    parser.add_argument("--full-range", action="store_true", help="use the DB's full TASK span")
    parser.add_argument("--unit", choices=("us", "ns", "ms"), default="us")
    parser.add_argument("--kernel-name", default=None, help="case-insensitive hotspot kernel name pattern")
    parser.add_argument("--min-gap-ms", type=float, default=None, help="only report device gaps at least this large")
    parser.add_argument("--max-rows", type=int, default=5000, help="per-table row cap")
    parser.add_argument("--limit", type=int, default=200, help="correlated call output cap")
    parser.add_argument("--timeout-s", type=float, default=20.0, help="sqlite deadline")
    parser.add_argument("--output", default=None, help="write JSON here (atomic); otherwise print to stdout")
    return parser


def _sql_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mindie_profiler sql",
        description="Read-only, row/timeout bounded SQL against one profiler DB.",
        allow_abbrev=False,
    )
    parser.add_argument("--db", required=True, help="profiler .db path")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help="one read-only SELECT/WITH/EXPLAIN/VALUES statement")
    group.add_argument("--query-file", help="file containing one read-only SQL statement")
    parser.add_argument("--limit", type=int, default=50, help="maximum rows; overflow is an error")
    parser.add_argument("--timeout-s", type=float, default=10.0, help="query deadline")
    parser.add_argument("--no-resolve-strings", dest="resolve_strings", action="store_false", help="skip STRING_IDS companions")
    parser.add_argument("--output", default=None, help="write JSON here (atomic); otherwise print to stdout")
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


def _emit(payload: dict, output: str | None, *, summary: dict | None = None) -> int:
    if output:
        from .report import write_report

        write_report(payload, output)
        print(json.dumps(summary or {"output": output}, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _run_analyze(argv: Sequence[str]) -> int:
    args = build_parser().parse_args(argv)
    windows = _collect_windows(args)
    report = analyze_profile(
        args.root,
        ranks=args.rank,
        windows=windows,
        full_range=args.full_range,
        source=args.source,
        top_n=args.top,
        hash_sources=bool(args.hash),
        max_rows=args.max_rows,
        query_timeout_s=args.query_timeout_s,
        csv_timeout_s=args.csv_timeout_s,
        group_by=args.group_by,
        request_wall_ms=args.request_wall_ms,
        scene=_scene(args),
        output=args.output,
    )
    summary = {
        "output": args.output,
        "schema_version": report["schema_version"],
        "ranks": report["step_accounting"]["rank_scope"],
        "steady_steps": report["step_accounting"]["steady_steps"],
        "query_elapsed_ms": report["source_identity"]["query_elapsed_ms"],
        "hotspot_grouping": report.get("hotspot_grouping"),
        "limitations": [item["code"] for item in report["limitations"]],
    }
    if args.output:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    return _emit(report, None)


def _run_inventory(argv: Sequence[str]) -> int:
    from .inventory import inventory_profile

    args = _inventory_parser().parse_args(argv)
    report = inventory_profile(args.root, max_dirs=args.max_dirs, timeout_s=args.timeout_s)
    summary = {
        "output": args.output,
        "schema_version": report["schema_version"],
        "rank_count": report["rank_count"],
        "limitations": [item["code"] for item in report["limitations"]],
    }
    if args.output:
        return _emit(report, args.output, summary=summary)
    return _emit(report, None)


def _run_correlate(argv: Sequence[str]) -> int:
    from .correlate import correlate_profile

    args = _correlate_parser().parse_args(argv)
    windows = parse_windows([args.window], unit=args.unit) if args.window else []
    if len(windows) > 1:
        raise ProfilerUsageError("correlate accepts one --window")
    report = correlate_profile(
        args.root,
        window=windows[0] if windows else None,
        full_range=args.full_range,
        ranks=args.rank,
        kernel_name=args.kernel_name,
        min_gap_ms=args.min_gap_ms,
        max_rows=args.max_rows,
        timeout_s=args.timeout_s,
        call_limit=args.limit,
    )
    summary = {
        "output": args.output,
        "schema_version": report["schema_version"],
        "rank_id": report["rank_id"],
        "matched_task_count": report["host"]["matched_task_count"],
        "unmatched_ratio": report["host"]["unmatched_ratio"],
        "ambiguous_global_task_ids": report["aclgraph"]["ambiguous_count"],
    }
    if args.output:
        return _emit(report, args.output, summary=summary)
    return _emit(report, None)


def _run_sql(argv: Sequence[str]) -> int:
    from .sql import run_sql

    args = _sql_parser().parse_args(argv)
    query = args.query
    if args.query_file:
        try:
            query = Path(args.query_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise ProfilerUsageError(f"cannot read query file: {exc}") from exc
    result = run_sql(
        args.db,
        str(query),
        limit=args.limit,
        timeout_s=args.timeout_s,
        resolve_strings=bool(args.resolve_strings),
    )
    summary = {
        "output": args.output,
        "row_count": result["row_count"],
        "row_limit": result["row_limit"],
        "elapsed_ms": result["elapsed_ms"],
    }
    if args.output:
        return _emit(result, args.output, summary=summary)
    return _emit(result, None)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in {"--version", "-V"}:
        print(f"mindie_profiler/{__version__}")
        return 0
    command = raw[0] if raw and raw[0] in SUBCOMMANDS else "analyze"
    rest = raw[1:] if raw and raw[0] in SUBCOMMANDS else raw
    try:
        if command == "compare":
            from .cli_compare import main as compare_main

            return compare_main(rest)
        if command == "inventory":
            return _run_inventory(rest)
        if command == "correlate":
            return _run_correlate(rest)
        if command == "sql":
            return _run_sql(rest)
        return _run_analyze(rest)
    except ProfilerError as exc:
        print(f"mindie_profiler: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"mindie_profiler: filesystem error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
