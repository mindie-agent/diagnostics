#!/usr/bin/env python3
"""Live T8.4 acceptance driver: online window and real framework capture.

This driver never starts a service and never fabricates an HTTP server.  It
operates on an already-running framework-injected service for the online mode,
or launches the standalone real torch_npu workload in a subprocess for the
framework/memory modes.

Exact online recipe on a host where the service already runs:

    python tests/live_profiler_collection_acceptance.py --mode online-cycle \
        --endpoint http://127.0.0.1:8000 \
        --execution-id <execution> \
        --profile-dir /path/on/host/vllm_profile \
        --receipt .mindie-003-wave4/online-window.json \
        --artifact-root /path/on/host/vllm_profile \
        --expected-ranks 1 \
        --workload-command "python /path/to/inference_wave.py"

Exact framework recipe (HostQueue, real NPU):

    python tests/live_profiler_collection_acceptance.py --mode framework \
        --run-id t8.4-baseline-001 --label baseline \
        --start-step 0 --end-step 3 --iterations-per-step 25 \
        --output-dir .mindie-003-wave4/t8.4-framework-baseline \
        --timeout 1800

``--mode online-cycle`` does not verify remote artifacts when
``--artifact-root`` is absent; it exits 2 with ``control_ok_artifacts_unverified``
instead of claiming success.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindie_profiler.collect import (
    ProfileController,
    ProfileWindowInjection,
    profile_window_from_serve_args,
)
from mindie_profiler.framework_capture import build_capture_plan


def _print(value: dict) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def verify_db_export(root: str | Path, *, expected_ranks: int = 1, max_files: int = 20_000) -> dict:
    base = Path(root)
    result = {
        "schema_version": "mindie-online-db-export/1",
        "root": str(base),
        "expected_ranks": expected_ranks,
        "status": "unverified",
        "rank_count": 0,
        "artifacts": [],
        "error": None,
    }
    if not base.is_dir():
        result["error"] = "artifact root does not exist"
        return result
    artifacts = []
    rank_dirs = set()
    seen = 0
    for path in base.rglob("ascend_pytorch_profiler*.db"):
        seen += 1
        if seen > max_files:
            result["error"] = f"artifact scan exceeded {max_files} files"
            return result
        if path.parent.name != "ASCEND_PROFILER_OUTPUT":
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            result["error"] = f"{path}: {exc}"
            continue
        if size <= 0:
            continue
        rank_dirs.add(path.parent.parent)
        artifacts.append({"path": str(path), "size_bytes": size})
    result["artifacts"] = sorted(artifacts, key=lambda item: item["path"])
    result["rank_count"] = len(rank_dirs)
    if result["error"]:
        result["status"] = "failed"
    elif len(rank_dirs) >= expected_ranks:
        result["status"] = "ok"
    else:
        result["status"] = "failed"
        result["error"] = f"expected {expected_ranks} rank DB artifact(s), found {len(rank_dirs)}"
    return result


def _injection(args: argparse.Namespace) -> ProfileWindowInjection:
    if args.profiler_config:
        return profile_window_from_serve_args(
            ["--profiler-config", args.profiler_config],
            endpoint=args.endpoint,
            execution_id=args.execution_id,
        )
    return ProfileWindowInjection(
        enabled=True,
        framework="vllm",
        profiler="torch",
        profile_dir=args.profile_dir,
        endpoint=args.endpoint,
        execution_id=args.execution_id,
        source="live-driver",
    )


def _online_start(args: argparse.Namespace) -> int:
    if args.receipt and Path(args.receipt).is_file():
        controller = ProfileController.from_receipt(args.receipt, timeout=args.timeout)
    else:
        controller = ProfileController(_injection(args), timeout=args.timeout, receipt_path=args.receipt)
    started = controller.start()
    _print(started)
    return 0 if started["status"] == "ok" else (2 if started["status"] == "unknown" else 1)


def _online_stop(args: argparse.Namespace) -> int:
    if not args.receipt:
        _print({"status": "failed", "error": "online-stop requires --receipt"})
        return 1
    controller = ProfileController.from_receipt(args.receipt, timeout=args.timeout)
    stopped = controller.stop()
    if args.artifact_root and stopped["status"] == "ok":
        stopped["artifact_check"] = verify_db_export(args.artifact_root, expected_ranks=args.expected_ranks)
        if stopped["artifact_check"]["status"] != "ok":
            stopped["status"] = "control_ok_artifacts_unverified"
    _print(stopped)
    if stopped["status"] == "ok":
        return 0
    return 2 if stopped["status"] in ("unknown", "control_ok_artifacts_unverified") else 1


def _online_cycle(args: argparse.Namespace) -> int:
    if not args.workload_command:
        _print({"status": "failed", "error": "online-cycle requires --workload-command"})
        return 1
    command = shlex.split(args.workload_command)
    if not command:
        _print({"status": "failed", "error": "empty --workload-command"})
        return 1
    controller = ProfileController(_injection(args), timeout=args.timeout, receipt_path=args.receipt)
    started = controller.start()
    result = {"start_profile": started, "workload": None, "stop_profile": None,
              "artifact_check": None, "status": "failed"}
    if started["status"] != "ok":
        result["error"] = "start_profile did not acknowledge"
        _print(result)
        return 2 if started["status"] == "unknown" else 1
    workload_start = time.monotonic()
    try:
        workload = subprocess.run(
            command, cwd=str(ROOT), capture_output=True, text=True, timeout=args.workload_timeout,
            check=False,
        )
        result["workload"] = {
            "command": command,
            "returncode": workload.returncode,
            "elapsed_s": time.monotonic() - workload_start,
            "stdout_tail": workload.stdout[-4000:],
            "stderr_tail": workload.stderr[-4000:],
        }
        workload_ok = workload.returncode == 0
    except subprocess.TimeoutExpired as exc:
        result["workload"] = {
            "command": command,
            "returncode": None,
            "elapsed_s": time.monotonic() - workload_start,
            "timeout": True,
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }
        workload_ok = False
    finally:
        stopped = controller.stop()
        result["stop_profile"] = stopped
    if args.artifact_root:
        result["artifact_check"] = verify_db_export(args.artifact_root, expected_ranks=args.expected_ranks)
    else:
        result["artifact_check"] = {"status": "unverified",
                                    "error": "--artifact-root not provided; DB export was not checked"}
    if not workload_ok:
        result["status"] = "failed"
        result["error"] = "workload command failed or timed out"
    elif stopped["status"] != "ok":
        result["status"] = "failed"
        result["error"] = "stop_profile did not acknowledge"
    elif result["artifact_check"]["status"] == "ok":
        result["status"] = "ok"
    else:
        result["status"] = "control_ok_artifacts_unverified"
        result["error"] = result["artifact_check"].get("error")
    _print(result)
    if result["status"] == "ok":
        return 0
    return 2 if result["status"] == "control_ok_artifacts_unverified" else 1


def _run_workload(args: argparse.Namespace, mode: str) -> int:
    script = Path(args.workload_script) if args.workload_script else ROOT / "tests" / "live_framework_capture_workload.py"
    if not script.is_file():
        _print({"status": "failed", "error": f"workload script not found: {script}"})
        return 1
    if not args.run_id or not args.label:
        _print({"status": "failed", "error": "framework/memory modes require explicit --run-id and --label"})
        return 1
    output_dir = (Path(args.output_dir) if args.output_dir else ROOT / ".mindie-003-wave4" / f"t8.4-{args.label}-{int(time.time())}").resolve()
    command = [
        sys.executable, str(script),
        "--run-id", args.run_id,
        "--label", args.label,
        "--start-step", str(args.start_step),
        "--end-step", str(args.end_step),
        "--iterations-per-step", str(args.iterations_per_step),
        "--output-dir", str(output_dir),
        "--export-budget-s", str(args.export_budget),
        "--clock-domain", args.clock_domain,
        "--host", args.host or socket.gethostname(),
        "--atol", str(args.atol),
        "--rtol", str(args.rtol),
    ]
    if args.physical_card is not None:
        command.extend(["--physical-card", str(args.physical_card)])
    if mode == "memory":
        command.append("--memory-only")
    if args.profile_memory:
        command.append("--profile-memory")
    if args.msmemscope:
        command.append("--msmemscope")
    if args.module_mem_csv:
        command.extend(["--module-mem-csv", args.module_mem_csv])
    if args.msmemscope_dump_csv:
        command.extend(["--msmemscope-dump-csv", args.msmemscope_dump_csv])
    try:
        completed = subprocess.run(
            command, cwd=str(ROOT), capture_output=True, text=True, timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _print({
            "status": "failed",
            "mode": mode,
            "command": command,
            "error": "hard subprocess timeout while capturing/exporting",
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        })
        return 1
    result_path = output_dir / "result.json"
    workload_result = None
    if result_path.is_file():
        try:
            workload_result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            workload_result = {"error": f"invalid result.json: {exc}"}
    workload_status = (
        workload_result.get("status") if isinstance(workload_result, dict)
        else ("ok" if completed.returncode == 0 else "failed")
    )
    _print({
        "status": workload_status,
        "mode": mode,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "result_path": str(result_path),
        "workload_result": workload_result,
    })
    if workload_status == "ok":
        return 0
    return 2 if workload_status == "partial" else 1


def _plan(args: argparse.Namespace) -> int:
    _print(build_capture_plan(args.start_step, args.end_step))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", required=True,
                        choices=("online-start", "online-stop", "online-cycle", "framework", "memory", "plan"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--execution-id", default="")
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--profiler-config", default=None)
    parser.add_argument("--receipt", default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--expected-ranks", type=int, default=1)
    parser.add_argument("--workload-command", default=None)
    parser.add_argument("--workload-timeout", type=float, default=1800.0)
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="hard subprocess/HTTP timeout for one live step")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, default=3)
    parser.add_argument("--iterations-per-step", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--label", choices=("baseline", "candidate"), default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--physical-card", type=int, default=None)
    parser.add_argument("--clock-domain", default="unix_wall")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--workload-script", default=None)
    parser.add_argument("--export-budget", type=float, default=600.0)
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--msmemscope", action="store_true")
    parser.add_argument("--module-mem-csv", default=None)
    parser.add_argument("--msmemscope-dump-csv", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "plan":
        return _plan(args)
    if args.mode == "online-start":
        return _online_start(args)
    if args.mode == "online-stop":
        if args.timeout <= 0:
            _print({"status": "failed", "error": "--timeout must be positive"})
            return 1
        return _online_stop(args)
    if args.mode == "online-cycle":
        return _online_cycle(args)
    return _run_workload(args, args.mode)


if __name__ == "__main__":
    sys.exit(main())
