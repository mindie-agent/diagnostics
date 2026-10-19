"""In-process torch_npu profiler capture with explicit caller step bounds.

[Role]
Run one real ``torch_npu.profiler`` window in the calling process with
``_ExperimentalConfig(profiler_level=Level1, aic_metrics=PipeUtilization,
export_type=Db)``.  The caller declares global step bounds; the runner derives
the schedule, executes exactly the required model steps, performs one terminal
flush step, then stops and exports under a bounded lifecycle.

[Invariants]
- The schedule is derived only from the caller's ``[start_step, end_step)``:
  ``skip_first=max(0,start-1)``, ``warmup=0 if start==0 else 1``,
  ``active=end-start``, ``wait=0``, ``repeat=1``.  Capture must land exactly
  inside the declared bounds; the inequality is validated before start.
- ``prof.step()`` calls required by the schedule are
  ``skip_first+wait+warmup+active*repeat`` plus one terminal flush.  A plan
  that cannot satisfy this is rejected before any profiler object is created.
- Step identities come from the caller's integer loop.  No step ID is parsed,
  inferred, or guessed from a trace, CSV, or event order.
- torch and torch_npu are imported only when a capture is actually requested.
  Importing this module works on a CPU-only Mac.
- Export is synchronous (``async_mode=False``); its wall time is reported
  against a caller-supplied budget and an external process timeout is the hard
  bound.  Missing DB artifacts are an explicit failure, not success.

[Timeouts & Locks]
No subprocess, network, lock, or sleep.  ``export_budget_s`` is a reporting
budget; the live acceptance driver applies the hard subprocess timeout.

[Side-Effect Policy]
Creates the caller-provided output directory, starts one profiler, and writes
profiler/export artifacts there.  No service, container, image, or workspace
mutation.  The module never imports or initializes torch_npu at import time.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

SCHEMA_VERSION = "mindie-framework-capture/1"
PLAN_SCHEMA = "mindie-framework-capture-plan/1"
DEFAULT_EXPORT_BUDGET_S = 600.0
MAX_EXPORT_BUDGET_S = 3600.0
MAX_SCAN_FILES = 20_000
EXPERIMENTAL_CONFIG = {
    "profiler_level": "Level1",
    "aic_metrics": "PipeUtilization",
    "export_type": "Db",
}
ACTIVITIES = ("CPU", "NPU")
ARTIFACT_PATTERNS = {
    "db": "ascend_pytorch_profiler*.db",
    "kernel_details": "kernel_details.csv",
    "trace_view": "trace_view.json",
    "npu_module_mem": "npu_module_mem_*.csv",
    "memory_record": "memory_record*.csv",
}


class ProfileWindowError(ValueError):
    """A caller-declared window cannot be mapped to a safe schedule."""

    def __init__(self, code: str, message: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


class FrameworkCaptureUnavailable(RuntimeError):
    """The installed runtime is absent or does not expose the required API."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProfileWindowError("invalid_step", f"{field} must be an integer", {field: value})
    return value


def build_capture_plan(start_step: int, end_step: int, *, repeat: int = 1) -> dict[str, Any]:
    """Map caller bounds to the exact torch_npu schedule and prove containment."""
    start_step = _integer(start_step, "start_step")
    end_step = _integer(end_step, "end_step")
    repeat = _integer(repeat, "repeat")
    if start_step < 0:
        raise ProfileWindowError("invalid_step", "start_step must be non-negative", {"start_step": start_step})
    if end_step <= start_step:
        raise ProfileWindowError("empty_window", "end_step must be greater than start_step",
                                 {"start_step": start_step, "end_step": end_step})
    if repeat != 1:
        raise ProfileWindowError("repeat_unsupported",
                                 "explicit step windows require repeat=1", {"repeat": repeat})
    active = end_step - start_step
    skip_first = max(0, start_step - 1)
    warmup = 0 if start_step == 0 else 1
    wait = 0
    capture_start = skip_first + wait + warmup
    capture_end = capture_start + active * repeat
    scheduled_steps = skip_first + wait + warmup + active * repeat
    required_calls = scheduled_steps + 1
    declared_calls = end_step + 1
    inequality = {
        "expression": "skip_first + wait + warmup + active*repeat + terminal_flush",
        "skip_first": skip_first,
        "wait": wait,
        "warmup": warmup,
        "active": active,
        "repeat": repeat,
        "scheduled_steps": scheduled_steps,
        "terminal_flush_calls": 1,
        "required_prof_step_calls": required_calls,
        "declared_model_steps": end_step,
        "declared_prof_step_calls": declared_calls,
        "capture_start_step": capture_start,
        "capture_end_step": capture_end,
        "capture_within_declared": capture_start >= start_step and capture_end <= end_step,
        "capture_matches_declared": capture_start == start_step and capture_end == end_step,
        "calls_match_declared": required_calls == declared_calls,
        "satisfied": (
            capture_start == start_step
            and capture_end == end_step
            and required_calls == declared_calls
        ),
    }
    if not inequality["satisfied"]:
        raise ProfileWindowError("window_crosses_bounds",
                                 "derived schedule escapes the caller-declared window", inequality)
    return {
        "schema_version": PLAN_SCHEMA,
        "caller_window": {
            "start_step": start_step,
            "end_step": end_step,
            "end_exclusive": True,
            "step_count": active,
        },
        "schedule": {
            "skip_first": skip_first,
            "wait": wait,
            "warmup": warmup,
            "active": active,
            "repeat": repeat,
        },
        "capture": {
            "start_step": capture_start,
            "end_step": capture_end,
            "captured_step_ids": list(range(capture_start, capture_end)),
            "step_identity_source": "caller_declared_range",
        },
        "inequality": inequality,
        "execution": {
            "model_step_ids": list(range(0, end_step)),
            "terminal_flush_step_id": end_step,
            "terminal_flush_is_model_step": False,
        },
        "experimental_config": dict(EXPERIMENTAL_CONFIG),
        "activities": list(ACTIVITIES),
    }


def validate_window_inequality(start_step: int, end_step: int, *, repeat: int = 1) -> dict[str, Any]:
    """Return the checked containment inequality for caller bounds."""
    return build_capture_plan(start_step, end_step, repeat=repeat)["inequality"]


def assert_plan_satisfied(plan: Mapping[str, Any]) -> None:
    inequality = plan.get("inequality") if isinstance(plan, Mapping) else None
    if not isinstance(inequality, Mapping) or not inequality.get("satisfied"):
        raise ProfileWindowError("window_crosses_bounds", "capture plan inequality is not satisfied",
                                 dict(inequality or {}))


class _LimitedWriter(io.TextIOBase):
    def __init__(self, limit: int = 64 * 1024) -> None:
        self.limit = limit
        self.parts: list[str] = []
        self.size = 0
        self.truncated = False

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        remaining = self.limit - self.size
        if remaining > 0:
            chunk = text[:remaining]
            self.parts.append(chunk)
            self.size += len(chunk)
        if len(text) > remaining:
            self.truncated = True
        return len(text)

    def getvalue(self) -> str:
        return "".join(self.parts)


def _load_runtime() -> tuple[Any, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        import torch_npu  # type: ignore[import-not-found]
    except ImportError as exc:
        raise FrameworkCaptureUnavailable("torch_npu_missing",
                                          f"torch/torch_npu import failed: {exc}") from exc
    npu = getattr(torch_npu, "npu", None)
    if npu is None:
        raise FrameworkCaptureUnavailable("torch_npu_api_missing", "torch_npu.npu is missing")
    is_available = getattr(npu, "is_available", None)
    if not callable(is_available) or not is_available():
        raise FrameworkCaptureUnavailable("npu_unavailable", "torch_npu.npu.is_available() is false")
    profiler = getattr(torch_npu, "profiler", None)
    if profiler is None:
        raise FrameworkCaptureUnavailable("torch_npu_api_missing", "torch_npu.profiler is missing")
    required = (
        "_ExperimentalConfig", "ProfilerLevel", "AiCMetrics", "ExportType",
        "ProfilerActivity", "profile", "schedule", "tensorboard_trace_handler",
    )
    missing = [name for name in required if not hasattr(profiler, name)]
    for parent, name in (("ProfilerLevel", "Level1"), ("AiCMetrics", "PipeUtilization"),
                         ("ExportType", "Db"), ("ProfilerActivity", "CPU"),
                         ("ProfilerActivity", "NPU")):
        owner = getattr(profiler, parent, None)
        if owner is not None and not hasattr(owner, name):
            missing.append(f"{parent}.{name}")
    if missing:
        raise FrameworkCaptureUnavailable("torch_npu_api_missing",
                                          "torch_npu.profiler is missing: " + ", ".join(missing))
    return torch, torch_npu


def _build_profiler(torch_npu: Any, plan: Mapping[str, Any], output_dir: Path,
                    *, profile_memory: bool, export_events: list[dict[str, Any]]) -> Any:
    profiler = torch_npu.profiler
    experimental = profiler._ExperimentalConfig(
        profiler_level=profiler.ProfilerLevel.Level1,
        aic_metrics=profiler.AiCMetrics.PipeUtilization,
        export_type=profiler.ExportType.Db,
    )
    activity = profiler.ProfilerActivity
    schedule = plan["schedule"]
    export_trace = profiler.tensorboard_trace_handler(
        str(output_dir), analyse_flag=True, async_mode=False
    )

    def measured_export(profile: Any) -> Any:
        # Scheduled capture calls the export handler from step() on the final
        # active step. stop() may therefore do no analysis at all.
        event = {"wall_start_ns": time.time_ns(), "status": "running"}
        started_ns = time.perf_counter_ns()
        try:
            result = export_trace(profile)
            event["status"] = "completed"
            return result
        except Exception as exc:
            event.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            event["duration_ms"] = _ms(started_ns, time.perf_counter_ns())
            event["wall_end_ns"] = time.time_ns()
            export_events.append(event)

    return profiler.profile(
        activities=[activity.CPU, activity.NPU],
        record_shapes=False,
        profile_memory=bool(profile_memory),
        with_stack=False,
        experimental_config=experimental,
        schedule=profiler.schedule(
            wait=schedule["wait"],
            warmup=schedule["warmup"],
            active=schedule["active"],
            repeat=schedule["repeat"],
            skip_first=schedule["skip_first"],
        ),
        on_trace_ready=measured_export,
    )


def scan_capture_artifacts(root: str | Path, *, max_files: int = MAX_SCAN_FILES) -> dict[str, Any]:
    """Scan exported artifacts by filename only; no step identity is inferred."""
    base = Path(root)
    if not base.is_dir():
        return {"ok": False, "root": str(base), "artifacts": [], "by_kind": {},
                "error": "output directory does not exist"}
    artifacts: list[dict[str, Any]] = []
    seen = 0
    for directory, _dirs, files in os.walk(base):
        for name in files:
            seen += 1
            if seen > max_files:
                return {"ok": False, "root": str(base), "artifacts": artifacts,
                        "by_kind": {}, "error": f"artifact scan exceeded {max_files} files"}
            path = Path(directory) / name
            kind = next((kind for kind, pattern in ARTIFACT_PATTERNS.items()
                         if path.match(pattern)), None)
            if kind is None:
                continue
            try:
                size = path.stat().st_size
            except OSError as exc:
                artifacts.append({"kind": kind, "path": str(path), "size_bytes": None,
                                  "error": str(exc), "relative_path": str(path.relative_to(base))})
                continue
            artifacts.append({
                "kind": kind,
                "path": str(path),
                "relative_path": str(path.relative_to(base)),
                "size_bytes": size,
                "non_empty": size > 0,
                "modified_ns": path.stat().st_mtime_ns,
            })
    by_kind: dict[str, int] = {}
    for item in artifacts:
        by_kind[item["kind"]] = by_kind.get(item["kind"], 0) + 1
    return {"ok": True, "root": str(base), "artifacts": sorted(artifacts, key=lambda item: item["path"]),
            "by_kind": by_kind, "error": None}


def _stats(durations_ns: list[int]) -> dict[str, Any] | None:
    if not durations_ns:
        return None
    values = [value / 1e6 for value in durations_ns]
    return {
        "count": len(values),
        "total_ms": sum(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": sum(values) / len(values),
    }


def _ms(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1e6


def _validate_export_budget(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("export_budget_s must be a positive number")
    return min(float(value), MAX_EXPORT_BUDGET_S)


def export_budget_exceeded(stop_and_export_ms: float, budget_s: float) -> bool:
    """Compare a millisecond measurement with a second budget explicitly."""
    if isinstance(stop_and_export_ms, bool) or not isinstance(stop_and_export_ms, (int, float)):
        raise ValueError("stop_and_export_ms must be numeric")
    if isinstance(budget_s, bool) or not isinstance(budget_s, (int, float)) or budget_s <= 0:
        raise ValueError("budget_s must be positive")
    return float(stop_and_export_ms) > float(budget_s) * 1000.0


def evaluate_capture_success(
    artifacts: Mapping[str, Any],
    *,
    stop_acknowledged: bool,
    capture_contains_active: bool,
    export_within_budget: bool,
) -> dict[str, Any]:
    """Gate success on a real, non-empty DB export and contained active timing."""
    non_empty = [item for item in artifacts.get("artifacts", []) if item.get("non_empty")]
    db_non_empty = [item for item in non_empty if item.get("kind") == "db"]
    requirements = {
        "stop_acknowledged": bool(stop_acknowledged),
        "artifact_scan_ok": bool(artifacts.get("ok")),
        "non_empty_db_present": bool(db_non_empty),
        "capture_contains_active": bool(capture_contains_active),
        "export_within_budget": bool(export_within_budget),
    }
    if not requirements["stop_acknowledged"]:
        return {"status": "failed", "requirements": requirements,
                "error": {"code": "stop_not_acknowledged",
                          "message": "profiler.stop() did not report a completed stop"}}
    if not requirements["artifact_scan_ok"]:
        return {"status": "failed", "requirements": requirements,
                "error": {"code": "artifact_scan_failed", "message": artifacts.get("error")}}
    if not requirements["non_empty_db_present"]:
        return {"status": "missing_db_export", "requirements": requirements,
                "error": {"code": "missing_db_export",
                          "message": "no non-empty ascend_pytorch_profiler*.db artifact found"}}
    if not requirements["capture_contains_active"]:
        return {"status": "failed", "requirements": requirements,
                "error": {"code": "active_window_outside_capture",
                          "message": "active step wall bounds are not contained by the total capture window"}}
    if not requirements["export_within_budget"]:
        return {"status": "export_over_budget", "requirements": requirements,
                "error": {"code": "export_over_budget",
                          "message": "export/analysis exceeded the caller's export budget"}}
    return {"status": "ok", "requirements": requirements, "error": None}


def run_framework_capture(
    step_fn: Callable[[int], Any],
    *,
    start_step: int,
    end_step: int,
    output_dir: str | Path,
    profile_memory: bool = False,
    export_budget_s: float = DEFAULT_EXPORT_BUDGET_S,
    device: int | None = None,
) -> dict[str, Any]:
    """Run one real in-process capture and return a traceable receipt."""
    if not callable(step_fn):
        raise TypeError("step_fn must be callable")
    plan = build_capture_plan(start_step, end_step)
    assert_plan_satisfied(plan)
    budget = _validate_export_budget(export_budget_s)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "clock_domain": "unix_wall",
        "wall_clock": "time.time_ns",
        "monotonic_clock": "time.perf_counter_ns",
        "phase_units": "ms",
        "plan": plan,
        "output_dir": str(root),
        "device": device,
        "phases": {},
        "steps": {},
        "step_records": [],
        "artifacts": None,
        "success_requirements": None,
        "profiler_output": {"stdout": "", "stderr": "", "truncated": False,
                            "error_lines": []},
        "error": None,
        "export_budget_s": budget,
        "export_budget_ms": budget * 1000.0,
    }
    run_wall_start_ns = time.time_ns()
    total_start = time.perf_counter_ns()
    result["run_wall_start_ns"] = run_wall_start_ns
    result["run_wall_start_ts"] = run_wall_start_ns / 1e9
    result["perf_start_ns"] = total_start
    writer = _LimitedWriter()
    profiler = None
    started = False
    terminal_done = False
    capture_wall_start_ns = None
    capture_wall_end_ns = None
    active_wall_start_ns = None
    active_wall_end_ns = None
    export_events: list[dict[str, Any]] = []
    try:
        import_start = time.perf_counter_ns()
        torch_module, torch_npu = _load_runtime()
        result["phases"]["import"] = _ms(import_start, time.perf_counter_ns())
        result["environment"] = {
            "torch_version": getattr(torch_module, "__version__", None),
            "torch_npu_version": getattr(torch_npu, "__version__", None),
            "device": device,
            "profile_memory": bool(profile_memory),
        }
        build_start = time.perf_counter_ns()
        profiler = _build_profiler(torch_npu, plan, root, profile_memory=profile_memory,
                                   export_events=export_events)
        result["phases"]["build"] = _ms(build_start, time.perf_counter_ns())
        capture_wall_start_ns = time.time_ns()
        capture_perf_start_ns = time.perf_counter_ns()
        result["capture_wall_start_ns"] = capture_wall_start_ns
        result["capture_wall_start_ts"] = capture_wall_start_ns / 1e9
        result["capture_perf_start_ns"] = capture_perf_start_ns
        start_start = time.perf_counter_ns()
        with _redirect(writer):
            profiler.start()
        started = True
        result["phases"]["start"] = _ms(start_start, time.perf_counter_ns())
        if getattr(profiler, "stopped", False):
            raise RuntimeError("profiler reported stopped immediately after start()")
        schedule = plan["schedule"]
        skipped_end = schedule["skip_first"]
        warmup_end = skipped_end + schedule["warmup"]
        durations = {"skipped": [], "warmup": [], "active": []}
        step_records: list[dict[str, Any]] = []
        loop_start = time.perf_counter_ns()
        for step in range(0, end_step):
            segment = "skipped" if step < skipped_end else ("warmup" if step < warmup_end else "active")
            wall_start_ns = time.time_ns()
            perf_start_ns = time.perf_counter_ns()
            step_fn(step)
            perf_end_ns = time.perf_counter_ns()
            wall_end_ns = time.time_ns()
            duration_ns = perf_end_ns - perf_start_ns
            durations[segment].append(duration_ns)
            if segment == "active":
                active_wall_start_ns = wall_start_ns if active_wall_start_ns is None else active_wall_start_ns
                active_wall_end_ns = wall_end_ns
            profiler_wall_start_ns = time.time_ns()
            profiler_perf_start_ns = time.perf_counter_ns()
            with _redirect(writer):
                profiler.step()
            profiler_perf_end_ns = time.perf_counter_ns()
            profiler_wall_end_ns = time.time_ns()
            step_records.append({
                "step": step,
                "segment": segment,
                "wall_start_ns": wall_start_ns,
                "wall_end_ns": wall_end_ns,
                "wall_start_ts": wall_start_ns / 1e9,
                "wall_end_ts": wall_end_ns / 1e9,
                "perf_start_ns": perf_start_ns,
                "perf_end_ns": perf_end_ns,
                "duration_ms": duration_ns / 1e6,
                "profiler_step_wall_start_ns": profiler_wall_start_ns,
                "profiler_step_wall_end_ns": profiler_wall_end_ns,
                "profiler_step_perf_start_ns": profiler_perf_start_ns,
                "profiler_step_perf_end_ns": profiler_perf_end_ns,
                "profiler_step_duration_ms": (profiler_perf_end_ns - profiler_perf_start_ns) / 1e6,
                "clock_domain": "unix_wall",
            })
        result["phases"]["loop"] = _ms(loop_start, time.perf_counter_ns())
        result["phases"]["collect"] = result["phases"]["loop"]
        result["phases"]["active"] = sum(durations["active"]) / 1e6
        flush_start = time.perf_counter_ns()
        with _redirect(writer):
            profiler.step()
        terminal_done = True
        result["phases"]["terminal_flush"] = _ms(flush_start, time.perf_counter_ns())
        stop_start = time.perf_counter_ns()
        with _redirect(writer):
            profiler.stop()
        stop_end = time.perf_counter_ns()
        stop_ms = _ms(stop_start, stop_end)
        export_ms = sum(event["duration_ms"] for event in export_events)
        result["phases"]["export_and_analysis"] = export_ms
        result["phases"]["stop_and_export"] = stop_ms
        result["analysis_included_in_export"] = True
        result["analysis_timing"] = {
            "status": "included_in_export_and_analysis",
            "export_and_analysis_ms": export_ms,
            "measurement_source": "synchronous_on_trace_ready_callback",
            "separately_measured": False,
        }
        result["export_events"] = export_events
        result["export_budget_exceeded"] = export_budget_exceeded(export_ms, budget)
        result["stop_acknowledged"] = getattr(profiler, "stopped", None) is True
        capture_wall_end_ns = time.time_ns()
        capture_perf_end_ns = time.perf_counter_ns()
        result["capture_wall_end_ns"] = capture_wall_end_ns
        result["capture_wall_end_ts"] = capture_wall_end_ns / 1e9
        result["capture_perf_end_ns"] = capture_perf_end_ns
        result["phases"]["capture_total"] = _ms(capture_perf_start_ns, capture_perf_end_ns)
        result["active_wall_start_ns"] = active_wall_start_ns
        result["active_wall_end_ns"] = active_wall_end_ns
        result["active_wall_start_ts"] = None if active_wall_start_ns is None else active_wall_start_ns / 1e9
        result["active_wall_end_ts"] = None if active_wall_end_ns is None else active_wall_end_ns / 1e9
        result["capture_contains_active"] = (
            capture_wall_start_ns is not None
            and capture_wall_end_ns is not None
            and active_wall_start_ns is not None
            and active_wall_end_ns is not None
            and capture_wall_start_ns <= active_wall_start_ns
            and active_wall_end_ns <= capture_wall_end_ns
        )
        result["step_records"] = step_records
        result["steps"] = {
            "model_step_calls": end_step,
            "skipped": _stats(durations["skipped"]),
            "warmup": _stats(durations["warmup"]),
            "active": _stats(durations["active"]),
            "active_sample_ms": [value / 1e6 for value in durations["active"]],
            "captured_step_ids": plan["capture"]["captured_step_ids"],
            "terminal_flush_executed": terminal_done,
            "step_identity_source": plan["capture"]["step_identity_source"],
        }
        scan_start = time.perf_counter_ns()
        artifacts = scan_capture_artifacts(root)
        result["phases"]["artifact_scan"] = _ms(scan_start, time.perf_counter_ns())
        result["artifacts"] = artifacts
        if result["error"] is None:
            decision = evaluate_capture_success(
                artifacts,
                stop_acknowledged=bool(result["stop_acknowledged"]),
                capture_contains_active=bool(result["capture_contains_active"]),
                export_within_budget=not bool(result["export_budget_exceeded"]),
            )
            result["success_requirements"] = decision["requirements"]
            result["status"] = decision["status"]
            result["error"] = decision["error"]
    except FrameworkCaptureUnavailable as exc:
        result["error"] = {"code": exc.code, "message": str(exc)}
        result["status"] = "failed"
    except Exception as exc:
        result["error"] = {"code": getattr(exc, "code", "capture_failed"),
                           "message": f"{type(exc).__name__}: {exc}"}
        result["status"] = "failed"
    finally:
        if profiler is not None and started and not result["phases"].get("stop_and_export"):
            cleanup_start = time.perf_counter_ns()
            try:
                with _redirect(writer):
                    profiler.stop()
            except Exception as exc:
                result["cleanup_error"] = {"code": "cleanup_stop_failed", "message": str(exc)}
            result["phases"]["cleanup_stop"] = _ms(cleanup_start, time.perf_counter_ns())
        if capture_wall_end_ns is None:
            capture_wall_end_ns = time.time_ns()
            result["capture_wall_end_ns"] = capture_wall_end_ns
            result["capture_wall_end_ts"] = capture_wall_end_ns / 1e9
        output = writer.getvalue()
        result["profiler_output"].update({
            "stdout": output,
            "truncated": writer.truncated,
            "error_lines": [line for line in output.splitlines() if "[ERROR]" in line],
        })
        run_total_ms = _ms(total_start, time.perf_counter_ns())
        result["export_events"] = export_events
        result["phases"]["run_total"] = run_total_ms
        result["phases"]["total"] = result["phases"].get("capture_total", run_total_ms)
        result["run_wall_end_ns"] = time.time_ns()
        result["run_wall_end_ts"] = result["run_wall_end_ns"] / 1e9
        result["perf_end_ns"] = time.perf_counter_ns()
    if result["status"] == "ok" and result["profiler_output"]["error_lines"]:
        result["status"] = "degraded"
        result["warning"] = {"code": "profiler_error_output",
                            "message": "profiler wrote error-like lines; artifacts were still produced"}
    return result


class _redirect:
    def __init__(self, writer: _LimitedWriter) -> None:
        self.writer = writer
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = self.writer
        sys.stderr = self.writer
        return self

    def __exit__(self, *_exc):
        sys.stdout, sys.stderr = self._stdout, self._stderr
        return False


def write_capture_result(path: str | Path, result: Mapping[str, Any]) -> None:
    """Write a capture receipt atomically to the caller's path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--end-step", type=int, required=True)
    parser.add_argument("--plan-only", action="store_true",
                        help="print the explicit plan without importing torch")
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_capture_plan(args.start_step, args.end_step)
    if args.plan_only or args.output_dir is None:
        print(json.dumps(plan, ensure_ascii=False))
        return 0
    raise SystemExit("run captures through tests/live_framework_capture_workload.py")


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "EXPERIMENTAL_CONFIG",
    "FrameworkCaptureUnavailable",
    "ProfileWindowError",
    "assert_plan_satisfied",
    "build_capture_plan",
    "evaluate_capture_success",
    "export_budget_exceeded",
    "run_framework_capture",
    "validate_window_inequality",
    "scan_capture_artifacts",
    "write_capture_result",
]
