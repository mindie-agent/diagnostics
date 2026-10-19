"""Opt-in online memory-phase probe validation and receipt scanning.

The worker that writes these probes is ``tests/live_vllm_memory_worker.py``.
This module is stdlib-only so local contract tests never import torch or vLLM.
It validates bounded sampler coverage and reads named JSON observations without
filling missing values.

[Role] Validate and read real online memory probe receipts.
[Invariants] Capacity reserves both controls and requests; native export is excluded from the measured scope.
[Timeouts & Locks] Pure budget arithmetic and bounded local receipt reads.
[Side-Effect Policy] No process, network, or filesystem mutations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .vllm_online_evidence import OnlineEvidenceError


def validate_memory_probe_options(
    *,
    interval_ms: float,
    max_samples: int,
    stop_timeout_s: float,
    profile_requests: int,
    request_timeout: float,
    control_timeout: float,
    max_allowed_samples: int = 16384,
) -> dict[str, Any]:
    """Validate bounded sampler cadence and worst-case profile-window coverage."""
    for name, value in (("interval_ms", interval_ms), ("stop_timeout_s", stop_timeout_s),
                        ("request_timeout", request_timeout), ("control_timeout", control_timeout)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise OnlineEvidenceError("invalid_memory_probe_options", f"{name} must be positive")
    if not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples <= 1:
        raise OnlineEvidenceError("invalid_memory_probe_options", "max_samples must be an integer > 1")
    if max_samples > max_allowed_samples:
        raise OnlineEvidenceError("invalid_memory_probe_options",
                                  f"max_samples {max_samples} exceeds bound {max_allowed_samples}")
    if not isinstance(profile_requests, int) or isinstance(profile_requests, bool) or profile_requests <= 0:
        raise OnlineEvidenceError("invalid_memory_probe_options", "profile_requests must be a positive integer")
    coverage_s = (float(interval_ms) / 1000.0) * int(max_samples)
    # Reserve both control budgets even though native stop/export is excluded.
    required_s = float(profile_requests) * float(request_timeout) + 2 * float(control_timeout) + float(stop_timeout_s)
    if coverage_s < required_s:
        raise OnlineEvidenceError(
            "sampler_window_too_short",
            f"sampler max window {coverage_s:.3f}s < worst-case control/request/stop window {required_s:.3f}s",
        )
    return {
        "interval_ms": float(interval_ms),
        "max_samples": int(max_samples),
        "stop_timeout_s": float(stop_timeout_s),
        "max_allowed_samples": int(max_allowed_samples),
        "coverage_s": coverage_s,
        "required_window_s": required_s,
        "coverage_margin_s": coverage_s - required_s,
    }


MEMORY_REQUIRED_PHASES = (
    "after_device_init_before_model_load",
    "after_model_load",
    "after_determine_available_memory",
    "after_cache_initialization",
    "after_warmup",
    "immediately_before_profile_start",
    "immediately_after_profile_stop",
)


def scan_memory_phase_probes(
    root: str | Path,
    *,
    required_phases: tuple[str, ...] = MEMORY_REQUIRED_PHASES,
    max_files: int = 64,
    require_model_state: bool = False,
) -> dict[str, Any]:
    """Read named worker probes without inferring or zero-filling observations."""
    directory = Path(root)
    if not directory.is_dir():
        return {
            "schema_version": "mindie-online-memory-probe-scan/1",
            "phase_dir": str(directory),
            "enabled": False,
            "status": "missing",
            "error": {"code": "phase_dir_missing", "message": f"missing {directory}"},
            "files": [],
            "phases": {},
            "missing_phases": list(required_phases),
            "errors": [],
            "sampler": None,
            "model_state": None,
            "worker": None,
        }
    files = sorted(path for path in directory.iterdir() if path.is_file())
    if len(files) > max_files:
        raise OnlineEvidenceError("phase_probe_file_bound",
                                  f"{len(files)} files exceeds bound {max_files}: {directory}")
    phases: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    sampler = None
    model_state = None
    worker = None
    parse_errors: list[dict[str, Any]] = []
    for path in files:
        if path.suffix != ".json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parse_errors.append({"path": str(path), "code": "probe_json_invalid",
                                 "message": f"{type(exc).__name__}: {exc}"})
            continue
        if path.name == "errors.json":
            if isinstance(payload, list):
                errors.extend(item for item in payload if isinstance(item, dict))
            continue
        if path.name == "sampler.json":
            sampler = payload if isinstance(payload, dict) else None
            continue
        if path.name == "model-state.json":
            model_state = payload if isinstance(payload, dict) else None
            continue
        if path.name == "model-state-error.json":
            model_state = payload if isinstance(payload, dict) else None
            continue
        if path.name == "worker.json":
            worker = payload if isinstance(payload, dict) else None
            continue
        if isinstance(payload, dict) and isinstance(payload.get("phase"), str):
            phases[payload["phase"]] = {
                "path": str(path),
                "sequence": payload.get("sequence"),
                "status": payload.get("status"),
                "pid": payload.get("pid"),
                "device": payload.get("device"),
                "query_bounds": payload.get("query_bounds"),
                "snapshot": payload.get("snapshot"),
                "error": payload.get("error"),
                "extra": payload.get("extra"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    missing = [phase for phase in required_phases if phase not in phases]
    phase_errors = [phase for phase, item in phases.items() if item.get("status") != "ok"]
    sampler_scope = (sampler or {}).get("scope") if isinstance(sampler, dict) else None
    sampler_scope_ok = bool(isinstance(sampler_scope, dict) and sampler_scope.get("export_excluded") is True)
    sampler_ok = bool(
        sampler
        and sampler.get("status") == "ok"
        and isinstance(sampler.get("summary"), dict)
        and sampler_scope_ok
    )
    model_state_ok = bool(model_state and (model_state.get("status") != "error") and model_state.get("sha256"))
    status = "ok"
    if parse_errors or errors or phase_errors or missing or not sampler_ok or not worker:
        status = "partial"
    if require_model_state and not model_state_ok:
        status = "partial"
    if not phases and not sampler and not model_state:
        status = "failed"
    return {
        "schema_version": "mindie-online-memory-probe-scan/1",
        "phase_dir": str(directory),
        "enabled": True,
        "status": status,
        "files": [{"name": path.name, "size_bytes": path.stat().st_size} for path in files],
        "phases": phases,
        "missing_phases": missing,
        "phase_errors": phase_errors,
        "errors": errors,
        "parse_errors": parse_errors,
        "sampler": sampler,
        "sampler_ok": sampler_ok,
        "sampler_scope": sampler_scope,
        "sampler_scope_ok": sampler_scope_ok,
        "sampler_export_excluded": bool(
            isinstance(sampler, dict)
            and ((sampler.get("scope") or {}).get("export_excluded"))
        ),
        "model_state": model_state,
        "model_state_ok": model_state_ok,
        "model_state_required": bool(require_model_state),
        "worker": worker,
    }

def classify_online_status(
    checks: dict[str, bool],
    *,
    control_start_status: str | None = None,
    control_stop_status: str | None = None,
) -> str:
    """Classify functional checks without inventing a DB or memory result."""
    if all(bool(value) for value in checks.values()):
        return "ok"
    if control_start_status == "unknown" or control_stop_status == "unknown":
        return "unknown"
    functional = (
        "server_ready",
        "pre_request",
        "start_profile",
        "during_requests",
        "stop_profile",
        "post_stop_health",
        "post_request",
        "process_cleanup",
    )
    if all(bool(checks.get(name)) for name in functional):
        return "partial"
    return "failed"


def partial_result_limitations(checks: dict[str, bool]) -> list[dict[str, str]]:
    """Return only limitations proven by the check map.

    Database absence is emitted by the harness from the actual DB artifact scan,
    not inferred from the generic partial branch.
    """
    limitations: list[dict[str, str]] = []
    if "memory_phase_probes" in checks and not checks["memory_phase_probes"]:
        limitations.append({
            "code": "memory_phase_probes_incomplete",
            "message": "memory phase/sampler evidence is missing or has errors; see memory_observation and named JSON probes",
        })
    return limitations


__all__ = [
    "MEMORY_REQUIRED_PHASES",
    "classify_online_status",
    "partial_result_limitations",
    "scan_memory_phase_probes",
    "validate_memory_probe_options",
]
