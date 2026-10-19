"""Explicit online profile-window control with honest outcome accounting.

[Role]
Send exactly one ``POST /start_profile`` or ``POST /stop_profile`` to a
framework-injected profile window.  This module never starts a service, never
runs a workload, never collects a trace, and never waits for a wall-clock
window.  ``stop_profile`` is the explicit export trigger; elapsed time is not
used to decide that a window opened, closed, or exported.

[Invariants]
- The window must first pass the framework-injection check.  A missing or
  invalid ``--profiler-config`` / ``VLLM_TORCH_PROFILER_DIR`` injection is a
  pre-request failure and is reported as ``failed``.
- An HTTP rejection (non-2xx) or a pre-request transport failure is
  ``failed``.  A timeout or a response lost after sending is ``unknown`` and
  keeps a reconciliation identity.  It is never retried blindly.
- A controller with an outstanding ``unknown`` action refuses further
  start/stop sends until ``reconcile()`` records explicit evidence.
- State transitions are driven only by acknowledgements and explicit
  reconciliation, never by timers.

[Timeouts & Locks]
One bounded ``http.client`` request per call (default 600s, hard cap 3600s)
with a 64 KiB response cap.  No retry loop, subprocess, or sleep.  A small
in-process state lock is non-reentrant and only held during transitions.

[Side-Effect Policy]
Network I/O happens only when a caller invokes ``start()`` or ``stop()``.
The optional receipt is written atomically to the caller-provided path; this
module never writes workspace state, reads ``.vaws-local``, or imports VAWS.
"""

from __future__ import annotations

import argparse
import errno
import http.client
import json
import os
import socket
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

SCHEMA_VERSION = "mindie-profiler-window/1"
CONTROL_SCHEMA = "mindie-profile-control/1"
RECONCILE_SCHEMA = "mindie-profile-reconcile/1"
ALLOWED_ACTIONS = ("start_profile", "stop_profile")
DEFAULT_TIMEOUT_SECONDS = 600.0
MAX_TIMEOUT_SECONDS = 3600.0
DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class ProfileWindowInjection:
    """Facts the serving framework injected before the endpoints can exist."""

    enabled: bool
    framework: str
    profiler: str
    profile_dir: str
    endpoint: str
    execution_id: str
    api_server_count: int | None = 1
    export_type: str | None = None
    source: str = "unknown"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "framework": self.framework,
            "profiler": self.profiler,
            "profile_dir": self.profile_dir,
            "endpoint": self.endpoint,
            "execution_id": self.execution_id,
            "api_server_count": self.api_server_count,
            "export_type": self.export_type,
            "source": self.source,
            "detail": self.detail,
        }


def _arg_value(argv: list[str], flag: str) -> str | None:
    value = None
    for index, item in enumerate(argv):
        if item == flag:
            if index + 1 < len(argv):
                value = argv[index + 1]
        elif item.startswith(flag + "="):
            value = item.split("=", 1)[1]
    return value


def profile_window_from_serve_args(
    serve_args: list[str] | tuple[str, ...],
    *,
    endpoint: str,
    execution_id: str,
    environ: Mapping[str, str] | None = None,
) -> ProfileWindowInjection:
    """Derive the injected window from a real ``vllm serve`` argv / environment.

    ``--profiler-config`` is the mechanism the referenced collection skill
    forwards.  ``VLLM_TORCH_PROFILER_DIR`` is the equivalent framework
    environment injection.  Absence of both means the endpoints must not be
    called.
    """
    argv = [str(item) for item in serve_args]
    config_text = _arg_value(argv, "--profiler-config")
    env = os.environ if environ is None else environ
    if config_text is not None:
        try:
            config = json.loads(config_text)
        except json.JSONDecodeError as exc:
            return ProfileWindowInjection(
                enabled=False,
                framework="vllm",
                profiler="torch",
                profile_dir="",
                endpoint=endpoint,
                execution_id=execution_id,
                api_server_count=None,
                source="serve_args_invalid",
                detail=f"invalid --profiler-config JSON: {exc}",
            )
        if not isinstance(config, Mapping):
            return ProfileWindowInjection(
                enabled=False,
                framework="vllm",
                profiler="torch",
                profile_dir="",
                endpoint=endpoint,
                execution_id=execution_id,
                api_server_count=None,
                source="serve_args_invalid",
                detail="--profiler-config is not an object",
            )
        count = _arg_value(argv, "--api-server-count")
        try:
            api_server_count = int(count) if count is not None else 1
        except ValueError:
            api_server_count = None
        raw_export = config.get("export_type")
        if isinstance(raw_export, (list, tuple)):
            export_type = ",".join(str(item) for item in raw_export)
        else:
            export_type = str(raw_export) if raw_export is not None else None
        return ProfileWindowInjection(
            enabled=True,
            framework="vllm",
            profiler=str(config.get("profiler") or "torch"),
            profile_dir=str(config.get("torch_profiler_dir") or ""),
            endpoint=endpoint,
            execution_id=execution_id,
            api_server_count=api_server_count,
            export_type=export_type,
            source="serve_args",
        )
    env_dir = str(env.get("VLLM_TORCH_PROFILER_DIR") or "")
    if env_dir:
        return ProfileWindowInjection(
            enabled=True,
            framework="vllm",
            profiler="torch",
            profile_dir=env_dir,
            endpoint=endpoint,
            execution_id=execution_id,
            api_server_count=None,
            source="environment",
        )
    return ProfileWindowInjection(
        enabled=False,
        framework="vllm",
        profiler="torch",
        profile_dir="",
        endpoint=endpoint,
        execution_id=execution_id,
        api_server_count=None,
        source="absent",
    )


def _as_injection(value: ProfileWindowInjection | Mapping[str, Any]) -> ProfileWindowInjection:
    if isinstance(value, ProfileWindowInjection):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("injection must be a ProfileWindowInjection or mapping")
    return ProfileWindowInjection(
        enabled=bool(value.get("enabled")),
        framework=str(value.get("framework") or "vllm"),
        profiler=str(value.get("profiler") or "torch"),
        profile_dir=str(value.get("profile_dir") or ""),
        endpoint=str(value.get("endpoint") or ""),
        execution_id=str(value.get("execution_id") or ""),
        api_server_count=value.get("api_server_count"),
        export_type=(str(value["export_type"]) if value.get("export_type") is not None else None),
        source=str(value.get("source") or "mapping"),
        detail=str(value.get("detail") or ""),
    )


def _endpoint_parts(endpoint: str) -> tuple[Any, list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    try:
        parts = urlsplit(endpoint)
    except ValueError as exc:
        return None, [{"code": "endpoint_invalid", "message": str(exc)}]
    if parts.scheme not in ("http", "https"):
        issues.append({"code": "endpoint_scheme", "message": "endpoint must be http or https"})
    if not parts.hostname:
        issues.append({"code": "endpoint_host", "message": "endpoint has no host"})
    if parts.username or parts.password:
        issues.append({"code": "endpoint_userinfo", "message": "endpoint must not carry userinfo"})
    if parts.query or parts.fragment:
        issues.append({"code": "endpoint_query", "message": "endpoint must not carry query or fragment"})
    if issues:
        return None, issues
    return parts, []


def check_profile_window(injection: ProfileWindowInjection | Mapping[str, Any]) -> dict[str, Any]:
    """Check the framework injection before any endpoint is contacted."""
    try:
        window = _as_injection(injection)
    except (TypeError, ValueError) as exc:
        return {
            "ok": False,
            "issues": [{"code": "window_missing", "message": str(exc)}],
            "warnings": [],
            "injection": None,
        }
    issues: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if not window.enabled:
        issues.append({
            "code": "window_not_enabled",
            "message": "no framework-injected profiler window is enabled",
        })
    if window.framework != "vllm":
        issues.append({
            "code": "framework_unsupported",
            "message": f"framework {window.framework!r} does not provide these endpoints",
        })
    if window.profiler != "torch":
        issues.append({
            "code": "profiler_unsupported",
            "message": f"profiler {window.profiler!r} is not the torch endpoint contract",
        })
    if not window.profile_dir.strip() or "\n" in window.profile_dir or "\x00" in window.profile_dir:
        issues.append({"code": "profile_dir_missing", "message": "profile output directory is empty or invalid"})
    if not window.execution_id.strip():
        issues.append({"code": "execution_id_missing", "message": "execution identity is required for reconciliation"})
    parts, endpoint_issues = _endpoint_parts(window.endpoint)
    issues.extend(endpoint_issues)
    if window.api_server_count is not None and (
        not isinstance(window.api_server_count, int) or window.api_server_count < 1
    ):
        issues.append({"code": "api_server_count_invalid", "message": "api server count must be a positive integer"})
    elif window.api_server_count is not None and window.api_server_count > 1:
        warnings.append({
            "code": "multi_api_server_routing",
            "message": "multi-api-server mode can answer a control request on a server that does not drive workers",
        })
    if window.export_type is not None:
        export = window.export_type.strip().lower()
        if export not in ("db", "both"):
            warnings.append({
                "code": "db_export_not_declared",
                "message": f"declared export_type={window.export_type!r} does not include DB",
            })
    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "injection": window.to_dict(),
        "endpoint_valid": parts is not None,
    }


def _action_url(endpoint: str, action: str) -> str:
    return endpoint.rstrip("/") + "/" + action


def _request_path(endpoint: str, action: str) -> str:
    parts = urlsplit(endpoint)
    base = parts.path.rstrip("/")
    return f"{base}/{action}" if base else f"/{action}"


def _body_excerpt(raw: bytes, limit: int = DEFAULT_MAX_RESPONSE_BYTES) -> tuple[str, bool]:
    truncated = len(raw) > limit
    return raw[:limit].decode("utf-8", errors="replace"), truncated


def _base_result(
    *,
    action: str,
    endpoint: str,
    execution_id: str,
    request_id: str,
    started_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": CONTROL_SCHEMA,
        "action": action,
        "endpoint": endpoint,
        "execution_id": execution_id,
        "request_id": request_id,
        "started_at": started_at,
        "finished_at": None,
        "status": "failed",
        "sent": False,
        "external_effect": "none",
        "http_status": None,
        "body": "",
        "body_truncated": False,
        "error": None,
        "reconcile": None,
    }


def _reconcile_identity(
    *, request_id: str, action: str, endpoint: str, execution_id: str, started_at: str
) -> dict[str, Any]:
    return {
        "schema_version": RECONCILE_SCHEMA,
        "request_id": request_id,
        "action": action,
        "endpoint": endpoint,
        "execution_id": execution_id,
        "started_at": started_at,
        "instruction": "inspect trace/output evidence before any further control request",
    }


def _validate_timeout(timeout: float) -> float:
    if not _is_number(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive number")
    return min(float(timeout), MAX_TIMEOUT_SECONDS)


def post_profile_action(
    endpoint: str,
    action: str,
    *,
    execution_id: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Send one explicit control request and classify its external effect.

    A returned ``unknown`` means the request was sent but no acknowledgement
    was observed.  The caller must reconcile it; this function never retries.
    """
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"unsupported action {action!r}; allowed {ALLOWED_ACTIONS}")
    timeout = _validate_timeout(timeout)
    if not isinstance(max_response_bytes, int) or max_response_bytes <= 0:
        raise ValueError("max_response_bytes must be a positive integer")
    started_at = _utc_now()
    request_id = request_id or uuid.uuid4().hex
    result = _base_result(
        action=action,
        endpoint=endpoint,
        execution_id=execution_id,
        request_id=request_id,
        started_at=started_at,
    )
    parts, issues = _endpoint_parts(endpoint)
    if issues or parts is None:
        result["error"] = {"code": "endpoint_invalid", "message": issues[0]["message"] if issues else "invalid endpoint"}
        result["finished_at"] = _utc_now()
        return result
    assert parts.hostname is not None
    port = parts.port or (443 if parts.scheme == "https" else 80)
    path = _request_path(endpoint, action)
    target = _action_url(endpoint, action)
    result["target"] = target
    connection_class = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    connection = connection_class(parts.hostname, port, timeout=timeout)
    try:
        connection.connect()
        result["sent"] = True
        connection.request(
            "POST",
            path,
            body=b"",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "mindie-profiler-window/1",
            },
        )
        response = connection.getresponse()
        raw = response.read(max_response_bytes + 1)
        body, truncated = _body_excerpt(raw, max_response_bytes)
        result.update({
            "http_status": response.status,
            "body": body,
            "body_truncated": truncated,
        })
        if 200 <= response.status < 300:
            result["status"] = "ok"
            result["external_effect"] = "acknowledged"
        else:
            result["status"] = "failed"
            result["external_effect"] = "rejected"
            result["error"] = {
                "code": "http_rejected",
                "message": f"{action} rejected with HTTP {response.status}",
            }
    except (socket.timeout, TimeoutError) as exc:
        if not result["sent"]:
            result["error"] = {
                "code": "connect_timeout",
                "message": str(exc) or "connection timed out before the request was sent",
            }
        else:
            result.update({
                "status": "unknown",
                "sent": True,
                "external_effect": "unknown",
                "error": {"code": "request_timeout", "message": str(exc) or "control request timed out"},
                "reconcile": _reconcile_identity(
                    request_id=request_id,
                    action=action,
                    endpoint=endpoint,
                    execution_id=execution_id,
                    started_at=started_at,
                ),
            })
    except (http.client.RemoteDisconnected, http.client.BadStatusLine, http.client.IncompleteRead, http.client.HTTPException) as exc:
        result.update({
            "status": "unknown",
            "sent": True,
            "external_effect": "unknown",
            "error": {"code": "response_lost", "message": f"{type(exc).__name__}: {exc}"},
            "reconcile": _reconcile_identity(
                request_id=request_id,
                action=action,
                endpoint=endpoint,
                execution_id=execution_id,
                started_at=started_at,
            ),
        })
    except (ConnectionRefusedError, socket.gaierror) as exc:
        result["error"] = {"code": "connect_failed", "message": str(exc) or type(exc).__name__}
    except OSError as exc:
        if not result["sent"] and exc.errno in (
            errno.ECONNREFUSED, errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EADDRNOTAVAIL
        ):
            result["error"] = {"code": "connect_failed", "message": str(exc)}
        elif result["sent"] and isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            result.update({
                "status": "unknown",
                "sent": True,
                "external_effect": "unknown",
                "error": {"code": "response_lost", "message": str(exc)},
                "reconcile": _reconcile_identity(
                    request_id=request_id,
                    action=action,
                    endpoint=endpoint,
                    execution_id=execution_id,
                    started_at=started_at,
                ),
            })
        else:
            result["error"] = {"code": "transport_failed", "message": str(exc)}
    finally:
        connection.close()
        result["finished_at"] = _utc_now()
    return result


def start_profile(endpoint: str, *, execution_id: str, **kwargs: Any) -> dict[str, Any]:
    return post_profile_action(endpoint, "start_profile", execution_id=execution_id, **kwargs)


def stop_profile(endpoint: str, *, execution_id: str, **kwargs: Any) -> dict[str, Any]:
    return post_profile_action(endpoint, "stop_profile", execution_id=execution_id, **kwargs)


def _failed_before_send(
    *, action: str, endpoint: str, execution_id: str, code: str, message: str
) -> dict[str, Any]:
    result = _base_result(
        action=action,
        endpoint=endpoint,
        execution_id=execution_id,
        request_id=uuid.uuid4().hex,
        started_at=_utc_now(),
    )
    result["error"] = {"code": code, "message": message}
    result["finished_at"] = _utc_now()
    return result


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise

class ProfileController:
    """State machine that refuses blind retries of an unknown control action."""

    def __init__(
        self,
        injection: ProfileWindowInjection | Mapping[str, Any],
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        receipt_path: str | Path | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self.window = _as_injection(injection)
        self.timeout = _validate_timeout(timeout)
        self.receipt_path = Path(receipt_path) if receipt_path is not None else None
        self.max_response_bytes = max_response_bytes
        self.window_check = check_profile_window(self.window)
        self.state = "closed" if self.window_check["ok"] else "unavailable"
        self.pending_unknown: dict[str, Any] | None = None
        self.actions: list[dict[str, Any]] = []
        self.last_error: dict[str, str] | None = None
        self._lock = threading.Lock()
        self._persist()
    @classmethod
    def from_receipt(cls, path: str | Path, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> "ProfileController":
        data = load_receipt(path)
        injection = ProfileWindowInjection(
            enabled=bool((data.get("window_check") or {}).get("injection", {}).get("enabled")),
            framework=str((data.get("window_check") or {}).get("injection", {}).get("framework") or "vllm"),
            profiler=str((data.get("window_check") or {}).get("injection", {}).get("profiler") or "torch"),
            profile_dir=str(data.get("profile_dir") or ""),
            endpoint=str(data.get("endpoint") or ""),
            execution_id=str(data.get("execution_id") or ""),
            api_server_count=(data.get("window_check") or {}).get("injection", {}).get("api_server_count"),
            export_type=(data.get("window_check") or {}).get("injection", {}).get("export_type"),
            source="receipt",
        )
        controller = cls(injection, timeout=timeout, receipt_path=path)
        controller.state = str(data.get("state") or controller.state)
        controller.pending_unknown = data.get("pending_unknown")
        controller.actions = list(data.get("actions") or [])
        controller.window_check = data.get("window_check") or controller.window_check
        controller.last_error = data.get("last_error")
        controller._persist()
        return controller

    @property
    def ready(self) -> bool:
        return bool(self.window_check["ok"])

    def _receipt(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "execution_id": self.window.execution_id,
            "endpoint": self.window.endpoint,
            "profile_dir": self.window.profile_dir,
            "window_check": self.window_check,
            "state": self.state,
            "pending_unknown": self.pending_unknown,
            "last_error": self.last_error,
            "actions": list(self.actions),
            "updated_at": _utc_now(),
        }

    def _persist(self) -> None:
        if self.receipt_path is not None:
            _write_json_atomic(self.receipt_path, self._receipt())

    def _remember(self, result: dict[str, Any]) -> None:
        self.actions.append(result)
        if result["status"] == "ok":
            self.state = "open" if result["action"] == "start_profile" else "closed"
            self.pending_unknown = None
            self.last_error = None
        elif result["status"] == "unknown":
            self.state = "unknown"
            self.pending_unknown = result.get("reconcile")
            self.last_error = result.get("error")
        else:
            self.last_error = result.get("error")
            if result["action"] == "start_profile" and self.state not in ("open", "unknown", "unavailable"):
                self.state = "closed"
        self._persist()

    def _blocked(self, action: str) -> dict[str, Any]:
        pending = self.pending_unknown or {}
        result = _failed_before_send(
            action=action,
            endpoint=self.window.endpoint,
            execution_id=self.window.execution_id,
            code="needs_reconciliation",
            message="an earlier control request has unknown external effect; reconcile before sending another",
        )
        result["pending_request_id"] = pending.get("request_id")
        result["reconcile"] = pending or None
        self._remember(result)
        return result

    def start(self) -> dict[str, Any]:
        """Explicitly open a window; no timer and no internal retry."""
        with self._lock:
            if self.pending_unknown is not None or self.state == "unknown":
                return self._blocked("start_profile")
            if not self.ready:
                result = _failed_before_send(
                    action="start_profile",
                    endpoint=self.window.endpoint,
                    execution_id=self.window.execution_id,
                    code="profile_window_unavailable",
                    message="; ".join(issue["code"] for issue in self.window_check["issues"]),
                )
                self._remember(result)
                return result
            if self.state == "open":
                result = _failed_before_send(
                    action="start_profile",
                    endpoint=self.window.endpoint,
                    execution_id=self.window.execution_id,
                    code="window_already_open",
                    message="profile window is already open",
                )
                self._remember(result)
                return result
            result = post_profile_action(
                self.window.endpoint,
                "start_profile",
                execution_id=self.window.execution_id,
                timeout=self.timeout,
                max_response_bytes=self.max_response_bytes,
            )
            self._remember(result)
            return result

    def stop(self) -> dict[str, Any]:
        """Explicitly close the window and trigger export; no timer path."""
        with self._lock:
            if self.pending_unknown is not None or self.state == "unknown":
                return self._blocked("stop_profile")
            if not self.ready:
                result = _failed_before_send(
                    action="stop_profile",
                    endpoint=self.window.endpoint,
                    execution_id=self.window.execution_id,
                    code="profile_window_unavailable",
                    message="; ".join(issue["code"] for issue in self.window_check["issues"]),
                )
                self._remember(result)
                return result
            if self.state != "open":
                result = _failed_before_send(
                    action="stop_profile",
                    endpoint=self.window.endpoint,
                    execution_id=self.window.execution_id,
                    code="window_not_open",
                    message="refusing to stop a window this controller did not open",
                )
                self._remember(result)
                return result
            result = post_profile_action(
                self.window.endpoint,
                "stop_profile",
                execution_id=self.window.execution_id,
                timeout=self.timeout,
                max_response_bytes=self.max_response_bytes,
            )
            self._remember(result)
            return result

    def start_profile(self) -> dict[str, Any]:
        return self.start()

    def stop_profile(self) -> dict[str, Any]:
        return self.stop()

    def reconcile(self, resolved_state: str, *, evidence: str) -> dict[str, Any]:
        """Resolve an outstanding unknown from explicit external evidence."""
        with self._lock:
            if self.pending_unknown is None:
                return _failed_before_send(
                    action="reconcile",
                    endpoint=self.window.endpoint,
                    execution_id=self.window.execution_id,
                    code="no_pending_unknown",
                    message="no control action is awaiting reconciliation",
                )
            if resolved_state not in ("open", "closed", "failed"):
                raise ValueError("resolved_state must be open, closed, or failed")
            if not evidence.strip():
                raise ValueError("reconcile requires non-empty evidence")
            pending = self.pending_unknown
            self.state = resolved_state
            self.pending_unknown = None
            result = {
                "schema_version": RECONCILE_SCHEMA,
                "action": "reconcile",
                "endpoint": self.window.endpoint,
                "execution_id": self.window.execution_id,
                "request_id": pending.get("request_id"),
                "status": "ok" if resolved_state != "failed" else "failed",
                "resolved_state": resolved_state,
                "evidence": evidence,
                "external_effect": "none",
                "finished_at": _utc_now(),
            }
            self.actions.append(result)
            self._persist()
            return result


def load_receipt(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("receipt must be a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--action", choices=ALLOWED_ACTIONS, required=True)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--receipt")
    parser.add_argument("--profiler-config", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.profiler_config:
        injection = profile_window_from_serve_args(
            ["--profiler-config", args.profiler_config],
            endpoint=args.endpoint,
            execution_id=args.execution_id,
        )
    else:
        injection = ProfileWindowInjection(
            enabled=True,
            framework="vllm",
            profiler="torch",
            profile_dir=args.profile_dir,
            endpoint=args.endpoint,
            execution_id=args.execution_id,
            source="cli",
        )
    controller = ProfileController(
        injection, timeout=args.timeout, receipt_path=args.receipt
    )
    result = controller.start() if args.action == "start_profile" else controller.stop()
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ALLOWED_ACTIONS",
    "CONTROL_SCHEMA",
    "ProfileController",
    "ProfileWindowInjection",
    "check_profile_window",
    "load_receipt",
    "post_profile_action",
    "profile_window_from_serve_args",
    "start_profile",
    "stop_profile",
]
