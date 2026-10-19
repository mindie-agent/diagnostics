"""Boundary tests for explicit profile-window control.

The HTTP server is a real loopback ``ThreadingHTTPServer``; no service is
started and no Ascend hardware or collection backend is mocked.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mindie_profiler.collect import (
    ProfileController,
    ProfileWindowInjection,
    check_profile_window,
    load_receipt,
    profile_window_from_serve_args,
)


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.requests.append({
            "method": "POST",
            "path": urllib.parse.urlsplit(self.path).path,
            "body": body,
            "content_type": self.headers.get("Content-Type"),
        })
        if self.server.delay:
            time.sleep(self.server.delay)
        payload = b'{"status":"ok"}'
        try:
            self.send_response(self.server.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def control_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    server.requests = []
    server.status = 200
    server.delay = 0.0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _endpoint(server: ThreadingHTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_port}"


def _injection(server: ThreadingHTTPServer, **overrides: object) -> ProfileWindowInjection:
    values: dict[str, object] = {
        "enabled": True,
        "framework": "vllm",
        "profiler": "torch",
        "profile_dir": "./vllm_profile",
        "endpoint": _endpoint(server),
        "execution_id": "exec-1",
        "api_server_count": 1,
        "source": "serve_args",
    }
    values.update(overrides)
    return ProfileWindowInjection(**values)


def test_start_stop_post_explicit_endpoints_and_persist_receipt(control_server, tmp_path: Path):
    receipt = tmp_path / "window.json"
    controller = ProfileController(_injection(control_server), timeout=2.0, receipt_path=receipt)
    assert control_server.requests == []
    assert controller.state == "closed"

    started = controller.start()
    assert started["status"] == "ok"
    assert started["http_status"] == 200
    assert started["sent"] is True
    assert started["external_effect"] == "acknowledged"
    assert started["reconcile"] is None
    assert controller.state == "open"

    stopped = controller.stop()
    assert stopped["status"] == "ok"
    assert stopped["action"] == "stop_profile"
    assert controller.state == "closed"

    assert [(item["path"], item["method"]) for item in control_server.requests] == [
        ("/start_profile", "POST"),
        ("/stop_profile", "POST"),
    ]
    assert all(item["content_type"] == "application/json" for item in control_server.requests)
    assert all(item["body"] == b"" for item in control_server.requests)

    on_disk = load_receipt(receipt)
    assert on_disk["schema_version"] == "mindie-profiler-window/1"
    assert on_disk["state"] == "closed"
    assert on_disk["pending_unknown"] is None
    assert len(on_disk["actions"]) == 2
    assert on_disk["window_check"]["ok"] is True


def test_receipt_resume_preserves_open_state_and_reconciles(control_server, tmp_path: Path):
    receipt = tmp_path / "window.json"
    first = ProfileController(_injection(control_server), timeout=2.0, receipt_path=receipt)
    assert first.start()["status"] == "ok"
    resumed = ProfileController.from_receipt(receipt, timeout=2.0)
    assert resumed.state == "open"
    assert resumed.window.execution_id == "exec-1"
    assert resumed.stop()["status"] == "ok"
    assert len(control_server.requests) == 2


def test_timeout_is_unknown_keeps_identity_and_blocks_blind_retry(control_server, tmp_path: Path):
    control_server.delay = 0.5
    controller = ProfileController(
        _injection(control_server), timeout=0.05, receipt_path=tmp_path / "window.json"
    )
    started = controller.start()
    assert started["status"] == "unknown"
    assert started["sent"] is True
    assert started["external_effect"] == "unknown"
    assert started["error"]["code"] == "request_timeout"
    identity = started["reconcile"]
    assert identity["request_id"] == started["request_id"]
    assert identity["action"] == "start_profile"
    assert identity["execution_id"] == "exec-1"
    assert identity["endpoint"] == _endpoint(control_server)
    assert controller.state == "unknown"
    assert len(control_server.requests) == 1
    pending = load_receipt(tmp_path / "window.json")
    assert pending["state"] == "unknown"
    assert pending["pending_unknown"]["request_id"] == started["request_id"]

    blocked = controller.start()
    assert blocked["status"] == "failed"
    assert blocked["sent"] is False
    assert blocked["error"]["code"] == "needs_reconciliation"
    assert blocked["pending_request_id"] == started["request_id"]
    assert len(control_server.requests) == 1

    blocked_stop = controller.stop()
    assert blocked_stop["status"] == "failed"
    assert blocked_stop["error"]["code"] == "needs_reconciliation"
    assert len(control_server.requests) == 1

    control_server.delay = 0.0
    resolved = controller.reconcile("open", evidence="trace directory for the execution appeared")
    assert resolved["status"] == "ok"
    assert resolved["resolved_state"] == "open"
    assert controller.pending_unknown is None
    assert controller.state == "open"

    stopped = controller.stop()
    assert stopped["status"] == "ok"
    assert len(control_server.requests) == 2


def test_http_rejection_is_failed_and_never_opens_window(control_server):
    control_server.status = 503
    controller = ProfileController(_injection(control_server), timeout=2.0)
    started = controller.start()
    assert started["status"] == "failed"
    assert started["http_status"] == 503
    assert started["sent"] is True
    assert started["external_effect"] == "rejected"
    assert started["reconcile"] is None
    assert started["error"]["code"] == "http_rejected"
    assert controller.state == "closed"

    stopped = controller.stop()
    assert stopped["status"] == "failed"
    assert stopped["sent"] is False
    assert stopped["error"]["code"] == "window_not_open"
    assert len(control_server.requests) == 1


def test_unavailable_endpoint_is_failed_before_send():
    # Reserve the port without listening so another process cannot take it.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        injection = ProfileWindowInjection(
            enabled=True,
            framework="vllm",
            profiler="torch",
            profile_dir="./vllm_profile",
            endpoint=f"http://127.0.0.1:{port}",
            execution_id="exec-refused",
            source="serve_args",
        )
        controller = ProfileController(injection, timeout=0.5)
        result = controller.start()
    assert result["status"] == "failed"
    assert result["sent"] is False
    assert result["external_effect"] == "none"
    # Winsock may exhaust the connect budget before returning refusal. Both
    # outcomes prove that no profiling request reached an external service.
    assert result["error"]["code"] in {"connect_failed", "connect_timeout"}


def test_missing_framework_injection_never_contacts_endpoint(control_server):
    controller = ProfileController(_injection(control_server, enabled=False), timeout=2.0)
    assert controller.ready is False
    result = controller.start()
    assert result["status"] == "failed"
    assert result["sent"] is False
    assert result["error"]["code"] == "profile_window_unavailable"
    assert control_server.requests == []

    check = check_profile_window(_injection(control_server, profile_dir=""))
    assert check["ok"] is False
    assert {item["code"] for item in check["issues"]} >= {"profile_dir_missing"}


def test_stop_requires_a_window_this_controller_opened(control_server):
    controller = ProfileController(_injection(control_server), timeout=2.0)
    result = controller.stop()
    assert result["status"] == "failed"
    assert result["error"]["code"] == "window_not_open"
    assert control_server.requests == []


def test_framework_injection_is_derived_from_serve_args_and_environment(monkeypatch):
    endpoint = "http://127.0.0.1:9999"
    config = json.dumps({"profiler": "torch", "torch_profiler_dir": "./traces"})
    injection = profile_window_from_serve_args(
        ["--profiler-config", config, "--api-server-count", "2"],
        endpoint=endpoint,
        execution_id="exec-args",
    )
    assert injection.enabled is True
    assert injection.profile_dir == "./traces"
    assert injection.api_server_count == 2
    assert injection.source == "serve_args"
    check = check_profile_window(injection)
    assert check["ok"] is True
    assert {item["code"] for item in check["warnings"]} == {"multi_api_server_routing"}

    text_config = json.dumps({"profiler": "torch", "torch_profiler_dir": "./traces", "export_type": "text"})
    text_injection = profile_window_from_serve_args(
        ["--profiler-config", text_config], endpoint=endpoint, execution_id="exec-text"
    )
    assert text_injection.export_type == "text"
    assert "db_export_not_declared" in {item["code"] for item in check_profile_window(text_injection)["warnings"]}

    invalid = profile_window_from_serve_args(
        ["--profiler-config", "{"], endpoint=endpoint, execution_id="exec-bad"
    )
    assert invalid.enabled is False
    assert invalid.source == "serve_args_invalid"

    monkeypatch.setenv("VLLM_TORCH_PROFILER_DIR", "/tmp/injected")
    from_env = profile_window_from_serve_args([], endpoint=endpoint, execution_id="exec-env")
    assert from_env.enabled is True
    assert from_env.profile_dir == "/tmp/injected"
    assert from_env.source == "environment"

    absent = profile_window_from_serve_args([], endpoint=endpoint, execution_id="exec-none", environ={})
    assert absent.enabled is False
    assert check_profile_window(absent)["issues"][0]["code"] == "window_not_enabled"
