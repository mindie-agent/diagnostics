"""
[Role]
Measure two deterministic CPU-only diagnostics workloads for formal releases:
r2 redaction over an 8 KiB mixed-secret payload and allowlist-only public event
projection (`export_public_event`) over a mixed-secret diagnostics record.

[Invariants]
Business output is asserted before any latency is emitted. The sanitized text
contains none of the planted secrets or paths and `scan_text` is clean. Public
event projection retains approved correlation/telemetry fields while stripping
planted credentials, paths, and free-form text, and is idempotent on re-export.
The numbers are component microbenchmark timings, not hardware or productivity
claims.

[Timeouts & Locks]
Fixed bounded iteration counts; no network, subprocess, sleeps, or locks.

[Side-Effect Policy]
Inputs are in-memory constants. The only filesystem write is the caller-owned
JSON at MINDIE_BENCH_RESULT_PATH.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mindie_diagnostics import export_public_event  # noqa: E402
from mindie_diagnostics.redact import redact_text, scan_text  # noqa: E402

REDACTION_PAYLOAD_BYTES = 8192
REDACTION_ITERATIONS = 100
PROJECTION_ITERATIONS = 500

_REDACTION_LINE = (
    "2026-09-22T14:54:51Z torch_npu 2.10.0 rank[0] task[686] "
    "Authorization: Bearer abcDEF1234567890token "
    "api_key=sk-abcdefghijklmnopqrstuvwx "
    "HCCL peer 10.24.7.19:64000, node-17, /home/exampleuser/project/model, "
    "operator=MatMulV2, shape=[8192,8192], dtype=bf16\n"
)
_PLANTED_SECRETS = (
    "abcDEF1234567890token",
    "sk-abcdefghijklmnopqrstuvwx",
    "10.24.7.19",
    "node-17",
    "/home/exampleuser",
)

_SECRET_TOKEN = "ghp_" + "A" * 36
_RAW_EVENT = {
    "schema": 1,
    "timestamp": "2026-09-13T12:00:00.000000Z",
    "monotonic_ns": 100,
    "pid": 42,
    "component": "remote-dev",
    "severity": "ERROR",
    "event": "operation.end",
    "operation_id": "a" * 32,
    "trace_id": "b" * 32,
    "operation": "run",
    "status": "error",
    "duration_ms": 12.5,
    "message": _SECRET_TOKEN,
    "credentials": _SECRET_TOKEN,
    "attributes": {
        "count": 2,
        "duration_ms": 12.5,
        "category": "transport",
        "submission_state": "unknown",
        "error_type": "ConnectionError",
        "error_code": 12,
        "tool": "remote.exec",
        "stage": "handshake",
        "status": "failed",
        "reason": _SECRET_TOKEN,
        "message": _SECRET_TOKEN,
        "path": "/home/exampleuser/private-project",
    },
}


def _redaction_payload() -> str:
    repeats = REDACTION_PAYLOAD_BYTES // len(_REDACTION_LINE)
    payload = _REDACTION_LINE * repeats
    payload += " " * (REDACTION_PAYLOAD_BYTES - len(payload))
    assert len(payload) == REDACTION_PAYLOAD_BYTES
    return payload


def _sanitize(payload: str) -> str:
    return redact_text(payload)


def _verify_redaction(sanitized: str) -> None:
    assert all(secret not in sanitized for secret in _PLANTED_SECRETS)
    assert "torch_npu" in sanitized and "MatMulV2" in sanitized
    findings = scan_text(sanitized)
    assert not findings, [finding.rule for finding in findings]


def _project_event() -> dict:
    projected = export_public_event(_RAW_EVENT)
    assert projected is not None
    return projected


def _verify_projection(result: dict) -> None:
    serialized = json.dumps(result)
    assert _SECRET_TOKEN not in serialized
    assert "/home/exampleuser" not in serialized
    assert "pid" not in result and "message" not in result
    assert result["attributes"] == {
        "count": 2,
        "duration_ms": 12.5,
        "category": "transport",
        "submission_state": "unknown",
        "error_type": "ConnectionError",
        "error_code": 12,
        "tool": "remote.exec",
        "stage": "handshake",
        "status": "failed",
    }
    assert export_public_event(result) == result


def _measure(name: str, iterations: int, operation, verify) -> dict:
    verify(operation())
    started = time.perf_counter()
    for _ in range(iterations):
        result = operation()
    elapsed = time.perf_counter() - started
    verify(result)
    value = elapsed * 1_000_000 / iterations
    assert math.isfinite(value) and value > 0
    return {
        "name": name,
        "unit": "us/op",
        "value": value,
        "iterations": iterations,
    }


def main(argv: list[str] | None = None) -> None:
    selected = list(sys.argv[1:] if argv is None else argv)
    payload = _redaction_payload()

    workloads = {
        "redaction": (
            "redact_8k_latency",
            REDACTION_ITERATIONS,
            lambda: _sanitize(payload),
            _verify_redaction,
        ),
        "public-event-projection": (
            "public_event_projection_latency",
            PROJECTION_ITERATIONS,
            _project_event,
            _verify_projection,
        ),
    }
    selected = selected or list(workloads)
    unknown = [name for name in selected if name not in workloads]
    if unknown:
        raise SystemExit(
            "unknown benchmark case(s): " + ", ".join(unknown)
            + "; choose " + ", ".join(workloads)
        )

    metrics = [
        _measure(*workloads[name])
        for name in selected
    ]
    destination = Path(os.environ["MINDIE_BENCH_RESULT_PATH"])
    destination.write_text(
        json.dumps({"status": "pass", "metrics": metrics}) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
