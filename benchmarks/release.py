"""
[Role]
Measure two deterministic CPU-only diagnostics workloads for formal releases:
r2 redaction over an 8 KiB mixed-secret payload and Ascend fault triage over a
realistic HCCL-timeout plus HBM-ECC cascade.

[Invariants]
Business output is asserted before any latency is emitted. The sanitized text
contains none of the planted secrets or paths and `scan_text` is clean. Triage
selects the HBM/driver root cause, extracts rank/device/operator evidence, and
records the later HCCL timeout as a cascade rather than the primary cause. The
numbers are component microbenchmark timings, not hardware or productivity
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

from mindie_diagnostics import triage_npu_fault  # noqa: E402
from mindie_diagnostics.redact import redact_text, scan_text  # noqa: E402

REDACTION_PAYLOAD_BYTES = 8192
REDACTION_ITERATIONS = 100
TRIAGE_ITERATIONS = 500

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

_HCCL_HBM_PLOG = "\n".join([
    "[ERROR] HCCL(4211,4220):2026-09-22-14:55:01.000.000 "
    "[hccl_comm.cc:88] rank[1] HCCL_E_WAIT_TIMEOUT "
    "Notify wait timeout peer rank 0",
    "[ERROR] RUNTIME(4211,4219):2026-09-22-14:54:51.635.208 "
    "[device_error_core_proc.cc:1227] rank[0] deviceId:0 streamID:[90] "
    "taskID[686] PrintCoreInfo:An error occurs on the device, "
    "HBM ECC uncorrectable error detected, failed operator: MatMulV2, "
    "aic error mask: 0x800000",
    "[ERROR] DRIVER(4211,4221):2026-09-22-14:54:52.000.000 "
    "rank[0] deviceId:0 ErrCode=E89999, "
    "desc=[HBM ECC uncorrectable fault on device 0]",
    "[ERROR] HCCL(4211,4220):2026-09-22-14:55:02.000.000 "
    "[hccl_socket.cc:99] rank[1] ErrCode=EI0006 Getting socket times out",
])


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


def _triage() -> dict:
    return triage_npu_fault(_HCCL_HBM_PLOG)


def _verify_triage(result: dict) -> None:
    assert result["fault_domain"] == "device_oom_or_driver"
    assert result["error_codes"][0] == "E89999"
    assert "HBM ECC" in result["root_cause_line"]
    assert result["root_cause_rank"] == "0"
    assert result["device_id"] == "0"
    assert result["failed_operator"] == "MatMulV2"
    assert result["decoded_codes"][0]["subsystem"] == "driver_memory"
    assert result["cascading_timeout_detected"] is True
    assert result["cascading_rank"] == "1"
    assert len(result["cascading_timeout_lines"]) == 2
    assert all(
        "HCCL" in line or "hccl" in line
        for line in result["cascading_timeout_lines"]
    )
    assert "npu-smi" in result["recommended_tools"]
    assert "HCCL_ENTRY_LOG_ENABLE=1" in result["recommended_tools"]


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
        "npu-fault-triage": (
            "npu_fault_triage_latency",
            TRIAGE_ITERATIONS,
            _triage,
            _verify_triage,
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
    )


if __name__ == "__main__":
    main()
