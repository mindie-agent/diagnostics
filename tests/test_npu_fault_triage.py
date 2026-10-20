"""CANNBot-distilled CANN error-code and Ascend plog triage contracts.

[Role]
  Pin the public decoding/triage contract for the listed CANN/RTS codes and the
  rank-0 AICore plus rank-1 HCCL cascade fixture.

[Invariants]
  * Known codes keep their severity/subsystem and tool-bearing commands.
  * A concrete AICore fault is selected over a cascading HCCL timeout.

[Timeouts & Locks]
  Pure in-memory tests; no locks, sleeps or subprocess execution.

[Side-Effect Policy]
  None.
"""
from __future__ import annotations

import pytest

from mindie_diagnostics import decode_cann_error_code, triage_ascend_plog


@pytest.mark.parametrize(
    ("code", "severity", "subsystem", "tool"),
    [
        ("EZ9999", "fatal", "aicore_kernel", "msaicerr"),
        ("EI0006", "error", "hccl_collective", "hccn_tool"),
        ("EL0004", "fatal", "driver_memory", "npu-smi"),
        ("507015", "fatal", "aicore_kernel", "asys"),
        ("507035", "fatal", "aicore_kernel", "npu-objdump"),
    ],
)
def test_decode_cann_error_code_known_codes(code, severity, subsystem, tool):
    result = decode_cann_error_code(code)

    assert result["error_code"] == code
    assert result["severity"] == severity
    assert result["subsystem"] == subsystem
    assert result["recognized"] is True
    assert result["summary"]
    assert result["deterministic_next_commands"]
    assert tool in " ".join(result["deterministic_next_commands"])


def test_decode_unknown_code_is_explicit_not_fabricated():
    result = decode_cann_error_code("not-a-cann-code")

    assert result["recognized"] is False
    assert result["documented"] is False
    assert result["severity"] == "error"
    assert result["deterministic_next_commands"]


def test_aicore_root_cause_wins_over_cascading_hccl_timeout():
    plog = "\n".join([
        "[ERROR] RUNTIME(4028848,4028877):2026-09-22-14:54:51.635.208 "
        "[device_error_core_proc.cc:1227] rank[0] deviceId:0 streamID:[90] taskID[686] "
        "PrintCoreInfo:An error occurs on the device, the error is aicore error, "
        "failed operator: MatMulV2, aic error mask: 0x800000",
        "[ERROR] RUNTIME(4028848,4028877):2026-09-22-14:54:51.653.742 "
        "[api_c_stream.cc:168] rank[0] rtStreamSynchronize:ErrCode=507015, "
        "desc=[aicore exception]",
        "[ERROR] HCCL(4028848,4028877):2026-09-22-14:55:01.000.000 "
        "[hccl_comm.cc:88] rank[1] HCCL_E_WAIT_TIMEOUT Notify wait timeout peer rank 0",
        "[ERROR] HCCL(4028848,4028877):2026-09-22-14:55:02.000.000 "
        "[hccl_socket.cc:99] rank[1] ErrCode=EI0006 Getting socket times out",
    ])

    result = triage_ascend_plog(plog)

    assert result["root_cause_rank"] == "0"
    assert "507015" in result["root_cause_line"]
    assert result["root_cause_pid"] == "4028848"
    assert result["root_cause_tid"] == "4028877"
    assert result["fault_domain"] == "aicore_memory_or_tiling"
    assert result["cascading_timeout_detected"] is True
    assert result["cascading_rank"] == "1"
    assert {item["error_code"] for item in result["decoded_codes"]} == {"507015", "EI0006"}
    assert result["aic_error_bitmask"] == 0x800000
    assert result["failed_operator"] == "MatMulV2"
    assert result["kernel_name"] == "MatMulV2"
    assert result["device_id"] == "0"
    assert result["stream_id"] == "90"
    assert result["task_id"] == "686"
    assert any("msaicerr" in action for action in result["recommended_actions"])
    assert any("hccn_tool" in action for action in result["recommended_actions"])


def test_aicore_root_cause_is_selected_even_when_timeout_line_appears_first():
    plog = "\n".join([
        "[ERROR] HCCL(1,2):2026-09-22-14:55:02.000.000 rank[1] EI0006 Getting socket times out",
        "[ERROR] RUNTIME(1,3):2026-09-22-14:55:01.000.000 rank[0] "
        "rtStreamSynchronize:ErrCode=507015, desc=[aicore exception]",
    ])

    result = triage_ascend_plog(plog)

    assert result["fault_domain"] == "aicore_memory_or_tiling"
    assert result["root_cause_rank"] == "0"
    assert result["cascading_timeout_detected"] is True


def test_driver_oom_and_empty_plog_are_classified_without_invention():
    oom = triage_ascend_plog(
        "[ERROR] DRIVER(7,8):2026-09-22-14:56:00.000.000 rank[0] "
        "EL0004 Failed to allocate memory"
    )
    empty = triage_ascend_plog("no errors in this plog\n")

    assert oom["fault_domain"] == "device_oom_or_driver"
    assert oom["error_codes"] == ["EL0004"]
    assert empty["root_cause_line"] is None
    assert empty["fault_domain"] == "unknown"
    assert empty["cascading_timeout_detected"] is False
    assert empty["decoded_codes"] == []
