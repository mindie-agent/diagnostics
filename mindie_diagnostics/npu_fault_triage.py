"""Deterministic CANN error-code and Ascend plog fault triage.

[Role]
  Distil CANNBot's error-code standard plus asys/msaicerr/msnpureport/
  npu-objdump guidance into two pure functions: ``decode_cann_error_code``
  explains one CANN/RTS code and ``triage_ascend_plog`` separates the earliest
  root-cause error from later cascading HCCL timeouts across peer ranks.

[Invariants]
  * Decoding is table-driven and deterministic; unknown codes remain explicit
    (``recognized=False``) and never inherit a fabricated root cause.
  * Root-cause selection prefers a concrete device/driver fault over a later or
    earlier HCCL wait/socket timeout, so a cascading peer-rank timeout is never
    reported as the primary cause when an AICore fault exists.
  * Missing plog fields stay ``None``; no shape, rank, stream or task ID is
    invented.  Recommended commands are returned as data and never executed.
  * This module performs no I/O at import time and has no third-party imports.

[Timeouts & Locks]
  Pure text/table processing with linear scans.  No locks, sleeps, subprocess,
  network or filesystem access.

[Side-Effect Policy]
  None.  Callers own execution of the returned diagnostic commands.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

CANN_CODE_RE = re.compile(r"(?<![A-Za-z0-9])E[A-Z0-9][0-9][0-9A-Z]{3}(?![A-Za-z0-9])")
RTS_CODE_RE = re.compile(r"(?<!\d)((?:107|507)\d{3})(?!\d)")
TIMESTAMP_RE = re.compile(
    r"(\d{4})[-/](\d{2})[-/](\d{2})[T _-](\d{2}):(\d{2}):(\d{2})"
    r"(?:[.,](\d{1,9})(?:[.,](\d{1,9}))?)?"
)
PID_TID_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")
RANK_RE = re.compile(r"\brank(?:_?\s*id)?[\s_:\=\[]*(\d+)", re.IGNORECASE)
DEVICE_RE = re.compile(r"\b(?:device|dev)(?:_?\s*id)?[\s_:\=\[]*(\d+)", re.IGNORECASE)
STREAM_RE = re.compile(r"\bstream(?:_?\s*id)?[\s_:\=\[]*(\d+)", re.IGNORECASE)
TASK_RE = re.compile(r"\btask(?:_?\s*id)?[\s_:\=\[]*(\d+)", re.IGNORECASE)
ERROR_RE = re.compile(r"\[(?:ERROR|Error|error)\]|\bERROR\b")
TIMEOUT_RE = re.compile(
    r"HCCL_E_WAIT_TIMEOUT|notify\s*wait\s*timeout|wait(?:ing)?\s*timeout|"
    r"get\s+socket\s+times?\s*out|error\s+cqe", re.IGNORECASE
)
AICORE_RE = re.compile(
    r"aicore\s+error|aic\s+error|aivec\s+error|vector\s+core\s+exception|"
    r"mte\s+error|cube\s+error|vec\s+error|ccu\s+instruction", re.IGNORECASE
)
OOM_RE = re.compile(
    r"out\s+of\s+memory|\boom\b|failed\s+to\s+allocate|allocat\w*\s+fail|"
    r"insufficient\s+(?:device\s+)?memory|memory\s+allocation\s+failure", re.IGNORECASE
)
GRAPH_RE = re.compile(
    r"aclgraph|graph\s+capture|stream\s+capture|capture\s+fail|invalid\s+stream|"
    r"invalid\s+context|rtStream|stream\s*/\s*context", re.IGNORECASE
)
NUMERIC_RE = re.compile(
    r"\bnan\b|\binf(?:inity)?\b|precision|overflow|underflow|numeric", re.IGNORECASE
)
OP_RE = re.compile(
    r"(?:failed\s+operator|failed\s+op|operator\s+name|operatorName|op_name|opName|"
    r"op_type|opType|kernel_name|kernelName|kernel\s+name|nodeInfo|node_info)"
    r"\s*(?:[:=]\s*|\s+)['\"]?([A-Za-z_][\w.:/-]*)"
)
TASK_TYPE_RE = re.compile(r"taskType\[([A-Za-z_][\w.:/-]*)\]")
BITMASK_RE = re.compile(
    r"(?:aic\s+error\s+mask|aic_error_mask|error\s+code|error_code|"
    r"mte\s+error\s+info|vec\s+error\s+info|cube\s+error\s+info|biu\s+error\s+info|"
    r"ccu\s+error\s+info|(?:AIC|MTE|VEC|CUBE|BIU)_ERR_?\d+|err_info|errinfo)"
    r"\s*[:=]\s*(0x[0-9a-fA-F]+)",
    re.IGNORECASE,
)

RTS_CODES: dict[str, tuple[str, str, str]] = {
    "507015": ("fatal", "aicore_kernel", "AICore exception (507015); a device kernel trapped or raised an AI Core Error."),
    "507018": ("fatal", "aicpu_kernel", "AICPU exception (507018); an AI CPU kernel or its scheduler failed."),
    "507035": ("fatal", "aicore_kernel", "Vector core exception (507035); an AIV instruction or UB address faulted."),
    "507001": ("fatal", "driver_memory", "Event/memory/driver fault (507001); a Runtime event timed out or memory access failed."),
    "107002": ("error", "runtime_rts", "Invalid stream or context (107002); the Runtime object was invalid or already released."),
    "507014": ("error", "aicore_kernel", "AICore execution timeout (507014); the kernel exceeded the device execution budget."),
    "507046": ("error", "runtime_rts", "Stream synchronize timeout (507046); the host waited past the stream deadline."),
    "507033": ("error", "driver_memory", "Device initialization or memory-mapping failure (507033)."),
}

CANN_CODES: dict[str, tuple[str, str, str]] = {
    "EZ9999": ("fatal", "aicore_kernel", "Generic internal operator error (EZ9999); the PLOG/operator context identifies the real root cause."),
    "EL0004": ("fatal", "driver_memory", "Driver memory allocation failure (EL0004); host or device memory is exhausted."),
    "EI0006": ("error", "hccl_collective", "HCCL get-socket timeout (EI0006); a peer rank is unreachable, hung or exited early."),
    "EI0002": ("error", "hccl_collective", "HCCL collective/Notify-wait timeout (EI0002); rank skew, peer failure or a broken link."),
    "EE8888": ("error", "runtime_rts", "Runtime/RTS execution failure (EE8888); inspect the surrounding Runtime trace."),
    "EZ3002": ("error", "ge_graph", "Graph/operator execution failure (EZ3002); inspect GE graph build and kernel selection."),
    "EA0001": ("error", "acl_interface", "ACL interface failure (EA0001); inspect the ACL call arguments and device context."),
}

_AICORE_COMMANDS = (
    "asys collect --tar=True --output=$HOME/dfx_info",
    "asys analyze -r=aicore_error --path=$HOME/aic_err_info",
    "cd /usr/local/Ascend/cann/tools/msaicerr && python3 msaicerr.py -p $HOME/aic_err_info -out $HOME/msaicerr_out -dev 0",
    "ASCEND_LAUNCH_BLOCKING=1 <reproduce-command>",
    "npu-objdump --dump-elf <failed_kernel.o>",
    "npu-smi info",
)
_HCCL_COMMANDS = (
    "hccn_tool -i 0 -link -g",
    "hccn_tool -i 0 -ip -g",
    'grep -rn "error cqe" "$ASCEND_PROCESS_LOG_PATH"',
    "asys collect --tar=True --output=$HOME/dfx_info",
    "msnpureport report -d 0",
    "export HCCL_EXEC_TIMEOUT=1800",
)
_MEMORY_COMMANDS = (
    "npu-smi info",
    "npu-smi info -t health -i 0",
    "free -h",
    "asys health",
    "msnpureport report -d 0",
    "asys collect --tar=True --output=$HOME/dfx_info",
)
_GRAPH_COMMANDS = (
    "ASCEND_LAUNCH_BLOCKING=1 <reproduce-command>",
    "asys collect --tar=True --output=$HOME/dfx_info",
    "asys info -r=status",
    "npu-smi info",
)
_ACL_COMMANDS = (
    "asys info -r=software",
    "asys collect --tar=True --output=$HOME/dfx_info",
    "npu-smi info",
)
_UNKNOWN_COMMANDS = (
    "asys collect --tar=True --output=$HOME/dfx_info",
    "npu-smi info",
    "asys info -r=software",
)
_SUBSYSTEM_COMMANDS = {
    "runtime_rts": _GRAPH_COMMANDS,
    "aicore_kernel": _AICORE_COMMANDS,
    "aicpu_kernel": _AICORE_COMMANDS,
    "hccl_collective": _HCCL_COMMANDS,
    "driver_memory": _MEMORY_COMMANDS,
    "ge_graph": _GRAPH_COMMANDS,
    "acl_interface": _ACL_COMMANDS,
}


def _code_commands(code: str, subsystem: str) -> tuple[str, ...]:
    base = _SUBSYSTEM_COMMANDS.get(subsystem, _UNKNOWN_COMMANDS)
    extras: tuple[str, ...] = ()
    if code in {"EI0002", "EI0006"}:
        extras = ("asys analyze -r=trace --file=<hccl_trace.bin>",)
    elif code in {"EL0004", "507001"}:
        extras = ("dmesg | grep -i -E 'hisi|svm|oom|alloc' | tail -50",)
    elif code in {"EZ9999", "507015", "507018", "507035"}:
        extras = ("grep -rn '<failed_operator>' \"$ASCEND_PROCESS_LOG_PATH\" | tail -50",)
    elif code == "107002":
        extras = ("asys analyze -r=coredump --file=<core> --exe=<binary>",)
    return base + extras


def decode_cann_error_code(code: str) -> dict[str, Any]:
    """Decode one CANN ``E...``/RTS numeric code into a deterministic record."""
    text = str(code or "").strip().upper()
    cann_match = CANN_CODE_RE.search(text)
    rts_match = RTS_CODE_RE.search(text)
    if cann_match:
        token = cann_match.group(0)
        severity, subsystem, summary = CANN_CODES.get(
            token, ("error", "runtime_rts", f"CANN error code {token}; consult the operator/Runtime PLOG context.")
        )
        kind, recognized, documented = "cann", True, token in CANN_CODES
        prefix = token[:2]
        if token not in CANN_CODES:
            subsystem = {"EI": "hccl_collective", "EL": "driver_memory", "EE": "runtime_rts",
                         "EZ": "aicore_kernel", "EA": "acl_interface", "EH": "acl_interface",
                         "E1": "ge_graph", "E2": "ge_graph", "E3": "aicpu_kernel"}.get(prefix, "runtime_rts")
            summary = f"CANN code {token}; module prefix {prefix} is recognized but this exact code is not catalogued."
    elif rts_match:
        token = rts_match.group(1)
        severity, subsystem, summary = RTS_CODES.get(
            token, ("error", "runtime_rts", f"RTS numeric code {token}; inspect the Runtime error trace.")
        )
        kind, recognized, documented = "rts", True, token in RTS_CODES
    else:
        token = text or "<empty>"
        severity, subsystem, summary = (
            "error", "runtime_rts", f"Unrecognized diagnostic code {token}; collect the full PLOG and Runtime trace."
        )
        kind, recognized, documented = "unknown", False, False
    commands = list(_code_commands(token, subsystem))
    return {
        "error_code": token, "code": token, "kind": kind, "severity": severity,
        "subsystem": subsystem, "summary": summary, "description": summary,
        "recognized": recognized, "documented": documented,
        "deterministic_next_commands": commands, "next_commands": list(commands),
        "commands": list(commands),
    }


def _timestamp(text: str) -> tuple[int | None, str | None]:
    match = TIMESTAMP_RE.search(text)
    if not match:
        return None, None
    year, month, day, hour, minute, second, fraction, sub_fraction = match.groups()
    if sub_fraction is not None:
        microsecond = int((fraction or "0").ljust(3, "0")[:3]) * 1000
        microsecond += int(sub_fraction.ljust(3, "0")[:3])
    else:
        microsecond = int((fraction or "0").ljust(6, "0")[:6])
    try:
        moment = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), microsecond)
    except ValueError:
        return None, match.group(0)
    return int(moment.timestamp() * 1_000_000), match.group(0)


def _first_int(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _codes(text: str) -> list[str]:
    found: list[str] = []
    for match in CANN_CODE_RE.finditer(text.upper()):
        if match.group(0) not in found:
            found.append(match.group(0))
    for match in RTS_CODE_RE.finditer(text):
        if match.group(1) not in found:
            found.append(match.group(1))
    return found


def _bitmask(text: str) -> int | None:
    for match in BITMASK_RE.finditer(text):
        token = match.group(1)
        try:
            value = int(token, 0)
        except (TypeError, ValueError):
            continue
        lowered = match.group(0).lower()
        if any(key in lowered for key in ("mte", "vec", "cube", "biu", "ccu")) and value > 0xFFFFFF:
            value &= 0xFFFFFF
        if value:
            return value
    return None


def _operator(text: str) -> str | None:
    for pattern in (OP_RE, TASK_TYPE_RE):
        match = pattern.search(text)
        if match:
            candidate = match.group(1).strip("'\"")
            if candidate and candidate.lower() not in {"aicore", "aicpu", "mix", "ai_core", "task"}:
                return candidate
    return None


def _parse_line(line: str, index: int) -> dict[str, Any]:
    codes = _codes(line)
    lowered = line.lower()
    is_error = bool(ERROR_RE.search(line)) or ("error" in lowered and bool(codes))
    timeout = bool(TIMEOUT_RE.search(line)) or any(code in {"EI0002", "EI0006"} for code in codes)
    moment, raw_time = _timestamp(line)
    return {
        "index": index, "line": line, "codes": codes, "is_error": is_error,
        "timeout": timeout, "moment": moment, "timestamp": raw_time,
        "pid": PID_TID_RE.search(line).group(1) if PID_TID_RE.search(line) else None,
        "tid": PID_TID_RE.search(line).group(2) if PID_TID_RE.search(line) else None,
        "rank": _first_int(RANK_RE, line), "device_id": _first_int(DEVICE_RE, line),
        "stream_id": _first_int(STREAM_RE, line), "task_id": _first_int(TASK_RE, line),
        "operator": _operator(line), "bitmask": _bitmask(line),
    }


def _score(parsed: dict[str, Any]) -> int:
    codes = parsed["codes"]
    line = parsed["line"]
    score = 0
    if any(code in {"507015", "507035", "507014"} for code in codes):
        score += 120
    if any(code in {"507001", "EL0004"} for code in codes):
        score += 110
    if any(code in {"507018", "EZ9999"} for code in codes):
        score += 90
    if AICORE_RE.search(line):
        score += 70
    if OOM_RE.search(line):
        score += 60
    if GRAPH_RE.search(line):
        score += 40
    if NUMERIC_RE.search(line):
        score += 30
    if parsed["timeout"]:
        score += 5
    else:
        score += 20
    return score


def _domain(root: dict[str, Any], codes: list[str], bitmask: int | None) -> str:
    line = root["line"] if root else ""
    root_codes = root.get("codes") or codes
    hccl_codes = any(code in {"EI0002", "EI0006"} for code in root_codes)
    if any(code in {"507015", "507035", "507014"} for code in root_codes):
        return "aicore_memory_or_tiling"
    if any(code in {"EL0004", "507001"} for code in root_codes):
        return "device_oom_or_driver"
    if hccl_codes or TIMEOUT_RE.search(line):
        return "hccl_collective_timeout_or_skew"
    if AICORE_RE.search(line):
        return "aicore_memory_or_tiling"
    if OOM_RE.search(line):
        return "device_oom_or_driver"
    if "107002" in root_codes or GRAPH_RE.search(line):
        return "aclgraph_capture_or_stream"
    if NUMERIC_RE.search(line):
        return "numeric_nan_or_precision"
    if bitmask and bitmask & (0x800000 | 0x400000):
        return "aicore_memory_or_tiling"
    return "unknown"


def _actions(codes: list[str], fault_domain: str) -> list[str]:
    actions: list[str] = []
    for code in codes:
        for command in decode_cann_error_code(code)["deterministic_next_commands"]:
            if command not in actions:
                actions.append(command)
    extras = {
        "aicore_memory_or_tiling": ("asys analyze -r=aicore_error --path=$HOME/aic_err_info",),
        "hccl_collective_timeout_or_skew": ("hccn_tool -i 0 -link -g", "hccn_tool -i 0 -ip -g"),
        "device_oom_or_driver": ("npu-smi info -t health -i 0", "free -h"),
        "aclgraph_capture_or_stream": ("ASCEND_LAUNCH_BLOCKING=1 <reproduce-command>",),
        "numeric_nan_or_precision": ("python3 -c \"import torch; print(torch.isfinite(x).all())\"",),
        "unknown": ("asys collect --tar=True --output=$HOME/dfx_info",),
    }.get(fault_domain, ())
    for command in extras:
        if command not in actions:
            actions.append(command)
    return actions[:16]


def triage_ascend_plog(plog_text: str) -> dict[str, Any]:
    """Parse raw Ascend plog text and separate root cause from cascade timeouts."""
    text = str(plog_text or "")
    parsed = [_parse_line(line, index) for index, line in enumerate(text.splitlines()) if line.strip()]
    errors = [item for item in parsed if item["is_error"]]
    root = min(errors, key=lambda item: (
        -_score(item), item["moment"] is None,
        item["moment"] if item["moment"] is not None else item["index"], item["index"],
    )) if errors else None
    if root is None:
        return {
            "root_cause_line": None, "root_cause_timestamp": None, "root_cause_pid": None,
            "root_cause_tid": None, "root_cause_rank": None, "timestamp": None, "pid": None,
            "tid": None, "error_codes": [], "aic_error_bitmask": None,
            "aic_error_bitmask_hex": None, "failed_operator": None,
            "kernel_name": None, "device_id": None, "stream_id": None, "task_id": None,
            "cascading_timeout_detected": False, "cascading_timeout_lines": [],
            "cascading_rank": None, "decoded_codes": [], "fault_domain": "unknown",
            "recommended_actions": _actions([], "unknown"),
            "summary": "No [ERROR] line or recognized fault code was found in the supplied plog text.",
        }
    def after_root(item: dict[str, Any]) -> bool:
        if item is root:
            return False
        if root["moment"] is not None and item["moment"] is not None:
            return item["moment"] >= root["moment"]
        return item["index"] > root["index"]

    cascades = [item for item in parsed if item["timeout"] and after_root(item)]
    codes = list(dict.fromkeys(root["codes"] + [code for item in parsed for code in item["codes"]]))
    bitmask = root["bitmask"]
    if bitmask is None:
        for item in parsed:
            if item["bitmask"] is not None:
                bitmask = item["bitmask"]
                break
    domain = _domain(root, codes, bitmask)
    decoded = [decode_cann_error_code(code) for code in codes]
    first_cascade = cascades[0] if cascades else None
    device_id = root["device_id"] or next((item["device_id"] for item in parsed if item["device_id"]), None)
    stream_id = root["stream_id"] or next((item["stream_id"] for item in parsed if item["stream_id"]), None)
    task_id = root["task_id"] or next((item["task_id"] for item in parsed if item["task_id"]), None)
    operator = root["operator"] or next((item["operator"] for item in parsed if item["operator"]), None)
    summary = f"{domain}: {decoded[0]['summary'] if decoded else 'unclassified error'}"
    if first_cascade is not None:
        summary += f" Cascading timeout also detected on rank {first_cascade['rank'] or 'unknown'}."
    return {
        "root_cause_line": root["line"], "root_cause_timestamp": root["timestamp"],
        "root_cause_pid": root["pid"], "root_cause_tid": root["tid"],
        "root_cause_rank": root["rank"], "timestamp": root["timestamp"], "pid": root["pid"],
        "tid": root["tid"], "error_codes": codes, "aic_error_bitmask": bitmask,
        "aic_error_bitmask_hex": None if bitmask is None else hex(bitmask),
        "failed_operator": operator, "kernel_name": operator, "device_id": device_id,
        "stream_id": stream_id, "task_id": task_id,
        "cascading_timeout_detected": bool(cascades),
        "cascading_timeout_lines": [item["line"] for item in cascades[:8]],
        "cascading_rank": first_cascade["rank"] if first_cascade else None,
        "decoded_codes": decoded, "fault_domain": domain,
        "recommended_actions": _actions(codes, domain), "summary": summary,
    }


__all__ = ["decode_cann_error_code", "triage_ascend_plog"]
