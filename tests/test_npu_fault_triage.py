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
        ("EI0002", "error", "hccl_collective", "HCCL_ENTRY_LOG_ENABLE=1"),
        ("EI0004", "error", "hccl_collective", "hccn_tool"),
        ("EI0006", "error", "hccl_collective", "hccn_tool"),
        ("EL0004", "fatal", "driver_memory", "npu-smi"),
        ("E80000", "fatal", "driver_memory", "npu-smi"),
        ("EE8888", "error", "runtime_rts", "asys"),
        ("EE9999", "fatal", "runtime_rts", "asys"),
        ("507015", "fatal", "aicore_kernel", "asys"),
        ("507018", "fatal", "aicpu_kernel", "msaicerr"),
        ("507035", "fatal", "aicore_kernel", "npu-objdump"),
        ("507014", "error", "aicore_kernel", "npu-check"),
        ("107003", "error", "runtime_rts", "perf"),
        ("161001", "error", "acl_interface", "ldd"),
        ("207001", "fatal", "driver_memory", "dmesg"),
        ("561103", "error", "acl_interface", "ldd"),
    ],
)
def test_decode_cann_error_code_known_codes(code, severity, subsystem, tool):
    result = decode_cann_error_code(code)

    assert result["error_code"] == code
    assert result["severity"] == severity
    assert result["subsystem"] == subsystem
    assert result["recognized"] is True
    assert result["documented"] is True
    assert result["summary"]
    assert result["recommended_tools"]
    assert result["remediation_steps"]
    assert result["deterministic_next_commands"]
    assert tool in " ".join(result["remediation_steps"]) or tool in result["recommended_tools"]


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
    hbm_oom = triage_ascend_plog(
        "[ERROR] RUNTIME(9,10):2026-09-22-14:57:00.000.000 rank[0] deviceId:3 "
        "ErrCode=E80000, desc=[HBM memory allocation failure]"
    )
    empty = triage_ascend_plog("no errors in this plog\n")

    assert oom["fault_domain"] == "device_oom_or_driver"
    assert oom["error_codes"] == ["EL0004"]
    assert "npu-smi" in oom["recommended_tools"]
    assert hbm_oom["fault_domain"] == "device_oom_or_driver"
    assert hbm_oom["error_codes"] == ["E80000"]
    assert hbm_oom["device_id"] == "3"
    assert empty["root_cause_line"] is None
    assert empty["fault_domain"] == "unknown"
    assert empty["cascading_timeout_detected"] is False
    assert empty["decoded_codes"] == []
    assert empty["aicore_subunits"] == []
    assert empty["sync_hazard_detected"] is False
    assert empty["graph_dfx_detected"] is False


def test_ascendc_sync_hang_and_pipe_race_triage_extracts_indicators_and_synccheck():
    plog = "\n".join([
        "[ERROR] RUNTIME(1200,1205):2026-09-22-15:00:01.100.000 "
        "rank[0] deviceId:1 streamID:[12] taskID[44] "
        "kernel_name: FlashAttentionScoreGrad, ErrCode=507014, "
        "desc=[aicore task execute timeout, CrossCoreWaitFlag hang after SetFlag/WaitFlag mismatch on PIPE_ALL]",
        "[ERROR] HCCL(1200,1209):2026-09-22-15:00:15.000.000 "
        "rank[2] ErrCode=EI0002 Notify wait timeout",
    ])

    result = triage_ascend_plog(plog)

    assert result["fault_domain"] == "ascendc_sync_or_pipe_hazard"
    assert result["sync_hazard_detected"] is True
    assert "SetFlag/WaitFlag_mismatch" in result["sync_hazard_indicators"]
    assert "CrossCoreWaitFlag_hang" in result["sync_hazard_indicators"]
    assert result["failed_operator"] == "FlashAttentionScoreGrad"
    assert result["cascading_timeout_detected"] is True
    assert result["cascading_rank"] == "2"
    assert "npu-check" in result["recommended_tools"]
    assert any("npu-check --tool synccheck" in step for step in result["remediation_steps"])
    assert any("msaicerr -i" in step for step in result["remediation_steps"])


def test_aicore_exception_subunits_extracted_across_mte_vec_cube_biu():
    plog = "\n".join([
        "[ERROR] RUNTIME(3100,3102):2026-09-22-15:10:00.000.000 "
        "rank[0] deviceId:0 streamID:[4] taskID[99] "
        "failed operator: GroupedMatmulSwiglu, ErrCode=507035, EZ9999 "
        "mte2_err gm_to_ub read overflow, mte3_err ub_to_gm, "
        "vec error info: 0x100200, cube error info: 0x04, biu error info: 0x01",
    ])

    result = triage_ascend_plog(plog)

    assert result["fault_domain"] == "aicore_memory_or_tiling"
    assert result["error_codes"] == ["EZ9999", "507035"]
    assert result["aicore_subunits"] == ["MTE2", "MTE3", "VEC", "CUBE", "BIU"]
    assert result["exception_subunits"] == ["MTE2", "MTE3", "VEC", "CUBE", "BIU"]
    assert result["aic_error_bitmask"] == 0x100200
    assert {"asys", "msaicerr", "npu-objdump", "msnpureport"} <= set(result["recommended_tools"])
    assert any("npu-objdump --dump-elf" in step for step in result["remediation_steps"])


def test_torch_npugraph_ex_and_ge_dynamic_shape_dfx_triage():
    capture_plog = "\n".join([
        "[ERROR] TORCHAIR(5001,5002):2026-09-22-16:00:00.000.000 "
        "rank[0] deviceId:0 torch.npugraph_ex BackendCompilerFailed: "
        "aclgraph stream capture fail on operator: AddRmsNorm, ErrCode=EZ3002",
    ])
    dyn_shape_plog = "\n".join([
        "[ERROR] GE(6001,6002):2026-09-22-16:05:00.000.000 "
        "rank[0] deviceId:2 op_name: MoeInitRoutingV2, ErrCode=E10009, EZ1013 "
        "dynamic shape range inference and TilingFailed",
    ])

    capture = triage_ascend_plog(capture_plog)
    dyn_shape = triage_ascend_plog(dyn_shape_plog)

    assert capture["fault_domain"] == "aclgraph_capture_or_stream"
    assert capture["graph_dfx_detected"] is True
    assert capture["graph_dfx_stage"] == "compile_or_capture"
    assert "TORCH_COMPILE_DEBUG=1" in capture["recommended_tools"]
    assert any("TORCH_COMPILE_DEBUG=1" in step for step in capture["remediation_steps"])

    assert dyn_shape["fault_domain"] == "aclgraph_capture_or_stream"
    assert dyn_shape["graph_dfx_detected"] is True
    assert dyn_shape["graph_dfx_stage"] == "dynamic_shape"
    assert dyn_shape["error_codes"] == ["E10009", "EZ1013"]
    assert any("DUMP_GE_GRAPH=2" in step for step in dyn_shape["remediation_steps"])


def test_hccl_config_and_runtime_stream_faults_recommend_targeted_commands():
    hccl_cfg = triage_ascend_plog(
        "[ERROR] HCCL(7001,7002):2026-09-22-17:00:00.000.000 "
        "rank[3] deviceId:3 ErrCode=EI0004 ranktable topology parse failed"
    )
    rts_fault = triage_ascend_plog(
        "[ERROR] RUNTIME(8001,8002):2026-09-22-17:01:00.000.000 "
        "rank[0] deviceId:0 streamID:[15] ErrCode=EE9999, EE8888 internal event fault"
    )

    assert hccl_cfg["fault_domain"] == "hccl_collective_timeout_or_skew"
    assert hccl_cfg["error_codes"] == ["EI0004"]
    assert "HCCL_ENTRY_LOG_ENABLE=1" in hccl_cfg["recommended_tools"]
    assert any("HCCL_ENTRY_LOG_ENABLE=1" in step for step in hccl_cfg["remediation_steps"])

    assert rts_fault["fault_domain"] == "runtime_stream_or_event"
    assert rts_fault["error_codes"] == ["EE9999", "EE8888"]
    assert "ASCEND_LAUNCH_BLOCKING=1" in rts_fault["recommended_tools"]


def test_vllm_ascend_five_model_fault_categories_and_remediations():
    aclgraph_res = triage_ascend_plog(
        "[ERROR] RUNTIME(9001,9002):2026-09-22-18:00:00.000.000 "
        "rank[0] deviceId:0 AclmdlRICaptureBegin failed: ErrCode=207008, EL0009 "
        "Insufficient_Stream_Resources during cudagraph_capture_sizes capture"
    )
    assert aclgraph_res["fault_domain"] == "vllm_ascend_aclgraph_capture_failure"
    assert aclgraph_res["model_fault_category"] == "vllm_ascend_aclgraph_capture_failure"
    assert {"EL0009", "207008"} <= set(aclgraph_res["error_codes"])
    assert any("FULL_DECODE_ONLY" in step for step in aclgraph_res["remediation_steps"])

    idle_dp_res = triage_ascend_plog(
        "[ERROR] HCCL(9101,9102):2026-09-22-18:01:00.000.000 "
        "rank[1] deviceId:1 ErrCode=EI0002 HcclAlltoAllV / npu_moe_distribute_dispatch_v2 timeout: "
        "idle dp rank 0 skipped forward with active_batch == 0 and num_tokens_across_dp mismatch"
    )
    assert idle_dp_res["fault_domain"] == "vllm_ascend_idle_dp_collective_hang"
    assert idle_dp_res["model_fault_category"] == "vllm_ascend_idle_dp_collective_hang"
    assert any("num_tokens_across_dp" in step for step in idle_dp_res["remediation_steps"])
    assert any("recompute_scheduler_enable" in step for step in idle_dp_res["remediation_steps"])

    quant_res = triage_ascend_plog(
        "[ERROR] VLLM_ASCEND(9201,9202):2026-09-22-18:02:00.000.000 "
        "rank[0] KeyError: msmodelslim quant_model_description.json missing weight_scale / dequant_scale "
        "for AscendW8A8LinearMethod on model.layers.0.mlp.down_proj"
    )
    assert quant_res["fault_domain"] == "msmodelslim_quant_weight_or_scale_mismatch"
    assert quant_res["model_fault_category"] == "msmodelslim_quant_weight_or_scale_mismatch"
    assert "msmodelslim" in quant_res["recommended_tools"]
    assert any("quant_model_description.json" in step for step in quant_res["remediation_steps"])

    numeric_res = triage_ascend_plog(
        "[ERROR] VLLM_ASCEND(9301,9302):2026-09-22-18:03:00.000.000 "
        "rank[0] deviceId:0 NaN/Inf overflow and cosine similarity determinism drift detected in MLA attention"
    )
    assert numeric_res["fault_domain"] == "numerical_nan_inf_or_determinism_drift"
    assert numeric_res["model_fault_category"] == "numerical_nan_inf_or_determinism_drift"
    assert "msprobe" in numeric_res["recommended_tools"]
    assert any("HCCL_DETERMINISTIC=true" in step for step in numeric_res["remediation_steps"])

    mooncake_res = triage_ascend_plog(
        "[ERROR] VLLM_ASCEND(9401,9402):2026-09-22-18:04:00.000.000 "
        "rank[0] deviceId:0 MooncakeConnector kv_transfer register_memory / ibv_reg_mr failed "
        "between kv_producer and kv_consumer with cp_kv_cache_interleave_size mismatch"
    )
    assert mooncake_res["fault_domain"] == "mooncake_or_kv_pool_transfer_fault"
    assert mooncake_res["model_fault_category"] == "mooncake_or_kv_pool_transfer_fault"
    assert any("MooncakeLayerwiseConnector" in step for step in mooncake_res["remediation_steps"])
    assert any("cp_kv_cache_interleave_size" in step for step in mooncake_res["remediation_steps"])


def test_codex_distilled_custom_op_multithread_context_and_shm_triage():
    custom_op_res = triage_ascend_plog(
        "[ERROR] OPAPI(9501,9502):2026-09-22-18:05:00.000.000 "
        "rank[0] deviceId:0 libascend_kernels.so / opapi_math symbol shadowing: "
        "ACLNN_ERR_PARAM_NULLPTR (161001) and FIA errcode 264 tiling offset out of range"
    )
    assert custom_op_res["fault_domain"] == "custom_op_abi_or_symbol_shadowing"
    assert custom_op_res["model_fault_category"] == "custom_op_abi_or_symbol_shadowing"
    assert "161001" in custom_op_res["error_codes"]
    assert any("opapi_math" in step for step in custom_op_res["remediation_steps"])
    assert any("actual_seq_lengths_kv" in step for step in custom_op_res["remediation_steps"])

    ctx_res = triage_ascend_plog(
        "[ERROR] RUNTIME(9601,9602):2026-09-22-18:06:00.000.000 "
        "rank[2] deviceId:2 ErrCode=107003 ACL_ERROR_RT_CONTEXT_NULL: "
        "context is null in ThreadPoolExecutor worker before set_device"
    )
    assert ctx_res["fault_domain"] == "multithread_npu_context_or_rogue_device0"
    assert ctx_res["model_fault_category"] == "multithread_npu_context_or_rogue_device0"
    assert "107003" in ctx_res["error_codes"]
    assert any("torch.npu.set_device" in step for step in ctx_res["remediation_steps"])
    assert any("perf probe" in step for step in ctx_res["remediation_steps"])

    shm_res = triage_ascend_plog(
        "[ERROR] VLLM(9701,9702):2026-09-22-18:07:00.000.000 "
        "rank[0] multiproc_executor UnpicklingError in shm_broadcast MessageQueue chunk corrupt "
        "causing EngineCore ZMQ hang"
    )
    assert shm_res["fault_domain"] == "enginecore_shm_or_zmq_frame_corruption"
    assert shm_res["model_fault_category"] == "enginecore_shm_or_zmq_frame_corruption"
    assert any("shm_broadcast" in step for step in shm_res["remediation_steps"])
