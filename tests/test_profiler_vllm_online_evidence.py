"""Real-contract tests for the tiny vLLM online evidence helper.

These tests do not start a server or initialize NPU.  They exercise the
generated model/config, exact command contract, budget arithmetic, artifact
classification, and process-tree cleanup semantics.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mindie_profiler.vllm_online_evidence import (
    OnlineEvidenceError,
    build_vllm_argv,
    completion_payload,
    find_profile_roots,
    load_source_bundle,
    load_source_pair,
    scan_profile_artifacts,
    terminate_process_group,
    tiny_llama_config,
    validate_budget,
    write_tiny_llama_model,
)
from mindie_profiler.vllm_online_memory import (
    MEMORY_REQUIRED_PHASES,
    classify_online_status,
    partial_result_limitations,
    scan_memory_phase_probes,
    validate_memory_probe_options,
)


def test_tiny_llama_model_writes_only_real_local_config(tmp_path: Path):
    model = write_tiny_llama_model(tmp_path / "tiny-llama", max_position_embeddings=128)
    config_path = Path(model["config_path"])
    assert config_path.is_file()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["architectures"] == ["LlamaForCausalLM"]
    assert config["model_type"] == "llama"
    assert config["hidden_size"] == 64
    assert config["num_hidden_layers"] == 1
    assert config["vocab_size"] == 128
    assert config["max_position_embeddings"] == 128
    assert model["config_sha256"]
    assert model["weights"]["kind"] == "vllm_load_format_dummy"
    assert model["weights"]["deterministic"] is True
    assert sorted(path.name for path in Path(model["path"]).iterdir()) == ["config.json"]


def test_tiny_llama_config_rejects_inconsistent_heads():
    with pytest.raises(ValueError, match="divisible"):
        tiny_llama_config(hidden_size=64, num_attention_heads=3)
    with pytest.raises(ValueError, match="positive"):
        tiny_llama_config(num_key_value_heads=0)


def test_budget_must_cover_all_stages_and_cleanup():
    kwargs = dict(
        readiness_timeout=300.0,
        control_timeout=180.0,
        request_timeout=60.0,
        profile_requests=2,
        db_export_timeout=300.0,
        cleanup_grace=10.0,
        cleanup_kill_timeout=5.0,
        overhead=30.0,
    )
    with pytest.raises(OnlineEvidenceError) as caught:
        validate_budget(total_timeout=100.0, **kwargs)
    assert caught.value.code == "budget_too_small"
    plan = validate_budget(total_timeout=1305.0, **kwargs)
    assert plan["stage_sum_s"] == 1305.0
    assert plan["margin_s"] == 0.0
    assert plan["stages"]["request_count"] == 5
    assert plan["stages"]["control_calls"] == 2
    memory_plan = validate_budget(total_timeout=1310.0, **kwargs, memory_phase_overhead=5.0)
    assert memory_plan["stage_sum_s"] == 1310.0
    assert memory_plan["stages"]["memory_phase_overhead"] == 5.0
    cold_plan = validate_budget(total_timeout=1073.0, **{**kwargs, "request_timeout": 2.0},
                                pre_request_timeout=60.0)
    assert cold_plan["stage_sum_s"] == 1073.0
    with pytest.raises(OnlineEvidenceError) as caught:
        validate_budget(total_timeout=2000.0, **{**kwargs, "cleanup_grace": 0})
    assert caught.value.code == "invalid_budget"


def test_completion_payload_uses_token_ids_and_no_tokenizer_text():
    payload = completion_payload("tiny-llama", [1, 5, 9], max_tokens=8)
    assert payload["prompt"] == [1, 5, 9]
    assert payload["max_tokens"] == 8
    assert payload["return_token_ids"] is True
    assert payload["add_special_tokens"] is False
    assert payload["ignore_eos"] is True
    with pytest.raises(ValueError):
        completion_payload("tiny-llama", [], max_tokens=8)


def test_vllm_argv_uses_installed_flags_and_absolute_profiler_dir(tmp_path: Path):
    argv = build_vllm_argv(
        python=sys.executable,
        model_dir=tmp_path / "model",
        served_model_name="tiny-llama",
        host="127.0.0.1",
        port=18765,
        profiler_dir=tmp_path / "profile",
        max_model_len=128,
        gpu_memory_utilization=0.2,
    )
    assert argv[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    assert "--load-format" in argv and argv[argv.index("--load-format") + 1] == "dummy"
    assert "--skip-tokenizer-init" in argv
    assert "--enforce-eager" in argv
    assert argv[argv.index("--port") + 1] == "18765"
    profiler_config = json.loads(argv[argv.index("--profiler-config") + 1])
    assert profiler_config["profiler"] == "torch"
    assert Path(profiler_config["torch_profiler_dir"]).is_absolute()
    assert profiler_config["torch_profiler_with_stack"] is False


def test_memory_probe_options_cover_control_requests_and_stop():
    plan = validate_memory_probe_options(
        interval_ms=20.0, max_samples=16384, stop_timeout_s=2.0,
        profile_requests=64, request_timeout=2.0, control_timeout=60.0,
    )
    assert plan["coverage_s"] == 327.68
    assert plan["required_window_s"] == 250.0
    assert plan["coverage_margin_s"] > 0
    with pytest.raises(OnlineEvidenceError) as caught:
        validate_memory_probe_options(
            interval_ms=100.0, max_samples=2048, stop_timeout_s=2.0,
            profile_requests=64, request_timeout=2.0, control_timeout=60.0,
        )
    assert caught.value.code == "sampler_window_too_short"


def test_memory_phase_scan_reports_named_probes_without_zero_fill(tmp_path: Path):
    root = tmp_path / "memory-phases"
    root.mkdir()
    for index, phase in enumerate(MEMORY_REQUIRED_PHASES):
        (root / f"{index:03d}_{phase}.json").write_text(json.dumps({
            "phase": phase, "sequence": index, "status": "ok", "pid": 42,
            "device": 0, "snapshot": {"allocator": {"allocated_bytes": 100}},
            "query_bounds": {"monotonic_start_ns": 1, "monotonic_end_ns": 2},
        }), encoding="utf-8")
    (root / "sampler.json").write_text(json.dumps({
        "status": "ok", "summary": {"device_peak": {"used_bytes_max": 300}},
        "scope": {"export_excluded": True},
    }), encoding="utf-8")
    (root / "worker.json").write_text(json.dumps({"base_worker_class": "NPUWorker"}), encoding="utf-8")
    (root / "model-state.json").write_text(json.dumps({"status": "ok", "sha256": "abc"}),
                                           encoding="utf-8")
    scan = scan_memory_phase_probes(root)
    assert scan["status"] == "ok"
    assert scan["missing_phases"] == []
    assert scan["phases"]["after_model_load"]["pid"] == 42
    missing = scan_memory_phase_probes(root, required_phases=MEMORY_REQUIRED_PHASES + ("extra",))
    assert missing["status"] == "partial"
    assert missing["missing_phases"] == ["extra"]
    (root / "model-state.json").unlink()
    assert scan_memory_phase_probes(root)["status"] == "ok"
    assert scan_memory_phase_probes(root, require_model_state=True)["status"] == "partial"
    (root / "errors.json").write_text(json.dumps([{"phase": "after_model_load",
                                                    "code": "probe_failed"}]), encoding="utf-8")
    failed = scan_memory_phase_probes(root)
    assert failed["status"] == "partial"
    assert failed["errors"][0]["code"] == "probe_failed"


def test_partial_result_with_actual_db_does_not_report_db_missing():
    checks = {
        "server_ready": True,
        "pre_request": True,
        "start_profile": True,
        "during_requests": True,
        "stop_profile": True,
        "post_stop_health": True,
        "post_request": True,
        "actual_db_artifact": True,
        "process_cleanup": True,
        "memory_phase_probes": False,
    }
    assert classify_online_status(
        checks, control_start_status="ok", control_stop_status="ok"
    ) == "partial"
    limitations = partial_result_limitations(checks)
    assert [item["code"] for item in limitations] == ["memory_phase_probes_incomplete"]
    assert all(item["code"] != "db_artifact_missing" for item in limitations)

    checks["memory_phase_probes"] = True
    assert classify_online_status(
        checks, control_start_status="ok", control_stop_status="ok"
    ) == "ok"

    checks["actual_db_artifact"] = False
    assert classify_online_status(
        checks, control_start_status="ok", control_stop_status="ok"
    ) == "partial"
    assert partial_result_limitations(checks) == []


def test_artifact_scan_distinguishes_nonempty_db_from_empty_db(tmp_path: Path):
    output = tmp_path / "rank0" / "ASCEND_PROFILER_OUTPUT"
    output.mkdir(parents=True)
    (output / "ascend_pytorch_profiler_0.db").write_bytes(b"db")
    (output / "ascend_pytorch_profiler_1.db").write_bytes(b"")
    (output / "kernel_details.csv").write_text("kernel,time\n", encoding="utf-8")
    (output / "trace_view.json").write_text("[]", encoding="utf-8")
    (tmp_path / "unrelated.txt").write_text("ignored", encoding="utf-8")
    scan = scan_profile_artifacts(tmp_path)
    assert scan["ok"] is True
    assert scan["by_kind"]["db"] == 2
    assert scan["by_kind"]["kernel_details"] == 1
    assert scan["by_kind"]["trace_view"] == 1
    non_empty = [entry for entry in scan["db"] if entry["non_empty"]]
    assert len(non_empty) == 1
    assert non_empty[0]["size_bytes"] == 2


def test_find_profile_roots_uses_actual_raw_directories(tmp_path: Path):
    root = tmp_path / "rank_0_2026_ascend_pt"
    (root / "ASCEND_PROFILER_OUTPUT").mkdir(parents=True)
    (root / "profiler_info_0.json").write_text("{}", encoding="utf-8")
    other = tmp_path / "other" / "ASCEND_PROFILER_OUTPUT"
    other.mkdir(parents=True)
    roots = find_profile_roots(tmp_path)
    assert root in roots
    assert other.parent in roots


def test_isolated_source_bundle_requires_actual_compat_symbol(tmp_path: Path):
    source = tmp_path / "vllm-source"
    (source / "vllm" / "v1" / "core").mkdir(parents=True)
    (source / "vllm" / "_version.py").write_text(
        '__version__ = "0.29.0"\n__version_tuple__ = (0, 29, 0)\n', encoding="utf-8"
    )
    kv = source / "vllm" / "v1" / "core" / "kv_cache_utils.py"
    kv.write_text("def _get_packed_kv_cache_groups(kv_cache_spec):\n    return {}\n", encoding="utf-8")
    metadata = {
        "source_commit": "84030bbe3d74d99bad477a3d2e37a973ccd8865c",
        "source_tarball_sha256": "4ca3089b4c68217b3dfb078a0775ee8f4bf1a0377db3697866b866f7bb88c199",
        "vllm_version": "0.29.0",
        "compat_basis": "vllm-ascend .github/vllm-main-verified.commit",
    }
    (source / "BUNDLE_METADATA.json").write_text(json.dumps(metadata), encoding="utf-8")
    loaded = load_source_bundle(
        source,
        expected_commit=metadata["source_commit"],
        expected_tarball_sha256=metadata["source_tarball_sha256"],
    )
    assert loaded["required_symbol_present"] is True
    assert loaded["vllm_version"] == "0.29.0"
    assert loaded["pythonpath"] == str(source.resolve())

    with pytest.raises(OnlineEvidenceError) as caught:
        load_source_bundle(source, expected_commit="deadbeef")
    assert caught.value.code == "source_commit_mismatch"

    kv.write_text("def _get_packed_kv_cache_layout(kv_cache_groups):\n    return 1\n", encoding="utf-8")
    with pytest.raises(OnlineEvidenceError) as caught:
        load_source_bundle(source)
    assert caught.value.code == "source_api_incompatible"


def test_clean_source_pair_requires_ascend_symbol_and_native_hashes(tmp_path: Path):
    vllm = tmp_path / "vllm-source"
    (vllm / "vllm" / "v1" / "core").mkdir(parents=True)
    (vllm / "vllm" / "_version.py").write_text('__version__ = "0.29.0"\n', encoding="utf-8")
    (vllm / "vllm" / "v1" / "core" / "kv_cache_utils.py").write_text(
        "def _get_packed_kv_cache_groups(kv_cache_spec):\n    return {}\n", encoding="utf-8"
    )
    (vllm / "BUNDLE_METADATA.json").write_text(json.dumps({
        "source_commit": "84030bbe3d74d99bad477a3d2e37a973ccd8865c",
        "source_tarball_sha256": "4ca3089b4c68217b3dfb078a0775ee8f4bf1a0377db3697866b866f7bb88c199",
        "vllm_version": "0.29.0",
        "compat_basis": "pin",
    }), encoding="utf-8")

    ascend = tmp_path / "ascend-source"
    (ascend / "vllm_ascend" / "ops").mkdir(parents=True)
    (ascend / "vllm_ascend" / "ops" / "layernorm.py").write_text(
        "class AscendFusedRMSNormGated(Base):\n    pass\n", encoding="utf-8"
    )
    (ascend / "vllm_ascend" / "utils.py").write_text(
        "from vllm_ascend.ops.layernorm import AscendFusedRMSNormGated\n", encoding="utf-8"
    )
    (ascend / "vllm_ascend" / "_version.py").write_text(
        '__version__ = "0.19.1rc2.dev2265+gb255ab596"\n', encoding="utf-8"
    )
    native = ascend / "vllm_ascend" / "native.so"
    native.write_bytes(b"native-placeholder")
    import hashlib
    native_sha = hashlib.sha256(native.read_bytes()).hexdigest()
    (ascend / "BUNDLE_METADATA.json").write_text(json.dumps({
        "source_commit": "b255ab59662d8c665e5534e926f913b1c1179b1c",
        "source_tarball_sha256": "065066f42c36c7e9f81616145a13603f753a70bdc5981cd744ef2101afda34c5",
        "vllm_ascend_version": "0.19.1rc2.dev2265+gb255ab596",
        "compat_vllm_commit": "84030bbe3d74d99bad477a3d2e37a973ccd8865c",
        "native_libraries": [{"path": "vllm_ascend/native.so", "sha256": native_sha}],
    }), encoding="utf-8")
    pair = load_source_pair(
        vllm, ascend,
        expected_vllm_commit="84030bbe3d74d99bad477a3d2e37a973ccd8865c",
        expected_ascend_commit="b255ab59662d8c665e5534e926f913b1c1179b1c",
    )
    assert pair["source_commits_equal_verified_pin"] is True
    assert pair["vllm_ascend"]["required_symbol_present"] is True
    assert pair["vllm_ascend"]["native_libraries"][0]["sha256"] == native_sha
    assert pair["env"]["PYTHONPATH"].split(":") == [str(vllm), str(ascend)]

    (ascend / "vllm_ascend" / "ops" / "layernorm.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(OnlineEvidenceError) as caught:
        load_source_pair(vllm, ascend)
    assert caught.value.code == "source_api_incompatible"


def test_harness_refuses_to_start_server_when_budget_is_too_small(tmp_path: Path):
    harness = Path(__file__).with_name("live_vllm_profile_acceptance.py")
    output = tmp_path / "online"
    completed = subprocess.run(
        [sys.executable, str(harness), "--run-id", "budget-refusal",
         "--output-dir", str(output), "--total-timeout", "1"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["error"]["code"] == "budget_too_small"
    assert not (output / "server.stdout.log").exists()
    assert not (output / "server.stderr.log").exists()


def test_terminate_process_group_owns_only_its_child():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    cleanup = terminate_process_group(process, grace_s=0.2, kill_timeout_s=0.5)
    assert cleanup["term_sent"] is True
    assert cleanup["exit_code"] is not None
    assert cleanup["cleanup_error"] is None
