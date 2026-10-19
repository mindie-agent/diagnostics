"""Actual pinned TensorCast provider acceptance tests.

Pure contracts need no upstream checkout or torch.  Set
``MINDIE_TEST_MSMODELING_SOURCE`` to a clean pinned checkout for actual upstream
acceptance, and ``MINDIE_TEST_A2_CAPTURE_RESULT`` to the original live result
JSON for measured-kernel comparisons.  Missing external inputs are skipped.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mindie_profiler.enrichment_model import build_c7_three_time_artifact, read_actual_kernels
from mindie_profiler.tensor_cast_provider import (
    PINNED_MSMODELING_COMMIT,
    TensorCastAnalyticProvider,
    main,
    probe_tensor_cast_provider,
)
from mindie_profiler.theoretical_model import (
    KernelInputs,
    ModelingRequest,
    ProviderIdentity,
    ProviderUnmodeledError,
    align_modeling,
    predict_with_provider,
)

SOURCE = Path(os.environ["MINDIE_TEST_MSMODELING_SOURCE"]) if os.environ.get("MINDIE_TEST_MSMODELING_SOURCE") else None
A2_RESULT = Path(os.environ["MINDIE_TEST_A2_CAPTURE_RESULT"]) if os.environ.get("MINDIE_TEST_A2_CAPTURE_RESULT") else None


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


SOURCE_READY = SOURCE is not None and SOURCE.is_dir()
PROVIDER_READY = SOURCE_READY and _torch_available()
requires_source = pytest.mark.skipif(
    not SOURCE_READY,
    reason="set MINDIE_TEST_MSMODELING_SOURCE to run actual pinned-source acceptance",
)
requires_provider = pytest.mark.skipif(
    not PROVIDER_READY,
    reason="explicit pinned msmodeling source or torch is unavailable",
)


@requires_source
def test_probe_verifies_pinned_commit_without_importing_source():
    status = probe_tensor_cast_provider(SOURCE)

    assert status.source_present is True
    assert status.commit == PINNED_MSMODELING_COMMIT
    assert status.commit_matches is True
    assert status.declared_version == "0.2.0"
    assert status.provider_version == f"0.2.0+{PINNED_MSMODELING_COMMIT}"
    assert status.imported is False
    assert status.verified is False
    assert status.reason == "source_commit_verified_clean_probe_only"
    assert status.git_status_clean is True
    assert status.dirty_count == 0


@requires_source
def test_probe_rejects_wrong_pinned_commit_without_importing_source():
    status = probe_tensor_cast_provider(SOURCE, expected_commit="0" * 40)

    assert status.commit_matches is False
    assert status.provider_version is None
    assert status.reason == "source_commit_mismatch"


def test_probe_reports_missing_source_without_importing(tmp_path):
    status = probe_tensor_cast_provider(tmp_path / "missing-msmodeling")

    assert status.source_present is False
    assert status.commit is None
    assert status.imported is False
    assert status.reason == "source_missing"


def test_probe_requires_explicit_source_path():
    status = probe_tensor_cast_provider(None)

    assert status.source_present is False
    assert status.commit is None
    assert status.reason == "source_path_not_configured"


def test_probe_and_provider_reject_dirty_tracked_source(tmp_path):
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git is unavailable")
    source = tmp_path / "msmodeling"
    analytic = source / "tensor_cast" / "performance_model" / "analytic.py"
    analytic.parent.mkdir(parents=True)
    analytic.write_text("# pinned analytic\n", encoding="utf-8")
    (source / "pyproject.toml").write_text(
        '[project]\nname = "msmodeling"\nversion = "0.2.0"\n', encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=source, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()

    analytic.write_text("# modified after pin\n", encoding="utf-8")
    status = probe_tensor_cast_provider(source, expected_commit=commit)
    assert status.commit == commit
    assert status.git_status_clean is False
    assert status.dirty_count and status.dirty_count > 0
    assert status.provider_version is None
    assert status.reason == "source_dirty"
    with pytest.raises(ProviderUnmodeledError, match="source_dirty"):
        TensorCastAnalyticProvider(source, device_name="TEST_DEVICE", expected_commit=commit)


def test_actual_a2_db_kernel_selection_reads_real_p50s_without_torch():
    if A2_RESULT is None or not A2_RESULT.is_file():
        pytest.skip("set MINDIE_TEST_A2_CAPTURE_RESULT to run actual A2 capture acceptance")
    actual = read_actual_kernels(A2_RESULT)
    by_shape = {item["kernel_key"]["input_shapes"]: item for item in actual["kernels"]}

    assert len(actual["kernels"]) == 3
    assert actual["db_schema_version"] == "1.2.0"
    assert len(actual["db_sha256"]) == 64
    assert by_shape['"8,256;512,256;512"']['measured']['value'] == pytest.approx(0.00416)
    assert by_shape['"8,512;256,512;256"']['measured']['value'] == pytest.approx(0.00424)
    assert by_shape['"8,512"']['measured']['value'] == pytest.approx(0.00128)
    assert by_shape['"8,512"']['pmu']['median_by_counter']['aiv_total_time'] > 0


def test_provider_unmodeled_error_reason_is_preserved():
    class _RejectingProvider:
        identity = ProviderIdentity("fixture-provider", "9.9.9")

        def available(self) -> bool:
            return True

        def predict(self, request):
            raise ProviderUnmodeledError("unsupported_layout:NZ")

    # Complete metadata so the provider boundary, rather than input
    # validation, is what rejects the request.
    from mindie_profiler.theoretical_model import HardwareSpec

    inputs = KernelInputs(
        shape=(1, 4, 8),
        dtype="bf16",
        layout="row_major",
        hardware=HardwareSpec("dummy", 1e12, 1e9),
        op_kind="matmul",
    )
    result = predict_with_provider(inputs, _RejectingProvider())

    assert result.status == "unmodeled"
    assert result.unmodeled_reason == "unsupported_layout:NZ"


@requires_provider
@pytest.mark.parametrize("device_name", ["TEST_DEVICE", "ATLAS_800_A2_280T_32G_PCIE"])
def test_provider_matches_direct_upstream_analytic_model_exactly(device_name):
    provider = TensorCastAnalyticProvider(SOURCE, device_name=device_name)
    import torch
    from tensor_cast.device import DeviceProfile
    from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
    from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo

    profile = DeviceProfile.all_device_profiles[device_name]
    hardware = provider.hardware_spec("bfloat16", "matmul")
    assert hardware.peak_flops_per_s == pytest.approx(float(profile.mma_ops[torch.bfloat16]))
    assert hardware.memory_bandwidth_bytes_per_s == pytest.approx(float(profile.memory_bandwidth_bytes_ps))
    gp_hardware = provider.hardware_spec("bfloat16", "rmsnorm")
    assert gp_hardware.peak_flops_per_s == pytest.approx(float(profile.gp_ops[torch.bfloat16]))
    inputs = KernelInputs(
        shape=(8, 16, 4),
        dtype="bfloat16",
        layout="row_major",
        hardware=hardware,
        op_kind="matmul",
    )
    request = ModelingRequest(inputs, measured_device_ms=0.123456)
    prediction = predict_with_provider(request, provider)

    assert prediction.status == "modeled"
    assert prediction.predicted_ms is not None and prediction.predicted_ms > 0
    assert prediction.provider_name == "msmodeling.tensor_cast.analytic"
    assert prediction.provider_version == f"0.2.0+{PINNED_MSMODELING_COMMIT}"
    assert any(PINNED_MSMODELING_COMMIT in item for item in prediction.evidence)
    assert any("op_schema:aten::mm" in item for item in prediction.evidence)
    assert any(f"compute_efficiency:{profile.compute_efficiency}" in item for item in prediction.evidence)
    assert any(f"memory_efficiency:{profile.memory_efficiency}" in item for item in prediction.evidence)
    assert any("device_profile_mapping:caller_explicit_unverified" in item for item in prediction.evidence)
    assert any("raw_peak_flops_for_op:" in item for item in prediction.evidence)
    assert any("raw_memory_bandwidth_bytes_ps:" in item for item in prediction.evidence)

    a = torch.empty((8, 16), device="meta", dtype=torch.bfloat16)
    b = torch.empty((16, 4), device="meta", dtype=torch.bfloat16)
    out = torch.ops.aten.mm.default(a, b)
    direct = AnalyticPerformanceModel(profile).process_op(
        OpInvokeInfo(torch.ops.aten.mm.default, (a, b), {}, out)
    )
    assert prediction.predicted_ms == pytest.approx(direct.execution_time_s * 1000.0, abs=1e-12)

    alignment = align_modeling(request, provider)
    assert alignment.status == "modeled"
    assert alignment.ideal_hardware_lower_bound_ms is not None
    assert alignment.ideal_hardware_lower_bound_ms > 0
    assert alignment.model_predicted_ms == pytest.approx(prediction.predicted_ms)
    assert alignment.measured_device_ms == pytest.approx(0.123456)
    assert alignment.model_provider == f"msmodeling.tensor_cast.analytic@0.2.0+{PINNED_MSMODELING_COMMIT}"
    assert f"device={device_name}" in alignment.model_assumptions
    assert "local_ideal_uses_raw_peaks=true" in alignment.model_assumptions


@requires_provider
def test_provider_addmm_matches_direct_upstream_on_actual_a2_shapes():
    provider = TensorCastAnalyticProvider(SOURCE, device_name="ATLAS_800_A2_280T_32G_PCIE")
    import torch
    from tensor_cast.device import DeviceProfile
    from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
    from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo

    profile = DeviceProfile.all_device_profiles["ATLAS_800_A2_280T_32G_PCIE"]
    inputs = KernelInputs(
        shape=(8, 256, 512),
        dtype="float32",
        layout="ND",
        hardware=provider.hardware_spec("float32", "addmm"),
        op_kind="addmm",
        annotations=(("profile_input_shapes", '"8,256;512,256;512"'),),
    )
    prediction = predict_with_provider(ModelingRequest(inputs), provider)

    assert prediction.status == "modeled"
    assert prediction.predicted_ms is not None and prediction.predicted_ms > 0
    assert any("op_schema:aten::addmm" in item for item in prediction.evidence)

    bias = torch.empty((512,), device="meta", dtype=torch.float32)
    a = torch.empty((8, 256), device="meta", dtype=torch.float32)
    b = torch.empty((512, 256), device="meta", dtype=torch.float32)
    out = torch.ops.aten.addmm.default(bias, a, b.t())
    direct = AnalyticPerformanceModel(profile).process_op(
        OpInvokeInfo(torch.ops.aten.addmm.default, (bias, a, b.t()), {}, out)
    )
    assert prediction.predicted_ms == pytest.approx(direct.execution_time_s * 1000.0, abs=1e-12)

    relu_inputs = KernelInputs(
        shape=(8, 512),
        dtype="float32",
        layout="ND",
        hardware=provider.hardware_spec("float32", "relu"),
        op_kind="relu",
    )
    relu_prediction = predict_with_provider(ModelingRequest(relu_inputs), provider)
    relu_x = torch.empty((8, 512), device="meta", dtype=torch.float32)
    relu_out = torch.ops.aten.relu.default(relu_x)
    relu_direct = AnalyticPerformanceModel(profile).process_op(
        OpInvokeInfo(torch.ops.aten.relu.default, (relu_x,), {}, relu_out)
    )
    assert relu_prediction.status == "modeled"
    assert relu_prediction.predicted_ms == pytest.approx(relu_direct.execution_time_s * 1000.0, abs=1e-12)
    assert "relu_or_copy_default_properties=upstream_memory_bandwidth_bound" in relu_prediction.model_assumptions


@requires_provider
def test_actual_a2_capture_builds_three_time_c7_artifact(tmp_path):
    if A2_RESULT is None or not A2_RESULT.is_file():
        pytest.skip("set MINDIE_TEST_A2_CAPTURE_RESULT to run actual A2 capture acceptance")
    output = tmp_path / "three-time.json"
    artifact = build_c7_three_time_artifact(
        A2_RESULT,
        source_path=SOURCE,
        device_name="ATLAS_800_A2_280T_32G_PCIE",
        output_path=output,
    )

    assert output.is_file()
    assert artifact["source_identity"]["msmodeling_source_commit"] == PINNED_MSMODELING_COMMIT
    assert artifact["source_identity"]["profile_db_schema_version"] == "1.2.0"
    assert len(artifact["source_identity"]["profile_db_sha256"]) == 64
    assert artifact["hardware_assumptions"]["local_ideal_uses_raw_peaks"] is True
    assert artifact["hardware_assumptions"]["device_profile_mapping"] == "caller_explicit_unverified"
    assert artifact["hardware_assumptions"]["soc_mapping_verified"] is False
    by_shape = {item["kernel_key"]["input_shapes"]: item for item in artifact["kernels"]}
    assert set(by_shape) == {'"8,256;512,256;512"', '"8,512;256,512;256"', '"8,512"'}
    for item in artifact["kernels"]:
        assert item["three_time"]["status"] == "modeled"
        assert item["three_time"]["measured_device_ms"] > 0
        assert item["three_time"]["ideal_hardware_lower_bound_ms"] > 0
        assert item["three_time"]["model_predicted_ms"] > 0
        assert item["pmu"]["status"] == "present"
        comparison = item["alignment"]["time_comparison"]
        assert comparison["measured_over_ideal"] == pytest.approx(
            item["three_time"]["measured_device_ms"] / item["three_time"]["ideal_hardware_lower_bound_ms"]
        )
        assert comparison["measured_over_prediction"] == pytest.approx(
            item["three_time"]["measured_device_ms"] / item["three_time"]["model_predicted_ms"]
        )
        assert comparison["descriptive_only"] is True
    assert by_shape['"8,256;512,256;512"']['three_time']['measured_device_ms'] == pytest.approx(0.00416)
    assert by_shape['"8,512;256,512;256"']['three_time']['measured_device_ms'] == pytest.approx(0.00424)
    assert by_shape['"8,512"']['three_time']['measured_device_ms'] == pytest.approx(0.00128)
    assert by_shape['"8,256;512,256;512"']['shape_proof']['weight_tensor']['name'] == "0.weight"
    assert by_shape['"8,512;256,512;256"']['shape_proof']['weight_tensor']['name'] == "2.weight"


@requires_provider
@pytest.mark.parametrize(
    ("op_kind", "shape", "evidence_fragment"),
    [
        ("bmm", (2, 8, 16, 4), "op_schema:aten::bmm"),
        ("rmsnorm", (2, 4, 16), "op_schema:tensor_cast::rms_norm"),
        ("elementwise", (4, 8), "op_schema:aten::relu"),
        ("copy", (4, 8), "op_schema:aten::clone"),
    ],
)
def test_provider_supports_actual_tensor_cast_and_aten_ops(op_kind, shape, evidence_fragment):
    provider = TensorCastAnalyticProvider(SOURCE, device_name="TEST_DEVICE")
    inputs = KernelInputs(
        shape=shape,
        dtype="bfloat16",
        layout="row_major",
        hardware=provider.hardware_spec("bfloat16", op_kind),
        op_kind=op_kind,
    )
    prediction = predict_with_provider(ModelingRequest(inputs), provider)

    assert prediction.status == "modeled"
    assert prediction.predicted_ms is not None and prediction.predicted_ms > 0
    assert any(evidence_fragment in item for item in prediction.evidence)


@requires_provider
def test_provider_rejects_unsupported_inputs_with_explicit_reasons():
    provider = TensorCastAnalyticProvider(SOURCE, device_name="TEST_DEVICE")
    hardware = provider.hardware_spec("bfloat16", "matmul")

    bad_layout = predict_with_provider(
        ModelingRequest(KernelInputs(
            shape=(4, 8, 2), dtype="bfloat16", layout="NZ", hardware=hardware, op_kind="matmul"
        )),
        provider,
    )
    bad_op = predict_with_provider(
        ModelingRequest(KernelInputs(
            shape=(4, 8), dtype="bfloat16", layout="row_major", hardware=hardware, op_kind="conv2d"
        )),
        provider,
    )
    bad_device = predict_with_provider(
        ModelingRequest(KernelInputs(
            shape=(4, 8), dtype="bfloat16", layout="row_major",
            hardware=type(hardware)("OTHER_DEVICE", hardware.peak_flops_per_s, hardware.memory_bandwidth_bytes_per_s),
            op_kind="copy",
        )),
        provider,
    )

    assert bad_layout.unmodeled_reason == "unsupported_layout:NZ"
    assert bad_op.unmodeled_reason == "unsupported_op_kind:conv2d"
    assert bad_device.unmodeled_reason == "hardware_device_mismatch:expected=TEST_DEVICE,got=OTHER_DEVICE"


@requires_provider
@pytest.mark.parametrize("device_name", ["TEST_DEVICE", "ATLAS_800_A2_280T_32G_PCIE"])
def test_acceptance_entrypoint_writes_real_upstream_result(tmp_path, capsys, device_name):
    output = tmp_path / f"provider-acceptance-{device_name}.json"
    code = main([
        "--source", str(SOURCE),
        "--device", device_name,
        "--shape", "8,16,4",
        "--dtype", "bfloat16",
        "--layout", "row_major",
        "--op", "matmul",
        "--output", str(output),
    ])

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "mindie-profiler-provider-acceptance/1"
    assert payload["upstream_executed"] is True
    assert payload["provider"]["verified"] is True
    assert payload["probe"]["verified"] is False
    assert payload["alignment"]["status"] == "modeled"
    assert payload["alignment"]["model_predicted_ms"] > 0
    assert payload["alignment"]["ideal_hardware_lower_bound_ms"] > 0
    assert payload["alignment"]["measured_device_ms"] is None
    assert payload["measured_device_ms_supplied"] is False


@requires_provider
def test_hardware_spec_requires_explicit_op_kind_and_uses_raw_peaks():
    provider = TensorCastAnalyticProvider(SOURCE, device_name="TEST_DEVICE")
    import torch
    from tensor_cast.device import DeviceProfile

    profile = DeviceProfile.all_device_profiles["TEST_DEVICE"]
    with pytest.raises(ProviderUnmodeledError, match="hardware_spec_op_kind_required"):
        provider.hardware_spec("bfloat16")
    matmul = provider.hardware_spec("bfloat16", "matmul")
    rmsnorm = provider.hardware_spec("bfloat16", "rmsnorm")
    assert matmul.peak_flops_per_s == pytest.approx(float(profile.mma_ops[torch.bfloat16]))
    assert rmsnorm.peak_flops_per_s == pytest.approx(float(profile.gp_ops[torch.bfloat16]))
    assert matmul.memory_bandwidth_bytes_per_s == pytest.approx(float(profile.memory_bandwidth_bytes_ps))
    assert rmsnorm.memory_bandwidth_bytes_per_s == pytest.approx(float(profile.memory_bandwidth_bytes_ps))


@requires_provider
def test_provider_does_not_run_git_or_filesystem_scan_per_prediction(monkeypatch):
    provider = TensorCastAnalyticProvider(SOURCE, device_name="TEST_DEVICE")
    inputs = KernelInputs(
        shape=(8, 16, 4),
        dtype="bfloat16",
        layout="row_major",
        hardware=provider.hardware_spec("bfloat16", "matmul"),
        op_kind="matmul",
    )

    def _unexpected_git(*args, **kwargs):
        pytest.fail("provider must not run git per prediction")

    monkeypatch.setattr("mindie_profiler.tensor_cast_provider.subprocess.run", _unexpected_git)
    prediction = predict_with_provider(ModelingRequest(inputs), provider)

    assert prediction.status == "modeled"
    assert prediction.predicted_ms is not None and prediction.predicted_ms > 0


@requires_provider
def test_actual_provider_reaches_analyze_profile_c7_path():
    from mindie_profiler import Window, analyze_profile

    if A2_RESULT is None or not A2_RESULT.is_file():
        pytest.skip("set MINDIE_TEST_A2_CAPTURE_RESULT to run actual A2 capture acceptance")
    actual = read_actual_kernels(A2_RESULT)
    observed = next(row for row in actual["kernels"] if row["kernel_key"]["op_type"] == "Relu")
    name = observed["kernel_key"]["name"]
    provider = TensorCastAnalyticProvider(SOURCE, device_name="ATLAS_800_A2_280T_32G_PCIE")
    inputs = KernelInputs(
        shape=(8, 512),
        dtype="float32",
        layout="ND",
        hardware=provider.hardware_spec("float32", "relu"),
        op_kind="relu",
    )
    assert observed["kernel_key"]["input_shapes"] == '"8,512"'
    assert observed["kernel_key"]["input_dtypes"] == "FLOAT"
    report = analyze_profile(
        A2_RESULT.parent / "profile",
        windows=[Window(row["start_ns"], row["end_ns"], step=row["step"]) for row in actual["steps"]],
        scene={"mode": "inference", "phase": "decode"},
        model_provider=provider,
        model_inputs={name: inputs},
    )

    hotspot = next(item for item in report["top_hotspot_kernels"] if item["kernel_name"] == name)
    alignment = hotspot["modeling_alignment"]
    assert alignment["status"] == "modeled"
    assert alignment["measured_device_ms"] == pytest.approx(observed["measured"]["value"])
    assert alignment["ideal_hardware_lower_bound_ms"] > 0
    assert alignment["model_predicted_ms"] > 0
    assert alignment["model_provider"].startswith("msmodeling.tensor_cast.analytic@0.2.0+")
    assert report["enrichment"]["modeling"]["hotspots_modeled"] == 1
    assert hotspot["wait_anchor_false_hotspot"] is None
    assert hotspot["wait_anchor_assessment"] in {"unverified", "not_available"}


@requires_provider
def test_import_source_is_checked_after_another_provider_wins_the_lock(tmp_path):
    import subprocess
    import sys

    other = tmp_path / "other-msmodeling"
    subprocess.run(["git", "clone", "--quiet", "--shared", str(SOURCE), str(other)], check=True, timeout=10)
    result = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).with_name("provider_import_interleaving.py")), str(SOURCE), str(other)],
        check=True, capture_output=True, text=True, timeout=20,
    )
    payload = json.loads(result.stdout)
    assert payload["first_upstream_executed"] is True
    assert payload["second_status"] == "tensor_cast_already_imported_from_other_source"
