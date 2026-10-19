"""Pure-logic tests for the version-pinned C7 modeling interface.

These are formula/protocol fixtures only. They do not assert that any NPU run,
msmodeling installation or hardware measurement happened on this machine.
"""
from __future__ import annotations

import math

import pytest

from mindie_profiler import theoretical_model as tm


def _hardware() -> tm.HardwareSpec:
    return tm.HardwareSpec("logic-test-device", 1e12, 1e9)


def _matmul_inputs() -> tm.KernelInputs:
    return tm.KernelInputs(
        shape=(2, 4, 8),
        dtype="bf16",
        layout="row_major",
        hardware=_hardware(),
        op_kind="matmul",
    )


class _Provider:
    def __init__(
        self,
        *,
        name: str = "fixture-only-provider",
        version: str = "9.9.9",
        protocol: str = tm.ADAPTER_PROTOCOL_VERSION,
        available: bool = True,
        predicted_ms: float = 0.0009,
        assumptions: str = "fixture-only formula assumptions",
        evidence: tuple[str, ...] = ("source:fixture",),
        signature: str | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.identity = tm.ProviderIdentity(name, version, protocol)
        self._available = available
        self._predicted_ms = predicted_ms
        self._assumptions = assumptions
        self._evidence = evidence
        self._signature = signature
        self._raises = raises
        self.calls = 0

    def available(self) -> bool:
        return self._available

    def predict(self, request: tm.ModelingRequest) -> tm.ProviderPrediction:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return tm.ProviderPrediction(
            predicted_ms=self._predicted_ms,
            model_assumptions=self._assumptions,
            input_signature=self._signature or request.input_signature,
            evidence=self._evidence,
        )


def test_ideal_bound_is_roofline_and_keeps_bottleneck_separate():
    result = tm.estimate_ideal_hardware_lower_bound(_matmul_inputs())

    assert result.status == tm.MODELED
    assert result.compute_ms == pytest.approx(128 / 1e12 * 1000)
    assert result.memory_ms == pytest.approx(112 / 1e9 * 1000)
    assert result.lower_bound_ms == pytest.approx(result.memory_ms)
    assert result.bottleneck == "memory"
    assert result.formula_version == tm.IDEAL_BOUND_FORMULA_VERSION
    assert result.unmodeled_reason is None


@pytest.mark.parametrize(
    ("field", "value", "reason_fragment"),
    [
        ("shape", None, "missing_shape"),
        ("dtype", None, "missing_dtype"),
        ("layout", None, "missing_layout"),
        ("hardware", None, "missing_hardware"),
    ],
)
def test_missing_required_input_never_produces_bound(field, value, reason_fragment):
    kwargs = {
        "shape": (2, 4, 8),
        "dtype": "bf16",
        "layout": "row_major",
        "hardware": _hardware(),
        "op_kind": "matmul",
    }
    kwargs[field] = value
    result = tm.estimate_ideal_hardware_lower_bound(tm.KernelInputs(**kwargs))

    assert result.status == tm.UNMODELED
    assert result.lower_bound_ms is None
    assert result.compute_ms is None
    assert result.memory_ms is None
    assert reason_fragment in result.unmodeled_reason
    assert reason_fragment in result.missing_inputs


def test_unknown_dtype_and_workload_are_explicitly_unmodeled():
    bad_dtype = tm.KernelInputs(
        shape=(2, 4), dtype="complex_magic", layout="row_major",
        hardware=_hardware(), op_kind="copy",
    )
    no_workload = tm.KernelInputs(
        shape=(2, 4), dtype="bf16", layout="row_major",
        hardware=_hardware(),
    )

    dtype_result = tm.estimate_ideal_hardware_lower_bound(bad_dtype)
    workload_result = tm.estimate_ideal_hardware_lower_bound(no_workload)

    assert dtype_result.status == tm.UNMODELED
    assert "unsupported_dtype" in dtype_result.unmodeled_reason
    assert workload_result.status == tm.UNMODELED
    assert "missing_workload" in workload_result.unmodeled_reason


def test_explicit_workload_is_used_without_name_guessing():
    inputs = tm.KernelInputs(
        shape=(3, 5), dtype="fp32", layout="nd",
        hardware=_hardware(), flops=1000.0, memory_bytes=4000.0,
    )
    result = tm.estimate_ideal_hardware_lower_bound(inputs)

    assert result.status == tm.MODELED
    assert result.compute_ms == pytest.approx(1e-6)
    assert result.memory_ms == pytest.approx(4e-3)
    assert result.bottleneck == "memory"
    assert "workload_source=explicit" in result.assumptions


def test_provider_prediction_needs_metadata_but_ideal_bound_also_needs_workload():
    metadata_only = tm.KernelInputs(
        shape=(1024, 1024), dtype="bf16", layout="row_major", hardware=_hardware(),
    )

    prediction = tm.predict_with_provider(metadata_only, _Provider())
    bound = tm.estimate_ideal_hardware_lower_bound(metadata_only)

    assert prediction.status == tm.MODELED
    assert prediction.predicted_ms == pytest.approx(0.0009)
    assert bound.status == tm.UNMODELED
    assert "missing_workload" in bound.unmodeled_reason


def test_prediction_and_measured_are_three_distinct_c7_values():
    provider = _Provider()
    request = tm.ModelingRequest(_matmul_inputs(), measured_device_ms=0.002)

    alignment = tm.align_modeling(request, provider)

    assert alignment.status == tm.MODELED
    assert alignment.ideal_hardware_lower_bound_ms == pytest.approx(0.000112)
    assert alignment.model_predicted_ms == pytest.approx(0.0009)
    assert alignment.measured_device_ms == pytest.approx(0.002)
    payload = alignment.to_c7_alignment()
    assert payload["status"] == "modeled"
    assert payload["model_provider"] == "fixture-only-provider@9.9.9"
    assert payload["unmodeled_reason"] is None
    assert payload["input_signature"] == request.input_signature
    comparison = payload["time_comparison"]
    assert comparison["measured_over_ideal"] == pytest.approx(0.002 / 0.000112)
    assert comparison["measured_over_prediction"] == pytest.approx(0.002 / 0.0009)
    assert comparison["physical_hardware_mapping_verified"] is False


@pytest.mark.parametrize("version", ["", "unknown", "unverified", "dev", "0.0.0", "latest"])
def test_placeholder_provider_version_is_rejected_before_prediction(version):
    provider = _Provider(version=version)

    result = tm.predict_with_provider(_matmul_inputs(), provider)

    assert result.status == tm.UNMODELED
    assert result.predicted_ms is None
    assert provider.calls == 0


def test_protocol_mismatch_is_rejected_before_prediction():
    provider = _Provider(protocol="mindie-msmodeling-adapter/999")

    result = tm.predict_with_provider(_matmul_inputs(), provider)

    assert result.status == tm.UNMODELED
    assert "adapter_protocol_mismatch" in result.unmodeled_reason
    assert provider.calls == 0


def test_provider_unavailable_or_missing_evidence_is_unmodeled():
    unavailable = _Provider(available=False)
    no_evidence = _Provider(evidence=())

    unavailable_result = tm.predict_with_provider(_matmul_inputs(), unavailable)
    evidence_result = tm.predict_with_provider(_matmul_inputs(), no_evidence)

    assert unavailable_result.status == tm.UNMODELED
    assert unavailable_result.unmodeled_reason == "provider_unavailable"
    assert evidence_result.status == tm.UNMODELED
    assert evidence_result.unmodeled_reason == "missing_provider_evidence"


def test_signature_mismatch_provider_exception_and_missing_provider_are_unmodeled():
    mismatch = _Provider(signature="sha256:not-the-request")
    exploding = _Provider(raises=RuntimeError("provider crashed"))

    mismatch_result = tm.predict_with_provider(_matmul_inputs(), mismatch)
    exception_result = tm.predict_with_provider(_matmul_inputs(), exploding)
    none_result = tm.predict_with_provider(_matmul_inputs(), None)

    assert mismatch_result.status == tm.UNMODELED
    assert mismatch_result.unmodeled_reason == "input_signature_mismatch"
    assert exception_result.status == tm.UNMODELED
    assert exception_result.unmodeled_reason == "provider_error:RuntimeError"
    assert none_result.status == tm.UNMODELED
    assert none_result.unmodeled_reason == "msmodeling_provider_not_configured"


def test_missing_inputs_do_not_call_provider():
    provider = _Provider()
    inputs = tm.KernelInputs(shape=(2, 4, 8), dtype="bf16", layout=None, hardware=_hardware(), op_kind="matmul")

    result = tm.predict_with_provider(inputs, provider)

    assert result.status == tm.UNMODELED
    assert result.predicted_ms is None
    assert "missing_layout" in result.unmodeled_reason
    assert provider.calls == 0


def test_alignment_without_provider_is_unmodeled_and_keeps_no_fake_difference():
    alignment = tm.align_modeling(_matmul_inputs(), None, measured_device_ms=1.0)

    assert alignment.status == tm.UNMODELED
    assert alignment.model_predicted_ms is None
    assert alignment.ideal_hardware_lower_bound_ms is not None
    assert alignment.unmodeled_reason == "msmodeling_provider_not_configured"
    assert alignment.to_c7_alignment()["model_provider"] is None
    comparison = alignment.to_c7_alignment()["time_comparison"]
    assert comparison["measured_over_ideal"] > 0
    assert comparison["measured_over_prediction"] is None


def test_invalid_measured_value_is_a_measurement_limitation_not_a_model_failure():
    alignment = tm.align_modeling(
        tm.ModelingRequest(_matmul_inputs()),
        _Provider(),
        measured_device_ms=math.nan,
    )

    assert alignment.status == tm.MODELED
    assert alignment.measured_device_ms is None
    assert alignment.measurement_limitation == "invalid_measured_device_ms"
    assert alignment.to_c7_alignment()["ideal_hardware_lower_bound_ms"] is not None


def test_adapter_probe_is_explicitly_unverified(monkeypatch):
    monkeypatch.setattr(tm.importlib.util, "find_spec", lambda name: None)
    missing = tm.probe_msmodeling_adapter()
    assert missing.installed is False
    assert missing.verified is False
    assert missing.provider_version is None
    assert missing.reason == "msmodeling_not_installed"

    monkeypatch.setattr(tm.importlib.util, "find_spec", lambda name: object())
    installed = tm.probe_msmodeling_adapter()
    assert installed.installed is True
    assert installed.verified is False
    assert installed.provider_version is None
    assert installed.reason == "msmodeling_provider_unverified"


def test_provider_unmodeled_error_reason_is_preserved():
    class _RejectingProvider:
        identity = tm.ProviderIdentity("fixture-only-provider", "9.9.9")

        def available(self) -> bool:
            return True

        def predict(self, request):
            raise tm.ProviderUnmodeledError("unsupported_layout:NZ")

    result = tm.predict_with_provider(_matmul_inputs(), _RejectingProvider())

    assert result.status == tm.UNMODELED
    assert result.predicted_ms is None
    assert result.unmodeled_reason == "unsupported_layout:NZ"


def test_rmsnorm_and_elementwise_ideal_workloads_are_source_derived():
    rmsnorm = tm.KernelInputs(
        shape=(2, 4, 16), dtype="bf16", layout="row_major",
        hardware=_hardware(), op_kind="rmsnorm",
    )
    elementwise = tm.KernelInputs(
        shape=(4, 8), dtype="bf16", layout="row_major",
        hardware=_hardware(), op_kind="elementwise",
    )

    rms = tm.estimate_ideal_hardware_lower_bound(rmsnorm)
    elem = tm.estimate_ideal_hardware_lower_bound(elementwise)

    assert rms.status == tm.MODELED
    assert rms.compute_ms == pytest.approx(640 / 1e12 * 1000)
    assert rms.memory_ms == pytest.approx(544 / 1e9 * 1000)
    assert "rmsnorm_gp_ops_from_upstream_cost_utils_5n" in rms.assumptions
    assert elem.status == tm.MODELED
    assert elem.compute_ms == pytest.approx(0.0)
    assert elem.memory_ms == pytest.approx(128 / 1e9 * 1000)
    assert "elementwise_memory_only_no_compute" in elem.assumptions


def test_exported_dt_dtype_alias_is_normalized_for_ideal_bound():
    inputs = tm.KernelInputs(
        shape=(2, 4, 8), dtype="DT_BF16", layout="ND",
        hardware=_hardware(), op_kind="matmul",
    )

    result = tm.estimate_ideal_hardware_lower_bound(inputs)

    assert result.status == tm.MODELED
    assert "workload_source=matmul_contiguous" in result.assumptions
