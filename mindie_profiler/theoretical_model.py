"""Version-pinned C7 modeling interface: provider protocol, prediction, alignment.

[Role]
  Define the versioned provider boundary and the three explicitly separated C7
  values: ``ideal_hardware_lower_bound_ms``, ``model_predicted_ms`` and
  ``measured_device_ms``.  Input metadata and the local roofline formula live
  in ``modeling_inputs``; this module accepts only an injected, identity-bearing
  provider for external predictions and never guesses a provider API.

[Invariants]
  * Missing metadata inputs or unavailable workload data return
    ``unmodeled_reason``; model-derived numeric fields stay ``None``.
  * A prediction requires a real provider name, a real version, the exact
    adapter protocol version, the request input signature and non-empty
    evidence.  Provider rejections preserve a stable reason.
  * Placeholder versions, protocol mismatches, signature mismatches and
    provider exceptions are unmodeled; no local roofline value is relabeled as
    an upstream prediction.
  * External provider efficiency is applied by the provider only; the local
    ideal bound uses raw physical peaks from ``modeling_inputs``.

[Timeouts & Locks]
  Pure computation.  No locks, sleeps, I/O, subprocesses, network or caches.

[Side-Effect Policy]
  Read-only ``importlib.util.find_spec`` probe for reporting install state.  The
  probe never imports or executes msmodeling and never claims it is verified.
"""
from __future__ import annotations

import importlib.util
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from .modeling_inputs import (
    IDEAL_BOUND_FORMULA_VERSION,
    HardwareSpec,
    IdealLowerBound,
    KernelInputs,
    estimate_ideal_hardware_lower_bound,
    ideal_hardware_lower_bound,
)

ADAPTER_PROTOCOL_VERSION = "mindie-msmodeling-adapter/1"
MSMODELING_MODULE = "msmodeling"

MODELED = "modeled"
UNMODELED = "unmodeled"

_VERSION_RE = re.compile(
    r"^(?:v?\d+\.\d+\.\d+(?:[.+-][0-9A-Za-z.-]+)*|git:[0-9a-fA-F]{7,40})$"
)
_PLACEHOLDER_VERSIONS = {
    "",
    "unknown",
    "unverified",
    "unavailable",
    "none",
    "null",
    "n/a",
    "na",
    "dev",
    "development",
    "latest",
    "0.0.0",
}
_MISSING = object()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _real_version(version: Any) -> bool:
    if not isinstance(version, str):
        return False
    text = version.strip()
    return text.lower() not in _PLACEHOLDER_VERSIONS and bool(_VERSION_RE.match(text))


@dataclass(frozen=True)
class ProviderIdentity:
    provider_name: str
    provider_version: str
    adapter_protocol_version: str = ADAPTER_PROTOCOL_VERSION

class ProviderUnmodeledError(ValueError):
    """Provider rejected a request with a stable, caller-safe reason code."""

    def __init__(self, reason: str) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)

@dataclass(frozen=True)
class ProviderPrediction:
    predicted_ms: float
    model_assumptions: str
    input_signature: str
    evidence: tuple[str, ...] = ()

@dataclass(frozen=True)
class ModelingRequest:
    inputs: KernelInputs
    measured_device_ms: float | None = None
    measured_source: str | None = None
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.inputs, Mapping):
            object.__setattr__(self, "inputs", KernelInputs.from_mapping(self.inputs))
        if isinstance(self.evidence, str):
            object.__setattr__(self, "evidence", (self.evidence,))
        else:
            object.__setattr__(self, "evidence", tuple(self.evidence))

    @property
    def input_signature(self) -> str:
        return self.inputs.input_signature()

@runtime_checkable
class ModelingProvider(Protocol):
    """Fixed adapter protocol; implementations must provide all three members."""

    @property
    def identity(self) -> ProviderIdentity: ...

    def available(self) -> bool: ...

    def predict(self, request: ModelingRequest) -> ProviderPrediction: ...

@dataclass(frozen=True)
class ModelPrediction:
    status: str
    predicted_ms: float | None
    provider_name: str | None
    provider_version: str | None
    adapter_protocol_version: str | None
    input_signature: str | None
    model_assumptions: str | None
    evidence: tuple[str, ...]
    unmodeled_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model_predicted_ms": self.predicted_ms,
            "model_provider": (
                f"{self.provider_name}@{self.provider_version}"
                if self.provider_name and self.provider_version
                else None
            ),
            "adapter_protocol_version": self.adapter_protocol_version,
            "input_signature": self.input_signature,
            "model_assumptions": self.model_assumptions,
            "evidence": list(self.evidence),
            "unmodeled_reason": self.unmodeled_reason,
        }

def _unmodeled_prediction(reason: str, *, signature: str | None = None) -> ModelPrediction:
    return ModelPrediction(
        status=UNMODELED,
        predicted_ms=None,
        provider_name=None,
        provider_version=None,
        adapter_protocol_version=None,
        input_signature=signature,
        model_assumptions=None,
        evidence=(),
        unmodeled_reason=reason,
    )

def _coerce_request(value: ModelingRequest | KernelInputs | Mapping[str, Any]) -> ModelingRequest:
    if isinstance(value, ModelingRequest):
        return value
    if isinstance(value, KernelInputs):
        return ModelingRequest(inputs=value)
    if isinstance(value, Mapping):
        if "inputs" in value:
            return ModelingRequest(
                inputs=value["inputs"],
                measured_device_ms=value.get("measured_device_ms"),
                measured_source=value.get("measured_source"),
                evidence=value.get("evidence", ()),
            )
        return ModelingRequest(inputs=KernelInputs.from_mapping(value))
    raise TypeError("request must be ModelingRequest, KernelInputs or a mapping")

def _provider_identity(provider: Any) -> tuple[ProviderIdentity | None, str | None]:
    identity = getattr(provider, "identity", None)
    if isinstance(identity, Mapping):
        try:
            identity = ProviderIdentity(**dict(identity))
        except (TypeError, ValueError):
            return None, "provider_identity_invalid"
    if not isinstance(identity, ProviderIdentity):
        return None, "provider_identity_missing"
    if not isinstance(identity.provider_name, str) or not identity.provider_name.strip():
        return None, "provider_name_missing"
    if not _real_version(identity.provider_version):
        return None, "provider_version_not_real"
    if identity.adapter_protocol_version != ADAPTER_PROTOCOL_VERSION:
        return None, (
            "adapter_protocol_mismatch:"
            f"expected={ADAPTER_PROTOCOL_VERSION},got={identity.adapter_protocol_version}"
        )
    return identity, None

def _provider_available(provider: Any) -> tuple[bool | None, str | None]:
    available = getattr(provider, "available", None)
    if available is None:
        return None, "provider_availability_unknown"
    try:
        result = available() if callable(available) else available
    except Exception as exc:  # noqa: BLE001 - provider boundary is untrusted
        return None, f"provider_availability_error:{type(exc).__name__}"
    if not isinstance(result, bool):
        return None, "provider_availability_invalid"
    return result, None

def _prediction_field(prediction: Any, name: str) -> Any:
    if isinstance(prediction, Mapping):
        return prediction.get(name, _MISSING)
    return getattr(prediction, name, _MISSING)

def _prediction_evidence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if value is None:
        return ()
    try:
        items = tuple(str(item).strip() for item in value)
    except TypeError:
        return ()
    return tuple(item for item in items if item)

def predict_with_provider(
    request: ModelingRequest | KernelInputs | Mapping[str, Any],
    provider: Any = None,
) -> ModelPrediction:
    """Call an injected versioned provider, or explain why no prediction exists."""
    try:
        req = _coerce_request(request)
    except (TypeError, ValueError) as exc:
        return _unmodeled_prediction(f"invalid_request:{type(exc).__name__}")
    signature = req.input_signature
    missing = req.inputs.missing_metadata_inputs()
    if missing:
        return _unmodeled_prediction("missing_inputs:" + ",".join(missing), signature=signature)
    if provider is None:
        return _unmodeled_prediction("msmodeling_provider_not_configured", signature=signature)
    identity, identity_error = _provider_identity(provider)
    if identity_error is not None:
        return _unmodeled_prediction(identity_error, signature=signature)
    assert identity is not None
    available, availability_error = _provider_available(provider)
    if availability_error is not None:
        return _unmodeled_prediction(availability_error, signature=signature)
    if not available:
        return _unmodeled_prediction("provider_unavailable", signature=signature)
    try:
        raw = provider.predict(req)
    except ProviderUnmodeledError as exc:
        return _unmodeled_prediction(exc.reason, signature=signature)
    except Exception as exc:  # noqa: BLE001 - provider boundary is untrusted
        return _unmodeled_prediction(f"provider_error:{type(exc).__name__}", signature=signature)
    predicted = _prediction_field(raw, "predicted_ms")
    assumptions = _prediction_field(raw, "model_assumptions")
    returned_signature = _prediction_field(raw, "input_signature")
    evidence = _prediction_evidence(_prediction_field(raw, "evidence"))
    if not _finite(predicted) or float(predicted) <= 0:
        return _unmodeled_prediction("provider_prediction_invalid", signature=signature)
    if not isinstance(assumptions, str) or not assumptions.strip():
        return _unmodeled_prediction("missing_model_assumptions", signature=signature)
    if returned_signature != signature:
        return _unmodeled_prediction("input_signature_mismatch", signature=signature)
    if not evidence:
        return _unmodeled_prediction("missing_provider_evidence", signature=signature)
    return ModelPrediction(
        status=MODELED,
        predicted_ms=float(predicted),
        provider_name=identity.provider_name,
        provider_version=identity.provider_version,
        adapter_protocol_version=identity.adapter_protocol_version,
        input_signature=signature,
        model_assumptions=assumptions.strip(),
        evidence=evidence,
        unmodeled_reason=None,
    )

@dataclass(frozen=True)
class ModelingAlignment:
    status: str
    measured_device_ms: float | None
    ideal_hardware_lower_bound_ms: float | None
    model_predicted_ms: float | None
    model_provider: str | None
    model_provider_version: str | None
    adapter_protocol_version: str | None
    input_signature: str | None
    model_assumptions: str | None
    evidence: tuple[str, ...]
    unmodeled_reason: str | None
    measurement_limitation: str | None = None

    def to_c7_alignment(self) -> dict[str, Any]:
        # These ratios compare a measured duration with declared profile/model
        # values. They do not measure an optimization or verify a board's peaks.
        measured = self.measured_device_ms
        ideal = self.ideal_hardware_lower_bound_ms
        predicted = self.model_predicted_ms
        return {
            "status": MODELED if self.status == MODELED else UNMODELED,
            "ideal_hardware_lower_bound_ms": self.ideal_hardware_lower_bound_ms,
            "model_predicted_ms": self.model_predicted_ms,
            "measured_device_ms": self.measured_device_ms,
            "model_provider": self.model_provider,
            "model_assumptions": self.model_assumptions,
            "unmodeled_reason": self.unmodeled_reason,
            "input_signature": self.input_signature,
            "model_evidence": list(self.evidence),
            "measurement_limitation": self.measurement_limitation,
            "time_comparison": {
                "claim_scope": "declared_hardware_profile",
                "descriptive_only": True,
                "physical_hardware_mapping_verified": False,
                "measured_over_ideal": (
                    measured / ideal
                    if measured is not None and ideal is not None and ideal > 0
                    else None
                ),
                "measured_over_prediction": (
                    measured / predicted
                    if measured is not None and predicted is not None and predicted > 0
                    else None
                ),
            },
        }

    as_c7_dict = to_c7_alignment

def _coerce_measured(value: Any) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    if not _finite(value) or float(value) < 0:
        return None, "invalid_measured_device_ms"
    return float(value), None

def align_modeling(
    request: ModelingRequest | KernelInputs | Mapping[str, Any],
    provider: Any = None,
    *,
    measured_device_ms: float | None = None,
    measured_source: str | None = None,
) -> ModelingAlignment:
    """Produce the C7 three-way alignment; partial evidence is never invented."""
    try:
        req = _coerce_request(request)
    except (TypeError, ValueError) as exc:
        return ModelingAlignment(
            status=UNMODELED,
            measured_device_ms=None,
            ideal_hardware_lower_bound_ms=None,
            model_predicted_ms=None,
            model_provider=None,
            model_provider_version=None,
            adapter_protocol_version=None,
            input_signature=None,
            model_assumptions=None,
            evidence=(),
            unmodeled_reason=f"invalid_request:{type(exc).__name__}",
        )
    lower = estimate_ideal_hardware_lower_bound(req.inputs)
    prediction = predict_with_provider(req, provider)
    measured_value = req.measured_device_ms if measured_device_ms is None else measured_device_ms
    measured, measurement_limitation = _coerce_measured(measured_value)
    if lower.status != MODELED:
        return ModelingAlignment(
            status=UNMODELED,
            measured_device_ms=measured,
            ideal_hardware_lower_bound_ms=None,
            model_predicted_ms=None,
            model_provider=None,
            model_provider_version=None,
            adapter_protocol_version=None,
            input_signature=req.input_signature,
            model_assumptions=None,
            evidence=(),
            unmodeled_reason=lower.unmodeled_reason,
            measurement_limitation=measurement_limitation,
        )
    if prediction.status != MODELED:
        return ModelingAlignment(
            status=UNMODELED,
            measured_device_ms=measured,
            ideal_hardware_lower_bound_ms=lower.lower_bound_ms,
            model_predicted_ms=None,
            model_provider=None,
            model_provider_version=None,
            adapter_protocol_version=None,
            input_signature=req.input_signature,
            model_assumptions=None,
            evidence=(),
            unmodeled_reason=prediction.unmodeled_reason,
            measurement_limitation=measurement_limitation,
        )
    return ModelingAlignment(
        status=MODELED,
        measured_device_ms=measured,
        ideal_hardware_lower_bound_ms=lower.lower_bound_ms,
        model_predicted_ms=prediction.predicted_ms,
        model_provider=f"{prediction.provider_name}@{prediction.provider_version}",
        model_provider_version=prediction.provider_version,
        adapter_protocol_version=prediction.adapter_protocol_version,
        input_signature=prediction.input_signature,
        model_assumptions=prediction.model_assumptions,
        evidence=prediction.evidence,
        unmodeled_reason=None,
        measurement_limitation=measurement_limitation,
    )

@dataclass(frozen=True)
class MsmodelingAdapterStatus:
    adapter_protocol_version: str
    installed: bool
    verified: bool
    provider_name: str | None
    provider_version: str | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter_protocol_version": self.adapter_protocol_version,
            "installed": self.installed,
            "verified": self.verified,
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "reason": self.reason,
        }

def probe_msmodeling_adapter() -> MsmodelingAdapterStatus:
    """Report install state only; no source-verified adapter is bundled here."""
    try:
        spec = importlib.util.find_spec(MSMODELING_MODULE)
    except Exception as exc:  # noqa: BLE001 - probe must not raise into callers
        return MsmodelingAdapterStatus(
            adapter_protocol_version=ADAPTER_PROTOCOL_VERSION,
            installed=False,
            verified=False,
            provider_name=None,
            provider_version=None,
            reason=f"msmodeling_probe_failed:{type(exc).__name__}",
        )
    if spec is None:
        return MsmodelingAdapterStatus(
            adapter_protocol_version=ADAPTER_PROTOCOL_VERSION,
            installed=False,
            verified=False,
            provider_name=None,
            provider_version=None,
            reason="msmodeling_not_installed",
        )
    return MsmodelingAdapterStatus(
        adapter_protocol_version=ADAPTER_PROTOCOL_VERSION,
        installed=True,
        verified=False,
        provider_name=None,
        provider_version=None,
        reason="msmodeling_provider_unverified",
    )
model_kernel = predict_with_provider
msmodeling_adapter_status = probe_msmodeling_adapter
align = align_modeling
__all__ = [
    "ADAPTER_PROTOCOL_VERSION",
    "IDEAL_BOUND_FORMULA_VERSION",
    "MODELED",
    "UNMODELED",
    "HardwareSpec",
    "KernelInputs",
    "IdealLowerBound",
    "ProviderIdentity",
    "ProviderUnmodeledError",
    "ProviderPrediction",
    "ModelingRequest",
    "ModelingProvider",
    "ModelPrediction",
    "ModelingAlignment",
    "MsmodelingAdapterStatus",
    "estimate_ideal_hardware_lower_bound",
    "ideal_hardware_lower_bound",
    "predict_with_provider",
    "model_kernel",
    "align_modeling",
    "align",
    "probe_msmodeling_adapter",
    "msmodeling_adapter_status",
]
