"""Layer 4 whole-network and multi-dimensional parallel decomposition.

[Role]
  Bridge ``mindie_profiler.theoretical_model`` and the pinned TensorCast
  provider, preserving ``ideal_hardware_lower_bound_ms``, ``model_predicted_ms``
  and ``measured_device_ms``.  Decompose a step across DP x TP x EP x PP x DCP,
  DSpark acceptance length and Prefix Cache hit rate.

[Invariants]
  * The ideal roofline comes only from ``modeling_inputs``; the provider
    prediction comes only from a versioned provider; measured time comes only
    from the caller.  None of the three is ever relabeled as another.
  * A missing provider, missing workload metadata or missing step components
    returns ``unmodeled_reason``; no local formula is presented as an upstream
    prediction and no shard factor is invented.
  * Parallel divisors are explicit expressions (``tp``, ``tp*dcp``, ``ep``,
    numeric) evaluated against the supplied degrees.  DP is a replica axis and
    does not divide a replica's step time.
  * DSpark divides tokens per target step by ``1 + acceptance_length``; Prefix
    Cache saves only the explicitly supplied reusable prefill fraction.
  * The provider is constructed only when an explicit source and device are
    supplied; importing this module never imports torch or msmodeling.

[Timeouts & Locks]
  Report arithmetic is pure.  Provider construction may run bounded git probes
  inside ``mindie_profiler.tensor_cast_provider``; prediction executes upstream
  analytic code and may import torch.  No lock is held here and no subprocess is
  started by this module directly.

[Side-Effect Policy]
  Reads explicit source paths through the provider boundary.  Does not install
  packages, write files, mutate sys.path, or change device state.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from mindie_profiler.modeling_inputs import KernelInputs, estimate_ideal_hardware_lower_bound
from mindie_profiler.theoretical_model import (
    ModelingRequest,
    ProviderUnmodeledError,
    predict_with_provider,
    probe_msmodeling_adapter,
)

from .comm import default_transfer_channels, evaluate_transfer_plan

SCHEMA_VERSION = "mindie-sim-net/1"
PARALLEL_KEYS = ("dp", "tp", "ep", "pp", "dcp")
DEFAULT_SHARD_DIVISORS = {
    "compute": "tp",
    "attention": "tp*dcp",
    "moe": "ep",
    "communication": "1",
    "overhead": "1",
}


class NetError(ValueError):
    """Stable network-model validation failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive_int(value: Any, reason: str) -> int:
    if isinstance(value, bool):
        raise NetError(reason)
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise NetError(reason) from exc
    if numeric <= 0:
        raise NetError(reason)
    return numeric


def _parallel_degrees(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise NetError("parallel_missing")
    result: dict[str, int] = {}
    for key in PARALLEL_KEYS:
        result[key] = _positive_int(value.get(key), f"parallel_invalid:{key}")
    return result


def _eval_divisor(expression: Any, parallel: Mapping[str, int]) -> float:
    if _finite(expression) and float(expression) > 0:
        return float(expression)
    text = str(expression).strip().lower()
    if text in {"1", "none", "one"}:
        return 1.0
    tokens = [token.strip() for token in text.split("*") if token.strip()]
    if not tokens:
        raise NetError("shard_divisor_empty")
    product = 1.0
    for token in tokens:
        try:
            numeric = float(token)
        except ValueError:
            numeric = None
        if numeric is not None and numeric > 0:
            product *= numeric
        elif token == "dp":
            product *= 1.0
        elif token in parallel:
            product *= float(parallel[token])
        else:
            raise NetError(f"shard_divisor_unknown:{token}")
    return product


def decompose_step(
    model_spec: Mapping[str, Any],
    parallel: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(model_spec, Mapping):
        raise NetError("model_spec_invalid")
    degrees = _parallel_degrees(parallel if parallel is not None else model_spec.get("parallel"))
    components = model_spec.get("step_components_ms")
    if not isinstance(components, Mapping) or not components:
        return {
            "status": "unmodeled",
            "unmodeled_reason": "missing_step_components_ms",
            "parallel": degrees,
            "step_ms": None,
            "tpot_ms": None,
            "ttft_ms": None,
        }
    divisors_raw = model_spec.get("shard_divisors", DEFAULT_SHARD_DIVISORS)
    if not isinstance(divisors_raw, Mapping):
        return {
            "status": "unmodeled",
            "unmodeled_reason": "shard_divisors_invalid",
            "parallel": degrees,
            "step_ms": None,
        }
    per_component: dict[str, float] = {}
    component_divisors: dict[str, float] = {}
    for key, raw in components.items():
        if not _finite(raw) or float(raw) < 0:
            return {
                "status": "unmodeled",
                "unmodeled_reason": f"step_component_invalid:{key}",
                "parallel": degrees,
                "step_ms": None,
            }
        try:
            divisor = _eval_divisor(divisors_raw.get(key, 1), degrees)
        except NetError as exc:
            return {
                "status": "unmodeled",
                "unmodeled_reason": exc.reason,
                "parallel": degrees,
                "step_ms": None,
            }
        component_divisors[str(key)] = divisor
        per_component[str(key)] = float(raw) / divisor
    pp_bubble = model_spec.get("pp_bubble_ms", 0.0)
    if not _finite(pp_bubble) or float(pp_bubble) < 0:
        return {
            "status": "unmodeled",
            "unmodeled_reason": "pp_bubble_ms_invalid",
            "parallel": degrees,
            "step_ms": None,
        }
    imbalance = model_spec.get("dp_load_imbalance_factor", 1.0)
    if not _finite(imbalance) or float(imbalance) <= 0:
        return {
            "status": "unmodeled",
            "unmodeled_reason": "dp_load_imbalance_factor_invalid",
            "parallel": degrees,
            "step_ms": None,
        }
    per_token_ms = sum(per_component.values())
    step_ms = per_token_ms / degrees["pp"] * float(imbalance) + float(pp_bubble)
    assumptions = [
        "step components are caller-supplied milliseconds",
        "compute is divided by TP, attention by TP*DCP, MoE by EP (explicit defaults)",
        "DP is a replica axis and does not divide one replica's step time",
        "PP divides per-token work and adds the explicit pp_bubble_ms",
        "DP load imbalance is applied only when explicitly supplied",
    ]
    result: dict[str, Any] = {
        "status": "modeled",
        "parallel": degrees,
        "step_components_ms": {str(key): float(value) for key, value in components.items()},
        "shard_divisors": {str(key): value for key, value in divisors_raw.items()},
        "component_divisors": component_divisors,
        "per_component_ms": per_component,
        "per_token_ms": per_token_ms,
        "pp_bubble_ms": float(pp_bubble),
        "dp_load_imbalance_factor": float(imbalance),
        "step_ms": step_ms,
        "assumptions": assumptions,
    }
    dspark = model_spec.get("dspark")
    if isinstance(dspark, Mapping):
        acceptance = dspark.get("acceptance_length")
        if _finite(acceptance) and float(acceptance) >= 0:
            tokens = 1.0 + float(acceptance)
            result["dspark"] = {
                "acceptance_length": float(acceptance),
                "tokens_per_target_step": tokens,
                "tpot_ms": step_ms / tokens,
                "formula": "TPOT = step_ms / (1 + acceptance_length)",
            }
        else:
            result["dspark"] = {"status": "unmodeled", "reason": "acceptance_length_invalid"}
    else:
        result["dspark"] = {"status": "not_supplied"}
    prefix = model_spec.get("prefix_cache")
    if isinstance(prefix, Mapping):
        hit_rate = prefix.get("hit_rate")
        prefill = prefix.get("prefill_ms")
        reusable = prefix.get("reusable_fraction", 1.0)
        if (
            _finite(hit_rate) and 0 <= float(hit_rate) <= 1
            and _finite(prefill) and float(prefill) >= 0
            and _finite(reusable) and 0 <= float(reusable) <= 1
        ):
            saved = float(prefill) * float(hit_rate) * float(reusable)
            result["prefix_cache"] = {
                "hit_rate": float(hit_rate),
                "reusable_fraction": float(reusable),
                "prefill_ms": float(prefill),
                "saved_prefill_ms": saved,
                "ttft_ms": float(prefill) - saved,
                "formula": "TTFT = prefill_ms * (1 - hit_rate * reusable_fraction)",
            }
        else:
            result["prefix_cache"] = {"status": "unmodeled", "reason": "prefix_cache_inputs_invalid"}
    else:
        result["prefix_cache"] = {"status": "not_supplied"}
    transfer_plan = model_spec.get("transfer_plan")
    if isinstance(transfer_plan, Mapping):
        family = str(model_spec.get("arch_family", "A5"))
        dies_pkg = int(model_spec.get("dies_per_package", 2 if family.upper() == "A3" else 1))
        channels = default_transfer_channels(
            family,
            dies_per_package=dies_pkg,
            overrides=transfer_plan.get("channel_overrides") if isinstance(transfer_plan.get("channel_overrides"), Mapping) else None,
        )
        evaluated = evaluate_transfer_plan(transfer_plan, channels, dies_per_package=dies_pkg)
        known_us = [
            float(item["lower_bound_us"])
            for item in (*evaluated["transfers"], *evaluated["collectives"])
            if item.get("lower_bound_us") is not None
        ]
        total_transfer_ms = sum(known_us) / 1000.0
        overlap_compute_ms = float(transfer_plan.get("overlapped_compute_ms", 0.0) or 0.0)
        exposed_transfer_ms = max(0.0, total_transfer_ms - overlap_compute_ms)
        result["communication_and_transfers"] = {
            **evaluated,
            "total_transfer_lower_bound_ms": total_transfer_ms,
            "overlapped_compute_ms": overlap_compute_ms,
            "exposed_transfer_ms": exposed_transfer_ms,
            "step_ms_with_exposed_transfer": step_ms + exposed_transfer_ms,
        }
    return result


def provider_bridge(
    source_path: str | None,
    device_name: str | None,
    *,
    expected_commit: str | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    if not source_path:
        probe = probe_msmodeling_adapter()
        return None, {
            "status": "unavailable",
            "reason": probe.reason,
            "probe": probe.as_dict(),
        }
    if not device_name:
        return None, {"status": "unavailable", "reason": "device_name_required"}
    try:
        from mindie_profiler.tensor_cast_provider import TensorCastAnalyticProvider
    except Exception as exc:  # noqa: BLE001 - optional provider boundary
        return None, {"status": "unavailable", "reason": f"provider_import_failed:{type(exc).__name__}"}
    try:
        kwargs = {"device_name": device_name}
        if expected_commit:
            kwargs["expected_commit"] = expected_commit
        provider = TensorCastAnalyticProvider(source_path, **kwargs)
    except ProviderUnmodeledError as exc:
        return None, {"status": "unavailable", "reason": exc.reason}
    except Exception as exc:  # noqa: BLE001 - provider construction is untrusted
        return None, {"status": "unavailable", "reason": f"provider_construction_error:{type(exc).__name__}"}
    return provider, {"status": "available", "provider": provider.status()}


def _kernel_inputs(model_spec: Mapping[str, Any]) -> KernelInputs | None:
    raw = model_spec.get("kernel_inputs")
    if raw is None and any(key in model_spec for key in ("shape", "dtype", "hardware")):
        raw = model_spec
    if not isinstance(raw, Mapping):
        return None
    try:
        return KernelInputs.from_mapping(raw)
    except (TypeError, ValueError):
        return None


def net_report(
    model_spec: Mapping[str, Any],
    parallel: Mapping[str, Any] | None = None,
    *,
    provider: Any = None,
    measured_device_ms: float | None = None,
    provider_source: str | None = None,
    device_name: str | None = None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    if not isinstance(model_spec, Mapping):
        raise NetError("model_spec_invalid")
    provider_status: dict[str, Any]
    if provider is None and provider_source:
        provider, provider_status = provider_bridge(provider_source, device_name, expected_commit=expected_commit)
    elif provider is None:
        probe = probe_msmodeling_adapter()
        provider_status = {"status": "unavailable", "reason": probe.reason, "probe": probe.as_dict()}
    else:
        identity = getattr(provider, "identity", None)
        provider_status = {
            "status": "injected",
            "provider_name": getattr(identity, "provider_name", None),
            "provider_version": getattr(identity, "provider_version", None),
        }
    inputs = _kernel_inputs(model_spec)
    measured_input = measured_device_ms if measured_device_ms is not None else model_spec.get("measured_device_ms")
    measurement_limitation = None
    if measured_input is not None and (not _finite(measured_input) or float(measured_input) < 0):
        measurement_limitation = "invalid_measured_device_ms"
        measured_input = None
    ideal_bound: float | None = None
    predicted: float | None = None
    measured: float | None = float(measured_input) if measured_input is not None else None
    alignment_reason: str | None = None
    ideal_reason: str | None = None
    prediction_reason: str | None = None
    model_provider: str | None = None
    model_assumptions: str | None = None
    evidence: list[str] = []
    if inputs is not None:
        lower = estimate_ideal_hardware_lower_bound(inputs)
        if lower.status == "modeled":
            ideal_bound = lower.lower_bound_ms
        else:
            ideal_reason = lower.unmodeled_reason
        request = ModelingRequest(inputs=inputs, measured_device_ms=measured)
        prediction = predict_with_provider(request, provider)
        if prediction.status == "modeled":
            predicted = prediction.predicted_ms
            model_provider = (
                f"{prediction.provider_name}@{prediction.provider_version}"
                if prediction.provider_name and prediction.provider_version
                else None
            )
            model_assumptions = prediction.model_assumptions
            evidence = list(prediction.evidence)
        else:
            prediction_reason = prediction.unmodeled_reason
        alignment_reason = prediction_reason or ideal_reason
    else:
        alignment_reason = "missing_kernel_inputs"
        ideal_reason = "missing_kernel_inputs"
        prediction_reason = "missing_kernel_inputs"
    try:
        step = decompose_step(model_spec, parallel)
    except NetError as exc:
        step = {"status": "unmodeled", "unmodeled_reason": exc.reason, "step_ms": None}
    if step.get("status") == "modeled" and predicted is not None and ideal_bound is not None:
        status = "modeled"
    elif predicted is not None or ideal_bound is not None or step.get("status") == "modeled" or measured is not None:
        status = "partial"
    else:
        status = "unmodeled"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "ideal_hardware_lower_bound_ms": ideal_bound,
        "model_predicted_ms": predicted,
        "measured_device_ms": measured,
        "model_provider": model_provider,
        "model_assumptions": model_assumptions,
        "model_evidence": evidence,
        "unmodeled_reason": alignment_reason,
        "ideal_unmodeled_reason": ideal_reason,
        "prediction_unmodeled_reason": prediction_reason,
        "measurement_limitation": measurement_limitation,
        "three_value_separation": {
            "ideal_source": "mindie_profiler.modeling_inputs.estimate_ideal_hardware_lower_bound",
            "prediction_source": model_provider or provider_status.get("reason") or alignment_reason,
            "measured_source": "caller_supplied_measured_device_ms",
            "values_are_distinct_fields": True,
        },
        "provider": provider_status,
        "step_decomposition": step,
        "assumptions": [
            "ideal bound is a single-kernel roofline, not a whole-network measurement",
            "provider prediction is upstream AnalyticPerformanceModel output",
            "measured time is caller evidence and is never synthesized",
            "the step decomposition is a transparent caller-component calculation",
        ],
    }


__all__ = [
    "SCHEMA_VERSION",
    "NetError",
    "PARALLEL_KEYS",
    "DEFAULT_SHARD_DIVISORS",
    "decompose_step",
    "provider_bridge",
    "net_report",
]
