"""C7 modeling inputs and the local ideal-hardware roofline lower bound.

[Role]
  Own the explicit Shape/dtype/layout/hardware/workload records consumed by
  ``theoretical_model`` and compute the local single-kernel roofline lower
  bound.  This module never calls an external provider and never consumes
  provider evidence.

[Invariants]
  * Missing Shape/dtype/layout/hardware/workload inputs return
    ``unmodeled_reason`` with no numeric bound; no placeholder value is made up.
  * The ideal bound uses caller-supplied raw physical peaks only.  It does not
    apply an empirical compute/memory efficiency or an upstream model result.
  * Workload derivation is explicit per op kind; matmul, copy, elementwise and
    rmsnorm use only their declared shape/dtype formulas.
  * The bound is a single-kernel roofline lower bound, never a whole-network
    lower bound and never proof that measured time must exceed it.

[Timeouts & Locks]
  Pure computation.  No locks, sleeps, I/O, subprocesses, network or caches.

[Side-Effect Policy]
  No filesystem, network, global-state mutation or provider import/execution.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

IDEAL_BOUND_FORMULA_VERSION = "roofline-max-v1"

_DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "bool": 1,
}
_DTYPE_ALIASES = {
    "fp16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
    "float": "float32",
    "fp64": "float64",
    "double": "float64",
    "i8": "int8",
    "u8": "uint8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "dt_bf16": "bfloat16",
    "dt_float16": "float16",
    "dt_float32": "float32",
    "dt_float": "float32",
    "dt_int8": "int8",
}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _normalise_dtype(dtype: str) -> str:
    key = dtype.strip().lower().replace("-", "_").replace(" ", "_")
    return _DTYPE_ALIASES.get(key, key)


def _canonical_annotations(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), str(v)) for k, v in value.items()))
    return tuple(sorted((str(k), str(v)) for k, v in value))


@dataclass(frozen=True)
class HardwareSpec:
    """User-supplied raw physical peaks for the ideal roofline bound."""

    name: str
    peak_flops_per_s: float
    memory_bandwidth_bytes_per_s: float

    def validation_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not isinstance(self.name, str) or not self.name.strip():
            errors.append("hardware_name")
        if not _finite(self.peak_flops_per_s) or float(self.peak_flops_per_s) <= 0:
            errors.append("hardware_peak_flops_per_s")
        if not _finite(self.memory_bandwidth_bytes_per_s) or float(self.memory_bandwidth_bytes_per_s) <= 0:
            errors.append("hardware_memory_bandwidth_bytes_per_s")
        return tuple(errors)

    def canonical(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "peak_flops_per_s": self.peak_flops_per_s,
            "memory_bandwidth_bytes_per_s": self.memory_bandwidth_bytes_per_s,
        }


@dataclass(frozen=True)
class KernelInputs:
    """All inputs required before a C7 modeling alignment may be attempted."""

    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    layout: str | None = None
    hardware: HardwareSpec | None = None
    op_kind: str | None = None
    flops: float | None = None
    memory_bytes: float | None = None
    kernel_name: str | None = None
    rank: int | None = None
    annotations: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.shape, list):
            object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "annotations", _canonical_annotations(self.annotations))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "KernelInputs":
        hardware = value.get("hardware")
        if isinstance(hardware, Mapping):
            hardware = HardwareSpec(**dict(hardware))
        shape = value.get("shape")
        if isinstance(shape, list):
            shape = tuple(shape)
        return cls(
            shape=shape,
            dtype=value.get("dtype"),
            layout=value.get("layout"),
            hardware=hardware,
            op_kind=value.get("op_kind"),
            flops=value.get("flops"),
            memory_bytes=value.get("memory_bytes", value.get("bytes")),
            kernel_name=value.get("kernel_name"),
            rank=value.get("rank"),
            annotations=value.get("annotations", ()),
        )

    def _shape_error(self) -> str | None:
        if self.shape is None:
            return "missing_shape"
        if not isinstance(self.shape, tuple) or not self.shape:
            return "invalid_shape"
        if any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
            for dim in self.shape
        ):
            return "invalid_shape"
        return None

    def _dtype_error(self) -> str | None:
        if self.dtype is None or (isinstance(self.dtype, str) and not self.dtype.strip()):
            return "missing_dtype"
        if not isinstance(self.dtype, str):
            return "invalid_dtype"
        normalized = _normalise_dtype(self.dtype)
        if normalized not in _DTYPE_BYTES:
            return f"unsupported_dtype:{normalized}"
        return None

    def _layout_error(self) -> str | None:
        if self.layout is None:
            return "missing_layout"
        if not isinstance(self.layout, str) or not self.layout.strip():
            return "missing_layout"
        return None

    def _hardware_error(self) -> str | None:
        if self.hardware is None:
            return "missing_hardware"
        if not isinstance(self.hardware, HardwareSpec):
            return "invalid_hardware"
        errors = self.hardware.validation_errors()
        if errors:
            return "invalid_hardware:" + ",".join(errors)
        return None

    def _explicit_workload_error(self) -> str | None:
        if self.flops is None and self.memory_bytes is None:
            return None
        if self.flops is None:
            return "missing_flops"
        if self.memory_bytes is None:
            return "missing_memory_bytes"
        if not _finite(self.flops) or float(self.flops) < 0:
            return "invalid_flops"
        if not _finite(self.memory_bytes) or float(self.memory_bytes) < 0:
            return "invalid_memory_bytes"
        return None

    def _workload_error(self) -> str | None:
        explicit = self._explicit_workload_error()
        if explicit is not None:
            return explicit
        if self.flops is not None or self.memory_bytes is not None:
            return None
        if self.op_kind is None or not str(self.op_kind).strip():
            return "missing_workload"
        kind = str(self.op_kind).strip().lower()
        if kind in {"matmul", "gemm", "mm", "bmm"}:
            return None if len(self.shape or ()) in {3, 4} else "invalid_matmul_shape"
        if kind == "addmm":
            return None if len(self.shape or ()) == 3 else "invalid_addmm_shape"
        if kind in {"copy", "memcpy", "clone"}:
            return None
        if kind in {"rmsnorm", "rms_norm"}:
            return None if len(self.shape or ()) >= 1 else "invalid_rmsnorm_shape"
        if kind in {"elementwise", "relu"}:
            return None if len(self.shape or ()) >= 1 else "invalid_elementwise_shape"
        return f"unsupported_op_kind:{kind}"

    def missing_metadata_inputs(self) -> tuple[str, ...]:
        problems = (
            self._shape_error(),
            self._dtype_error(),
            self._layout_error(),
            self._hardware_error(),
        )
        return tuple(item for item in problems if item)

    def missing_inputs(self) -> tuple[str, ...]:
        return self.missing_metadata_inputs() + tuple(
            item for item in (self._workload_error(),) if item
        )

    def workload(self) -> tuple[float, float, str] | None:
        """Return (flops, memory_bytes, source) or None when under-specified."""
        if self.missing_inputs():
            return None
        if self.flops is not None and self.memory_bytes is not None:
            return float(self.flops), float(self.memory_bytes), "explicit"
        assert self.shape is not None and self.dtype is not None
        dtype_bytes = _DTYPE_BYTES[_normalise_dtype(self.dtype)]
        kind = str(self.op_kind).strip().lower()
        if kind in {"matmul", "gemm", "mm", "bmm"}:
            dims = self.shape
            if len(dims) == 3:
                batch, m, k, n = 1, dims[0], dims[1], dims[2]
            else:
                batch, m, k, n = dims
            flops = 2.0 * batch * m * n * k
            memory = (batch * m * k + batch * k * n + batch * m * n) * dtype_bytes
            return float(flops), float(memory), "matmul_contiguous"
        if kind == "addmm":
            m, k, n = self.shape
            flops = 2.0 * m * n * k
            memory = (m * k + n * k + n + m * n) * dtype_bytes
            return float(flops), float(memory), "addmm_transposed_bias"
        if kind in {"copy", "memcpy", "clone"}:
            numel = math.prod(self.shape)
            return 0.0, float(2 * numel * dtype_bytes), "copy_read_write"
        if kind in {"rmsnorm", "rms_norm"}:
            numel = math.prod(self.shape)
            row_width = self.shape[-1]
            return (
                float(5 * numel),
                float((2 * numel + row_width) * dtype_bytes),
                "rmsnorm_gp_5n",
            )
        if kind in {"elementwise", "relu"}:
            numel = math.prod(self.shape)
            return 0.0, float(2 * numel * dtype_bytes), "elementwise_read_write"
        return None

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape) if self.shape is not None else None,
            "dtype": _normalise_dtype(self.dtype) if isinstance(self.dtype, str) else self.dtype,
            "layout": self.layout,
            "hardware": self.hardware.canonical() if isinstance(self.hardware, HardwareSpec) else None,
            "op_kind": self.op_kind,
            "flops": self.flops,
            "memory_bytes": self.memory_bytes,
            "kernel_name": self.kernel_name,
            "rank": self.rank,
            "annotations": [list(item) for item in self.annotations],
        }

    def input_signature(self) -> str:
        payload = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coerce_inputs(value: KernelInputs | Mapping[str, Any]) -> KernelInputs:
    if isinstance(value, KernelInputs):
        return value
    if isinstance(value, Mapping):
        return KernelInputs.from_mapping(value)
    raise TypeError("inputs must be KernelInputs or a mapping")


@dataclass(frozen=True)
class IdealLowerBound:
    status: str
    lower_bound_ms: float | None
    compute_ms: float | None
    memory_ms: float | None
    bottleneck: str | None
    formula_version: str
    assumptions: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    unmodeled_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ideal_hardware_lower_bound_ms": self.lower_bound_ms,
            "compute_ms": self.compute_ms,
            "memory_ms": self.memory_ms,
            "bottleneck": self.bottleneck,
            "formula_version": self.formula_version,
            "assumptions": list(self.assumptions),
            "missing_inputs": list(self.missing_inputs),
            "unmodeled_reason": self.unmodeled_reason,
        }


def estimate_ideal_hardware_lower_bound(
    inputs: KernelInputs | Mapping[str, Any],
) -> IdealLowerBound:
    """Compute a single-kernel roofline lower bound or state why it is absent."""
    try:
        spec = _coerce_inputs(inputs)
    except (TypeError, ValueError) as exc:
        return IdealLowerBound(
            status="unmodeled",
            lower_bound_ms=None,
            compute_ms=None,
            memory_ms=None,
            bottleneck=None,
            formula_version=IDEAL_BOUND_FORMULA_VERSION,
            assumptions=(),
            missing_inputs=("invalid_inputs",),
            unmodeled_reason=f"invalid_inputs:{type(exc).__name__}",
        )
    missing = spec.missing_inputs()
    if missing:
        return IdealLowerBound(
            status="unmodeled",
            lower_bound_ms=None,
            compute_ms=None,
            memory_ms=None,
            bottleneck=None,
            formula_version=IDEAL_BOUND_FORMULA_VERSION,
            assumptions=(),
            missing_inputs=missing,
            unmodeled_reason="missing_inputs:" + ",".join(missing),
        )
    workload = spec.workload()
    if workload is None:
        return IdealLowerBound(
            status="unmodeled",
            lower_bound_ms=None,
            compute_ms=None,
            memory_ms=None,
            bottleneck=None,
            formula_version=IDEAL_BOUND_FORMULA_VERSION,
            assumptions=(),
            missing_inputs=("missing_workload",),
            unmodeled_reason="missing_inputs:missing_workload",
        )
    assert spec.hardware is not None
    flops, memory_bytes, source = workload
    compute_ms = flops / float(spec.hardware.peak_flops_per_s) * 1000.0
    memory_ms = memory_bytes / float(spec.hardware.memory_bandwidth_bytes_per_s) * 1000.0
    lower_ms = max(compute_ms, memory_ms)
    bottleneck = "memory" if memory_ms > compute_ms else "compute"
    assumptions = [
        f"formula={IDEAL_BOUND_FORMULA_VERSION}",
        "single_kernel_no_launch_dependency_or_cache_cost",
        "peak_compute_and_memory_fully_overlapped",
        "hardware_peaks_are_user_supplied_raw_physical_values",
        f"workload_source={source}",
    ]
    if source in {"matmul_contiguous", "addmm_transposed_bias", "copy_read_write", "rmsnorm_gp_5n", "elementwise_read_write"}:
        assumptions.append("layout_is_treated_as_contiguous_for_byte_count")
    if source == "addmm_transposed_bias":
        assumptions.append("addmm_profile_shape_order_a_mk_b_nk_bias_n")
    if source == "rmsnorm_gp_5n":
        assumptions.append("rmsnorm_gp_ops_from_upstream_cost_utils_5n")
    if source == "elementwise_read_write":
        assumptions.append("elementwise_memory_only_no_compute")
    return IdealLowerBound(
        status="modeled",
        lower_bound_ms=lower_ms,
        compute_ms=compute_ms,
        memory_ms=memory_ms,
        bottleneck=bottleneck,
        formula_version=IDEAL_BOUND_FORMULA_VERSION,
        assumptions=tuple(assumptions),
        missing_inputs=(),
        unmodeled_reason=None,
    )


ideal_hardware_lower_bound = estimate_ideal_hardware_lower_bound

__all__ = [
    "IDEAL_BOUND_FORMULA_VERSION",
    "HardwareSpec",
    "KernelInputs",
    "IdealLowerBound",
    "estimate_ideal_hardware_lower_bound",
    "ideal_hardware_lower_bound",
]
