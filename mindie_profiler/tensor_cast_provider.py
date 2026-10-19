"""Pinned TensorCast analytic provider boundary for C7 model predictions.

[Role]
  Load the actual read-only ``msmodeling`` source at a pinned commit and invoke
  its ``tensor_cast.performance_model.analytic.AnalyticPerformanceModel`` via
  ``OpInvokeInfo`` for a small, explicit set of op signatures.  The provider
  returns the upstream execution-time prediction in milliseconds; the local
  roofline bound remains a separate value owned by ``modeling_inputs``.

[Invariants]
  * Import and model execution happen only after an explicit provider
    construction.  A probe validates explicit source/commit/version and bounded
    git status and never imports torch, tensor_cast, or executes a model.
  * Source is read-only and must be a clean pinned checkout.  Dirty tracked or
    untracked source is rejected; a modified HEAD is never presented as pinned.
  * Local hardware specs use declared DeviceProfile ``mma_ops``/``gp_ops`` and
    bandwidth, selected by op kind; board mapping remains explicitly unverified.
    Upstream analytic efficiency applies only inside the upstream prediction.
  * Supported inputs are exactly mm/bmm, clone/copy, relu/elementwise and
    tensor_cast rms_norm with contiguous layouts, a known dtype, a non-zero
    shape, and hardware matching the actual DeviceProfile.

[Timeouts & Locks]
  Git identity/status runs once per probe/construction with a bounded timeout;
  no git subprocess or filesystem scan occurs per prediction.  A module-local
  lock serializes temporary ``sys.path`` mutation during source import.

[Side-Effect Policy]
  Reads explicit source path and git metadata.  Does not write source files,
  install packages, mutate global configuration, or leave ``sys.path`` or
  bytecode settings changed after construction.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .theoretical_model import (
    ADAPTER_PROTOCOL_VERSION,
    HardwareSpec,
    KernelInputs,
    ModelingRequest,
    ProviderIdentity,
    ProviderPrediction,
    ProviderUnmodeledError,
    align_modeling,
)

PINNED_MSMODELING_COMMIT = "1adde03a04264fa976893fdf50495270175043e6"
PROVIDER_NAME = "msmodeling.tensor_cast.analytic"
PROBE_SCHEMA = "mindie-profiler-provider-probe/1"
ACCEPTANCE_SCHEMA = "mindie-profiler-provider-acceptance/1"
GIT_TIMEOUT_S = 10.0
GIT_MAX_BYTES = 1_048_576

_IMPORT_LOCK = threading.Lock()
_SUPPORTED_LAYOUTS = {"nd", "row_major", "contiguous", "contiguous_row_major"}
_MMA_OP_KINDS = {"matmul", "gemm", "mm", "bmm", "addmm"}
_GP_OP_KINDS = {"rmsnorm", "rms_norm", "elementwise", "relu", "reduction", "reduce", "copy", "memcpy", "clone"}
_DTYPE_NAMES = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "float32": "float32",
    "fp32": "float32",
    "int8": "int8",
    "dt_bf16": "bfloat16",
    "dt_float16": "float16",
    "dt_float32": "float32",
    "dt_float": "float32",
    "dt_int8": "int8",
}


def _normal(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _hardware_category(op_kind: str | None) -> str:
    kind = _normal(op_kind)
    if kind in _MMA_OP_KINDS:
        return "mma"
    if kind in _GP_OP_KINDS:
        return "gp"
    raise ProviderUnmodeledError(f"unsupported_op_kind:{op_kind}")


def _run_git(source_path: Path, *arguments: str) -> str:
    command = ["git", "-C", str(source_path), *arguments]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProviderUnmodeledError(f"source_git_{arguments[0]}_failed") from exc
    if completed.returncode != 0:
        raise ProviderUnmodeledError(f"source_git_{arguments[0]}_failed")
    if len(completed.stdout.encode("utf-8")) > GIT_MAX_BYTES:
        raise ProviderUnmodeledError(f"source_git_{arguments[0]}_output_too_large")
    return completed.stdout


def read_source_commit(source_path: str | Path) -> str:
    """Read the checked-out commit via one bounded git invocation."""
    commit = _run_git(Path(source_path).expanduser().resolve(), "rev-parse", "HEAD").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ProviderUnmodeledError("source_git_commit_unreadable")
    return commit


def read_source_status(source_path: str | Path) -> list[str]:
    """Read one bounded porcelain status; empty means a clean checkout."""
    output = _run_git(
        Path(source_path).expanduser().resolve(),
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
    )
    return [line for line in output.splitlines() if line.strip()]


def read_source_version(source_path: str | Path) -> str:
    pyproject = Path(source_path).expanduser().resolve() / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        version = str(data["project"]["version"]).strip()
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise ProviderUnmodeledError(f"source_version_unreadable:{type(exc).__name__}") from exc
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[.+-][0-9A-Za-z.-]+)*", version):
        raise ProviderUnmodeledError("source_version_invalid")
    return version


@dataclass(frozen=True)
class TensorCastProviderStatus:
    source_path: str
    source_present: bool
    declared_version: str | None
    commit: str | None
    expected_commit: str
    commit_matches: bool
    git_status_clean: bool | None
    dirty_count: int | None
    provider_name: str
    provider_version: str | None
    imported: bool
    verified: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROBE_SCHEMA,
            "source_path": self.source_path,
            "source_present": self.source_present,
            "declared_version": self.declared_version,
            "commit": self.commit,
            "expected_commit": self.expected_commit,
            "commit_matches": self.commit_matches,
            "git_status_clean": self.git_status_clean,
            "dirty_count": self.dirty_count,
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "imported": self.imported,
            "verified": self.verified,
            "reason": self.reason,
        }


def probe_tensor_cast_provider(
    source_path: str | Path | None = None,
    *,
    expected_commit: str = PINNED_MSMODELING_COMMIT,
) -> TensorCastProviderStatus:
    """Validate explicit source identity and cleanliness without importing it."""
    expected = expected_commit.strip().lower()
    if source_path is None:
        return TensorCastProviderStatus(
            source_path="",
            source_present=False,
            declared_version=None,
            commit=None,
            expected_commit=expected,
            commit_matches=False,
            git_status_clean=None,
            dirty_count=None,
            provider_name=PROVIDER_NAME,
            provider_version=None,
            imported=False,
            verified=False,
            reason="source_path_not_configured",
        )
    source = Path(source_path).expanduser().resolve()
    present = (source / "tensor_cast" / "performance_model" / "analytic.py").is_file()
    version: str | None = None
    commit: str | None = None
    status_lines: list[str] | None = None
    if present:
        try:
            version = read_source_version(source)
        except ProviderUnmodeledError:
            version = None
        try:
            commit = read_source_commit(source)
        except ProviderUnmodeledError:
            commit = None
        if commit is not None:
            try:
                status_lines = read_source_status(source)
            except ProviderUnmodeledError:
                status_lines = None
    matches = bool(commit and commit == expected)
    if not present:
        reason = "source_missing"
    elif version is None:
        reason = "source_version_unreadable"
    elif commit is None:
        reason = "source_git_commit_unreadable"
    elif not matches:
        reason = "source_commit_mismatch"
    elif status_lines is None:
        reason = "source_git_status_unavailable"
    elif status_lines:
        reason = "source_dirty"
    else:
        reason = "source_commit_verified_clean_probe_only"
    success = reason == "source_commit_verified_clean_probe_only"
    provider_version = f"{version}+{commit}" if success else None
    return TensorCastProviderStatus(
        source_path=str(source),
        source_present=present,
        declared_version=version,
        commit=commit,
        expected_commit=expected,
        commit_matches=matches,
        git_status_clean=(not status_lines) if status_lines is not None else None,
        dirty_count=len(status_lines) if status_lines is not None else None,
        provider_name=PROVIDER_NAME,
        provider_version=provider_version,
        imported=False,
        verified=False,
        reason=reason,
    )


class TensorCastAnalyticProvider:
    """Explicit, versioned provider that genuinely calls pinned TensorCast."""

    def __init__(
        self,
        source_path: str | Path | None = None,
        *,
        device_name: str,
        expected_commit: str = PINNED_MSMODELING_COMMIT,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        if source_path is None:
            raise ProviderUnmodeledError("source_path_not_configured")
        self._source_path = Path(source_path).expanduser().resolve()
        self._expected_commit = expected_commit.strip().lower()
        self._device_name = str(device_name)
        self._rms_norm_eps = float(rms_norm_eps)
        self._available = False
        self._executions = 0
        self._torch: Any = None
        self._op_invoke_info: Any = None
        self._model: Any = None
        self._device: Any = None
        self._source_version: str | None = None
        self._commit: str | None = None
        self._probe_status: TensorCastProviderStatus | None = None
        self.identity = ProviderIdentity(PROVIDER_NAME, "0.0.0", ADAPTER_PROTOCOL_VERSION)
        self._load()

    @property
    def source_path(self) -> str:
        return str(self._source_path)

    @property
    def source_commit(self) -> str | None:
        return self._commit

    @property
    def source_version(self) -> str | None:
        return self._source_version

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def executions(self) -> int:
        return self._executions

    def available(self) -> bool:
        return bool(self._available)

    def hardware_spec(self, dtype: str = "bfloat16", op_kind: str | None = None) -> HardwareSpec:
        """Return raw declared profile peaks; efficiency belongs to upstream only."""
        if not self._available:
            raise ProviderUnmodeledError("provider_unavailable")
        if op_kind is None:
            raise ProviderUnmodeledError("hardware_spec_op_kind_required")
        category = _hardware_category(op_kind)
        torch_dtype = self._torch_dtype(dtype)
        ops = self._device.mma_ops if category == "mma" else self._device.gp_ops
        peak = ops.get(torch_dtype)
        if peak is None:
            raise ProviderUnmodeledError(f"unsupported_profile_compute_dtype:{category}:{_normal(dtype)}")
        return HardwareSpec(
            name=self._device.name,
            peak_flops_per_s=float(peak),
            memory_bandwidth_bytes_per_s=float(self._device.memory_bandwidth_bytes_ps),
        )

    def probe_status(self) -> TensorCastProviderStatus:
        if self._probe_status is None:
            raise ProviderUnmodeledError("source_identity_incomplete")
        return self._probe_status

    def profile_summary(self, dtype: str = "bfloat16") -> dict[str, Any]:
        """Raw profile values plus upstream efficiencies for evidence."""
        if not self._available:
            raise ProviderUnmodeledError("provider_unavailable")
        torch_dtype = self._torch_dtype(dtype)
        return {
            "device_profile_name": self._device.name,
            "device_profile_mapping": "caller_explicit_unverified",
            "observed_device_name": None,
            "soc_mapping_verified": False,
            "raw_mma_ops": self._device.mma_ops.get(torch_dtype),
            "raw_gp_ops": self._device.gp_ops.get(torch_dtype),
            "raw_memory_bandwidth_bytes_ps": self._device.memory_bandwidth_bytes_ps,
            "raw_memory_size_bytes": self._device.memory_size_bytes,
            "upstream_compute_efficiency": self._device.compute_efficiency,
            "upstream_memory_efficiency": self._device.memory_efficiency,
            "local_ideal_uses_raw_peaks": True,
            "source": "pinned tensor_cast/device.py DeviceProfile",
        }

    def status(self) -> dict[str, Any]:
        return {
            "provider_name": self.identity.provider_name,
            "provider_version": self.identity.provider_version,
            "adapter_protocol_version": self.identity.adapter_protocol_version,
            "source_path": self.source_path,
            "source_version": self._source_version,
            "source_commit": self._commit,
            "device_name": self._device_name,
            "available": self.available(),
            "executions": self._executions,
            "verified": self._executions > 0,
        }

    def _load(self) -> None:
        status = probe_tensor_cast_provider(self._source_path, expected_commit=self._expected_commit)
        if status.reason != "source_commit_verified_clean_probe_only":
            raise ProviderUnmodeledError(status.reason)
        if status.declared_version is None or status.commit is None:
            raise ProviderUnmodeledError("source_identity_incomplete")
        self._probe_status = status
        self._source_version = status.declared_version
        self._commit = status.commit
        expected_dir = (self._source_path / "tensor_cast").resolve()
        with _IMPORT_LOCK:
            existing = sys.modules.get("tensor_cast")
            if existing is not None:
                existing_file = getattr(existing, "__file__", None)
                if existing_file is None or Path(existing_file).resolve().parent != expected_dir:
                    raise ProviderUnmodeledError("tensor_cast_already_imported_from_other_source")
            saved_path = list(sys.path)
            saved_bytecode = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            sys.path.insert(0, str(self._source_path))
            try:
                import torch  # noqa: PLC0415
                import tensor_cast.ops  # noqa: F401, PLC0415
                from tensor_cast.device import DeviceProfile  # noqa: PLC0415
                from tensor_cast.performance_model.analytic import AnalyticPerformanceModel  # noqa: PLC0415
                from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo  # noqa: PLC0415
            except Exception as exc:  # noqa: BLE001 - import boundary is untrusted
                raise ProviderUnmodeledError(f"provider_import_failed:{type(exc).__name__}") from exc
            finally:
                sys.path[:] = saved_path
                sys.dont_write_bytecode = saved_bytecode
        device = DeviceProfile.all_device_profiles.get(self._device_name)
        if device is None:
            raise ProviderUnmodeledError(f"unsupported_device:{self._device_name}")
        self._torch = torch
        self._op_invoke_info = OpInvokeInfo
        self._device = device
        self._model = AnalyticPerformanceModel(device)
        self.identity = ProviderIdentity(
            PROVIDER_NAME, f"{self._source_version}+{self._commit}", ADAPTER_PROTOCOL_VERSION
        )
        self._available = True

    def _torch_dtype(self, dtype: str) -> Any:
        name = _normal(dtype)
        target = _DTYPE_NAMES.get(name)
        if target is None:
            raise ProviderUnmodeledError(f"unsupported_provider_dtype:{name}")
        value = getattr(self._torch, target, None)
        if value is None:
            raise ProviderUnmodeledError(f"unsupported_provider_dtype:{name}")
        return value

    def _meta(self, shape: Sequence[int], dtype: Any) -> Any:
        return self._torch.empty(tuple(int(dim) for dim in shape), dtype=dtype, device="meta")

    def _validate_hardware(self, inputs: KernelInputs) -> None:
        hardware = inputs.hardware
        if hardware is None:
            raise ProviderUnmodeledError("missing_hardware")
        expected = self.hardware_spec(str(inputs.dtype), str(inputs.op_kind))
        if hardware.name != expected.name:
            raise ProviderUnmodeledError(
                f"hardware_device_mismatch:expected={expected.name},got={hardware.name}"
            )
        if not math.isclose(
            float(hardware.peak_flops_per_s), expected.peak_flops_per_s, rel_tol=1e-9
        ):
            raise ProviderUnmodeledError("hardware_peak_flops_mismatch")
        if not math.isclose(
            float(hardware.memory_bandwidth_bytes_per_s),
            expected.memory_bandwidth_bytes_per_s,
            rel_tol=1e-9,
        ):
            raise ProviderUnmodeledError("hardware_memory_bandwidth_mismatch")

    def _build_invocation(self, inputs: KernelInputs) -> tuple[Any, tuple[Any, ...], Any, str]:
        missing = inputs.missing_metadata_inputs()
        if missing:
            raise ProviderUnmodeledError("missing_inputs:" + ",".join(missing))
        layout = _normal(inputs.layout)
        if layout not in _SUPPORTED_LAYOUTS:
            raise ProviderUnmodeledError(f"unsupported_layout:{inputs.layout}")
        self._validate_hardware(inputs)
        dims = tuple(int(dim) for dim in (inputs.shape or ()))
        if not dims or any(dim <= 0 for dim in dims):
            raise ProviderUnmodeledError("unsupported_zero_or_empty_shape")
        dtype = self._torch_dtype(str(inputs.dtype))
        kind = _normal(inputs.op_kind)
        if kind == "addmm":
            if len(dims) != 3:
                raise ProviderUnmodeledError("invalid_addmm_shape")
            m, k, n = dims
            args = (
                self._meta((n,), dtype),
                self._meta((m, k), dtype),
                self._meta((n, k), dtype).t(),
            )
            func = self._torch.ops.aten.addmm.default
        elif kind in _MMA_OP_KINDS:
            if len(dims) == 3:
                m, k, n = dims
                args = (self._meta((m, k), dtype), self._meta((k, n), dtype))
                func = self._torch.ops.aten.mm.default
            elif len(dims) == 4:
                batch, m, k, n = dims
                args = (
                    self._meta((batch, m, k), dtype),
                    self._meta((batch, k, n), dtype),
                )
                func = self._torch.ops.aten.bmm.default
            else:
                raise ProviderUnmodeledError("invalid_matmul_shape")
        elif kind in {"copy", "memcpy", "clone"}:
            args = (self._meta(dims, dtype),)
            func = self._torch.ops.aten.clone.default
        elif kind in {"elementwise", "relu"}:
            args = (self._meta(dims, dtype),)
            func = self._torch.ops.aten.relu.default
        elif kind in {"rmsnorm", "rms_norm"}:
            args = (self._meta(dims, dtype), self._meta((dims[-1],), dtype), self._rms_norm_eps)
            func = self._torch.ops.tensor_cast.rms_norm.default
        else:
            raise ProviderUnmodeledError(f"unsupported_op_kind:{inputs.op_kind}")
        out = func(*args)
        return func, args, out, str(func)

    def predict(self, request: ModelingRequest | KernelInputs) -> ProviderPrediction:
        if not self._available:
            raise ProviderUnmodeledError("provider_unavailable")
        if isinstance(request, KernelInputs):
            request = ModelingRequest(inputs=request)
        func, args, out, label = self._build_invocation(request.inputs)
        info = self._op_invoke_info(func, args, {}, out)
        result = self._model.process_op(info)
        predicted_s = float(result.execution_time_s)
        if not math.isfinite(predicted_s) or predicted_s <= 0:
            raise ProviderUnmodeledError("upstream_prediction_nonpositive")
        self._executions += 1
        stats = []
        for key, value in sorted(result.statistics.items(), key=lambda item: str(item[0])):
            stat_name = str(getattr(key, "value", key))
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                stats.append(f"stat:{stat_name}={float(value):.12g}")
        category = _hardware_category(request.inputs.op_kind)
        raw_ops = self._device.mma_ops if category == "mma" else self._device.gp_ops
        selected_peak = raw_ops.get(self._torch_dtype(str(request.inputs.dtype)))
        evidence = [
            f"msmodeling_source:{self._source_path}@{self._commit}",
            f"msmodeling_version:{self._source_version}",
            f"device_profile:{self._device.name}",
            "device_profile_mapping:caller_explicit_unverified",
            f"hardware_category:{category}",
            f"raw_peak_flops_for_op:{selected_peak}",
            f"raw_memory_bandwidth_bytes_ps:{self._device.memory_bandwidth_bytes_ps}",
            f"compute_efficiency:{self._device.compute_efficiency}",
            f"memory_efficiency:{self._device.memory_efficiency}",
            f"torch_version:{self._torch.__version__}",
            f"op_schema:{func._schema}",
            f"op_cache_key:{info.cache_key}",
            *stats[:12],
        ]
        kind = _normal(request.inputs.op_kind)
        if kind in {"relu", "elementwise", "copy", "memcpy", "clone"}:
            evidence.append("op_properties:upstream_default_memory_bandwidth_bound")
        eps = f"rms_norm_eps={self._rms_norm_eps}; " if kind in {"rmsnorm", "rms_norm"} else ""
        default_note = (
            "relu_or_copy_default_properties=upstream_memory_bandwidth_bound; "
            if kind in {"relu", "elementwise", "copy", "memcpy", "clone"}
            else ""
        )
        assumptions = (
            f"device={self._device.name}; device_profile_mapping=caller_explicit_unverified; "
            f"op={label}; layout=contiguous; dtype={_normal(request.inputs.dtype)}; {eps}"
            "local_ideal_uses_raw_peaks=true; upstream_applies_compute_efficiency="
            f"{self._device.compute_efficiency}; upstream_applies_memory_efficiency="
            f"{self._device.memory_efficiency}; {default_note}"
            "upstream=AnalyticPerformanceModel; static_cost_included=true; "
            "measured_device_ms_not_used_by_provider"
        )
        return ProviderPrediction(
            predicted_ms=predicted_s * 1000.0,
            model_assumptions=assumptions,
            input_signature=request.input_signature,
            evidence=tuple(evidence),
        )


def _parse_shape(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise ProviderUnmodeledError("invalid_shape_argument") from exc
    if not values:
        raise ProviderUnmodeledError("invalid_shape_argument")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mindie_profiler.tensor_cast_provider",
        description="Run the pinned TensorCast analytic provider acceptance probe.",
    )
    parser.add_argument("--source", required=True, help="explicit pinned msmodeling checkout")
    parser.add_argument("--device", required=True, help="explicit upstream DeviceProfile name")
    parser.add_argument("--shape", default="8,16,4")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layout", default="row_major")
    parser.add_argument("--op", default="matmul")
    parser.add_argument("--measured-ms", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    probe: TensorCastProviderStatus | None = None
    try:
        provider = TensorCastAnalyticProvider(args.source, device_name=args.device)
        probe = provider.probe_status()
        inputs = KernelInputs(
            shape=_parse_shape(args.shape),
            dtype=args.dtype,
            layout=args.layout,
            hardware=provider.hardware_spec(args.dtype, args.op),
            op_kind=args.op,
        )
        alignment = align_modeling(
            ModelingRequest(inputs=inputs, measured_device_ms=args.measured_ms),
            provider,
        )
        payload = {
            "schema_version": ACCEPTANCE_SCHEMA,
            "probe": probe.as_dict(),
            "provider": provider.status(),
            "inputs": inputs.canonical_payload(),
            "measured_device_ms_supplied": args.measured_ms is not None,
            "alignment": alignment.to_c7_alignment(),
            "upstream_executed": alignment.status == "modeled",
        }
        code = 0 if alignment.status == "modeled" else 1
    except ProviderUnmodeledError as exc:
        if probe is None:
            probe = probe_tensor_cast_provider(args.source)
        payload = {
            "schema_version": ACCEPTANCE_SCHEMA,
            "probe": probe.as_dict(),
            "inputs": {
                "shape": args.shape,
                "dtype": args.dtype,
                "layout": args.layout,
                "op_kind": args.op,
                "device": args.device,
            },
            "alignment": {
                "status": "unmodeled",
                "ideal_hardware_lower_bound_ms": None,
                "model_predicted_ms": None,
                "measured_device_ms": args.measured_ms,
                "unmodeled_reason": exc.reason,
            },
            "upstream_executed": False,
        }
        code = 1
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = Path(args.output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    print(text, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PINNED_MSMODELING_COMMIT",
    "PROVIDER_NAME",
    "PROBE_SCHEMA",
    "ACCEPTANCE_SCHEMA",
    "TensorCastProviderStatus",
    "TensorCastAnalyticProvider",
    "read_source_commit",
    "read_source_status",
    "read_source_version",
    "probe_tensor_cast_provider",
    "main",
]
