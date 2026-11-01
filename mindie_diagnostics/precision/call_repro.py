"""Call freezer, exact-view replayer, jitter and explicit reference evaluators.

[Role]
  Primitive 3 (``call-repro``): snapshot the shared pre-call storage graph,
  RNG/execution state and callable metadata; execute once; save a capsule;
  then replay from the captured pre-call state for every repeat trial and run
  optional jitter, explicit-reference and explicit fake-quant evaluators.

[Invariants]
  * A capsule stores shared ``UntypedStorage`` bytes once plus exact tensor
    views; replay reconstructs aliases instead of independent contiguous copies.
  * Every repeat trial re-enters from the pre-call snapshot and restores RNG
    state, so in-place input/parameter mutations never accumulate.
  * ``reference_check`` without an explicit reference function reports
    ``unsupported`` unless ``mode='reference_check'`` requests the automatic
    FP32 enclosure (intermediate half-precision casts stay FP32).
  * ``fakequant_check`` without a contract reports ``unsupported``; original
    weight loss is reported only when original weights are explicitly supplied.
  * Det-lock is reported with a warning when a distributed backend was already
    initialized; environment changes are never presented as retroactive.

[Timeouts & Locks]
  Caller-controlled synchronous execution.  No sleeps, subprocess, network or
  locks.  Replay timings are measured, not promised; device synchronization is
  performed only around timed trials on available backends.

[Side-Effect Policy]
  Capture executes the supplied callable and may write ``save_path`` via
  ``torch.save``.  Replay mutates only reconstructed/live call tensors and
  module buffers.  ``det_lock`` may set documented HCCL environment variables.
"""
from __future__ import annotations
import contextlib
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from .storage_graph import (
    _live_module_map,
    _require_torch,
    capture_state_snapshot,
    flatten_tree_with_paths,
    map_tree,
    path_to_string,
    reconstruct_from_snapshot,
    restore_state_snapshot,
    snapshot_tensor_map,
)
from .lockstep_diff import _tensor_metrics
from .fp32_reference import (
    _UnsupportedAutoReferenceError,
    _auto_fp32_reference,
    _fakequant_verdict,
    _replay_verdict,
)
_EVAL_UNSUPPORTED = "unsupported"
@dataclass
class FakeQuantContract:
    """Explicit fake-quant decomposition; never inferred from a model name."""
    bits: int
    symmetric: bool
    scale_axis: Any
    group_size: int | None = None
    zero_point: Any = None
    round_mode: str = "nearest"
    saturate: bool = True
    accum_dtype: str = "float32"
    def __post_init__(self):
        if not isinstance(self.bits, int) or self.bits < 2 or self.bits > 32:
            raise ValueError("bits must be an integer in [2, 32]")
        if self.round_mode not in {"nearest", "floor", "ceil", "trunc"}:
            raise ValueError("round_mode must be nearest/floor/ceil/trunc")
        if self.group_size is not None and int(self.group_size) <= 0:
            raise ValueError("group_size must be positive")
@dataclass
class CallCapsule:
    """Self-contained captured invocation, serializable with ``torch.save``."""
    pre: Any
    post: Any
    rng_state: dict[str, Any]
    execution_state: dict[str, Any]
    callable_metadata: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    format_version: int = 1
from .call_state import (
    _DET_LOCK_ENV, _apply_det_lock, _callable_metadata, _capture_execution_state,
    _capture_generators, _capture_rng, _distributed_initialized,
    _distributed_topology, _restore_generators, _restore_rng,
    normalized_collective_trace, trace_collectives,
)
from .jitter import JitterAccumulator, _tensor_bytes
def capture_call(fn_or_module, *args, save_path=None, metadata=None, **kwargs):
    """Capture one call before/after the shared storage graph is mutated."""
    torch = _require_torch()
    if not callable(fn_or_module):
        raise ValueError("fn_or_module must be callable")
    module = fn_or_module if isinstance(fn_or_module, torch.nn.Module) else None
    pre = capture_state_snapshot(args=args, kwargs=kwargs, module=module)
    rng_state = _capture_rng()
    generators = _capture_generators(args, kwargs, module)
    if generators:
        rng_state["generators"] = generators
    execution_state = _capture_execution_state(fn_or_module)
    callable_metadata = _callable_metadata(fn_or_module)
    error = None
    result = None
    post = None
    collective_trace = []
    try:
        with trace_collectives(torch, collective_trace):
            result = fn_or_module(*args, **kwargs)
    except BaseException as exc:
        error = exc
        raise
    finally:
        execution_state["collective_trace"] = normalized_collective_trace(
            collective_trace)
        post = capture_state_snapshot(args=args, kwargs=kwargs, outputs=result, module=module)
        capsule = CallCapsule(
            pre=pre,
            post=post,
            rng_state=rng_state,
            execution_state=execution_state,
            callable_metadata=callable_metadata,
            metadata=dict(metadata or {}),
        )
        if save_path is not None:
            torch.save(capsule, save_path)
    if error is not None:  # pragma: no cover - re-raised above
        raise error
    return capsule
def _load_capsule(capsule_or_path):
    torch = _require_torch()
    if isinstance(capsule_or_path, CallCapsule):
        return capsule_or_path
    if isinstance(capsule_or_path, dict):
        data = capsule_or_path
    else:
        path = Path(capsule_or_path)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - older torch
            data = torch.load(path, map_location="cpu")
    if isinstance(data, CallCapsule):
        return data
    if isinstance(data, dict) and "pre" in data and "post" in data:
        return CallCapsule(**data)
    raise ValueError("capsule must be a CallCapsule or a saved CallCapsule")
def capsule_summary(capsule_or_path):
    """Host-side, torch-only capsule inspection used by the CLI."""
    capsule = _load_capsule(capsule_or_path)
    return {
        "format_version": capsule.format_version,
        "callable": capsule.callable_metadata,
        "metadata": capsule.metadata,
        "execution_state": capsule.execution_state,
        "pre": _snapshot_summary(capsule.pre),
        "post": _snapshot_summary(capsule.post),
    }
def _snapshot_summary(snapshot):
    return {
        "roots": sorted(snapshot.roots),
        "storages": len(snapshot.storages),
        "bytes": sum(node.nbytes for node in snapshot.storages.values()),
        "views": len(snapshot.views),
        "tensor_paths": [view.tree_path for view in snapshot.views],
    }
def _resolve_callable(capsule, fn_or_module):
    if fn_or_module is not None:
        return fn_or_module
    path = capsule.callable_metadata.get("import_path")
    if not path or ":" not in path:
        raise ValueError("callable reconstruction requires fn_or_module=...")
    module_name, qualname = path.split(":", 1)
    try:
        target = __import__(module_name, fromlist=[qualname.split(".")[0]])
        for part in qualname.split("."):
            target = getattr(target, part)
    except (ImportError, AttributeError) as exc:
        raise ValueError("callable reconstruction requires fn_or_module=...") from exc
    if not callable(target):
        raise ValueError("captured import path is not callable")
    return target
def _module_storage_overrides(snapshot, module):
    live = _live_module_map(module)
    overrides = {}
    for view in snapshot.views:
        tensor = live.get(view.tree_path)
        if tensor is None:
            continue
        node = snapshot.storages.get(view.storage_id)
        if node is None:
            continue
        storage = tensor.untyped_storage()
        if int(storage.nbytes()) == node.nbytes:
            overrides[view.storage_id] = storage
    return overrides
def _module_tensor_overrides(capsule, module):
    """Map captured tensor identities to live parameters/buffers by path."""
    template = capsule.pre.roots.get("module") or {}
    live = _live_module_map(module)
    by_id = {}
    for table, prefix in (("parameters", "module.parameters"),
                          ("buffers", "module.buffers")):
        for name, ref in template.get(table, {}).items():
            tensor = live.get(f"{prefix}.{name}")
            tensor_id = getattr(ref, "tensor_id", None)
            if tensor is not None and tensor_id is not None:
                by_id[tensor_id] = tensor
    return {view.tree_path: by_id[view.tensor_id]
            for view in capsule.pre.views if view.tensor_id in by_id}
def _prepare_state(capsule, module):
    torch = _require_torch()
    outer_inference = (not capsule.execution_state.get("inference_mode")
                       and torch.is_inference_mode_enabled())
    with (torch.inference_mode(False) if outer_inference else contextlib.nullcontext()):
        overrides = (_module_storage_overrides(capsule.pre, module)
                     if module is not None else None)
        tensor_overrides = (_module_tensor_overrides(capsule, module)
                            if module is not None else None)
        state = reconstruct_from_snapshot(
            capsule.pre, storage_overrides=overrides,
            tensor_overrides=tensor_overrides)
        args = state.get("args", ())
        kwargs = state.get("kwargs", {})
        if module is not None and "module" in capsule.pre.roots:
            restore_state_snapshot(capsule.pre, module=module)
        restore_state_snapshot(
            capsule.pre,
            args=args if "args" in capsule.pre.roots else None,
            kwargs=kwargs if "kwargs" in capsule.pre.roots else None,
        )
        if outer_inference:
            args = _clone_inference_values(args)
            kwargs = _clone_inference_values(kwargs)
    return args, kwargs
def _sync_devices():
    torch = _require_torch()
    for name in ("npu", "cuda"):
        backend = getattr(torch, name, None)
        if backend is None:
            continue
        try:
            if backend.is_available():
                backend.synchronize()
        except (RuntimeError, OSError):
            continue
def _clone_inference_values(value):
    torch = _require_torch()
    checker = getattr(torch, "is_inference", None)
    if checker is None:
        return value
    memo = {}
    def convert(leaf):
        if isinstance(leaf, torch.Tensor):
            try:
                if checker(leaf):
                    clone = memo.get(id(leaf))
                    if clone is None:
                        clone = leaf.clone()
                        memo[id(leaf)] = clone
                    return clone
            except (RuntimeError, TypeError):
                pass
        return leaf
    return map_tree(value, convert)
def _run_with_state(fn, args, kwargs, execution_state, module=None):
    torch = _require_torch()
    previous_training = {}
    if module is not None:
        previous_training = {
            name: bool(submodule.training)
            for name, submodule in module.named_modules(remove_duplicate=False)}
    try:
        with contextlib.ExitStack() as stack:
            captured_inference = bool(execution_state.get("inference_mode"))
            if captured_inference:
                stack.enter_context(torch.inference_mode())
            elif torch.is_inference_mode_enabled():
                stack.enter_context(torch.inference_mode(False))
            # inference_mode(False) resets grad mode, so re-apply it afterwards.
            stack.enter_context(torch.set_grad_enabled(
                bool(execution_state.get("grad_enabled", True))))
            if execution_state.get("autocast"):
                device_type = execution_state.get("autocast_device_type") or "cpu"
                dtype_name = execution_state.get("autocast_dtype") or ""
                dtype = getattr(torch, dtype_name.rsplit(".", 1)[-1], None)
                if dtype is None:
                    stack.enter_context(torch.autocast(device_type=device_type, enabled=True))
                else:
                    stack.enter_context(torch.autocast(
                        device_type=device_type, dtype=dtype, enabled=True))
            else:
                # Disable inherited outer autocast for every supported device
                # type; unavailable extensions (e.g. npu without torch_npu) are
                # simply skipped.
                for device_type in ("cpu", "cuda", "npu"):
                    try:
                        stack.enter_context(
                            torch.autocast(device_type=device_type, enabled=False))
                    except (RuntimeError, TypeError):
                        continue
            if module is not None:
                targets = execution_state.get("submodule_training")
                if targets:
                    for name, submodule in module.named_modules(remove_duplicate=False):
                        if name in targets:
                            submodule.training = bool(targets[name])
                elif execution_state.get("training") is not None:
                    module.train(bool(execution_state["training"]))
            return fn(*args, **kwargs)
    finally:
        if module is not None:
            for name, submodule in module.named_modules(remove_duplicate=False):
                if name in previous_training:
                    submodule.training = previous_training[name]
def _trial_output_values(output):
    torch = _require_torch()
    result = {}
    for path, leaf in flatten_tree_with_paths(output, dedupe_containers=True):
        key = path_to_string(("outputs",) + tuple(path))
        if isinstance(leaf, torch.Tensor):
            result[key] = leaf.detach().to("cpu").clone()
        elif leaf is None or isinstance(leaf, (bool, int, float, complex, str, bytes)):
            result[key] = leaf
    return result
def _trial_output_tensors(output):
    return {path: value for path, value in _trial_output_values(output).items()
            if _require_torch().is_tensor(value)}
class _TrialWindow(list):
    """At most two retained trial dicts plus the streaming accumulator."""
def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(fraction * len(ordered))) - 1))
    return float(ordered[index])
def _jitter_repeat(trials):
    accumulator = getattr(trials, "accumulator", None)
    if accumulator is None:
        accumulator = JitterAccumulator()
        for trial in trials:
            accumulator.update(trial)
    return accumulator.report()
def _scalar_equal(left, right):
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    try:
        return bool(left == right)
    except Exception:
        return left is right
def _reference_check(capsule, module, reference_fn, actual_post,
                     reference_atol, reference_rtol):
    if reference_fn is None:
        return {"status": _EVAL_UNSUPPORTED, "reason": "explicit_reference_fn_required"}
    torch = _require_torch()
    _restore_rng(capsule.rng_state)
    try:
        # Restore the original module even when the reference is a plain
        # function closing over it; otherwise it would see the last trial's
        # in-place mutations.
        args, kwargs = _prepare_state(capsule, module)
        _restore_generators(capsule.rng_state, args, kwargs, module)
        reference_output = _run_with_state(
            reference_fn, args, kwargs, capsule.execution_state, module)
        ref_post = capture_state_snapshot(
            args=args, kwargs=kwargs, outputs=reference_output, module=module)
    except _UnsupportedAutoReferenceError as exc:
        return {"status": _EVAL_UNSUPPORTED, "reason": str(exc)}
    except Exception as exc:
        return {"status": "error", "reason": type(exc).__name__}
    def scalar_leaves(snapshot):
        rebuilt = reconstruct_from_snapshot(snapshot)
        leaves = {}
        for root, value in rebuilt.items():
            for path, leaf in flatten_tree_with_paths(value):
                if not isinstance(leaf, torch.Tensor):
                    leaves[path_to_string((root,) + tuple(path))] = leaf
        return leaves
    actual_leaves = scalar_leaves(actual_post)
    ref_leaves = scalar_leaves(ref_post)
    if set(actual_leaves) != set(ref_leaves):
        return {"status": "fail", "reason": "state_structure_mismatch"}
    scalar_mismatches = [
        path for path in actual_leaves
        if not _scalar_equal(ref_leaves[path], actual_leaves[path])
    ]
    if scalar_mismatches:
        return {"status": "fail", "reason": "scalar_state_mismatch",
                "scalar_mismatch_paths": sorted(scalar_mismatches)[:16]}
    ref = snapshot_tensor_map(ref_post, device="cpu")
    act = snapshot_tensor_map(actual_post, device="cpu")
    if set(ref) != set(act):
        return {"status": "fail", "reason": "state_structure_mismatch"}
    mismatches = 0
    max_abs = 0.0
    max_rel = 0.0
    per_path = []
    for path in sorted(ref):
        metric = _tensor_metrics(path, ref[path], act[path], [], None)
        if metric.get("status") != "ok":
            return {"status": "fail", "reason": f"tensor_metric_{metric.get('status')}",
                    "per_path": per_path}
        left, right = act[path], ref[path]
        if not left.dtype.is_floating_point and not left.is_complex():
            count = int((left != right).sum().item())
        elif left.is_complex():
            close = torch.isclose(
                left.to(torch.complex128), right.to(torch.complex128),
                atol=float(reference_atol), rtol=float(reference_rtol), equal_nan=True)
            count = int((~close).sum().item())
        else:
            close = torch.isclose(
                left.to(dtype=torch.float64), right.to(dtype=torch.float64),
                atol=float(reference_atol), rtol=float(reference_rtol), equal_nan=True)
            count = int((~close).sum().item())
        mismatches += count
        max_abs = max(max_abs, float(metric.get("max_abs_err", 0.0)))
        max_rel = max(max_rel, float(metric.get("rel_l2_err", 0.0)))
        per_path.append({"tree_path": path, "mismatch_count": count, **metric})
    return {
        "status": "pass" if mismatches == 0 else "fail",
        "atol": float(reference_atol),
        "rtol": float(reference_rtol),
        "mismatch_count": mismatches,
        "max_abs_err": max_abs,
        "rel_l2_err": max_rel,
        "per_path": per_path,
    }
def _round_values(values, mode):
    if mode == "nearest":
        return values.round()
    if mode == "floor":
        return values.floor()
    if mode == "ceil":
        return values.ceil()
    return values.trunc()
def fake_quantize(tensor, contract: FakeQuantContract):
    """Apply a declared scale-axis/grouping fake-quant contract."""
    torch = _require_torch()
    source = tensor.detach().to(torch.float32)
    original_shape = tuple(source.shape)
    if contract.group_size is not None and source.numel():
        if source.dim() == 0:
            raise ValueError("group_size requires a tensor with a last dimension")
        group = int(contract.group_size)
        if source.shape[-1] % group:
            raise ValueError("group_size must divide the last dimension")
        flat = source.reshape(-1, group)
        return fake_quantize(flat, FakeQuantContract(
            bits=contract.bits, symmetric=contract.symmetric,
            scale_axis=0, group_size=None, zero_point=contract.zero_point,
            round_mode=contract.round_mode, saturate=contract.saturate,
            accum_dtype=contract.accum_dtype)).reshape(original_shape)
    axis = contract.scale_axis
    if source.dim() == 0 or axis is None:
        reduce_dims = tuple(range(source.dim()))
    elif isinstance(axis, int):
        reduce_dims = tuple(index for index in range(source.dim()) if index != axis % source.dim())
    else:
        keep = {int(index) % max(source.dim(), 1) for index in axis}
        reduce_dims = tuple(index for index in range(source.dim()) if index not in keep)
    if contract.symmetric:
        qmax = float(2 ** (contract.bits - 1) - 1)
        qmin = -qmax
        absmax = source.abs().amax(dim=reduce_dims, keepdim=True) if reduce_dims else source.abs()
        scale = (absmax / max(qmax, 1.0)).clamp_min(torch.finfo(torch.float32).tiny)
        zero = torch.zeros_like(scale)
    else:
        qmax = float(2 ** contract.bits - 1)
        qmin = 0.0
        minimum = source.amin(dim=reduce_dims, keepdim=True) if reduce_dims else source
        maximum = source.amax(dim=reduce_dims, keepdim=True) if reduce_dims else source
        constant = maximum == minimum
        qrange = max(qmax - qmin, 1.0)
        scale = torch.where(
            constant,
            torch.where(maximum.abs() > 0, maximum.abs() / qrange,
                        torch.ones_like(maximum)),
            (maximum - minimum) / qrange,
        ).clamp_min(torch.finfo(torch.float32).tiny)
        if contract.zero_point is None:
            zero = torch.round(qmin - minimum / scale)
        else:
            zero = torch.as_tensor(contract.zero_point, dtype=torch.float32, device=source.device)
            zero = torch.broadcast_to(zero, scale.shape)
    quantized = _round_values(source / scale + zero, contract.round_mode)
    if contract.saturate:
        quantized = quantized.clamp(qmin, qmax)
    accum = getattr(torch, contract.accum_dtype, torch.float32)
    return ((quantized - zero) * scale).to(accum).to(torch.float32)
def _find_kernel_weight(module):
    if not isinstance(module, _require_torch().nn.Module):
        return None
    for named in (module.named_parameters(), module.named_buffers()):
        for name, tensor in named:
            if name.endswith("weight") and tensor.is_floating_point():
                return name, tensor
    return None
def _owner_module(module, tensor_name):
    for _, submodule in module.named_modules():
        if (tensor_name in submodule._parameters
                or tensor_name in submodule._buffers):
            return submodule
    return None
def _metric_pair(name, left, right):
    torch = _require_torch()
    metric = _tensor_metrics(
        name, left.detach().to(device="cpu", dtype=torch.float32),
        right.detach().to(device="cpu", dtype=torch.float32), [], None)
    return {"max_abs_err": metric.get("max_abs_err", 0.0),
            "rel_l2_err": metric.get("rel_l2_err", 0.0),
            "sample_count": metric.get("sample_count", 0)}
def _independent_linear_output(owner, args, kwargs, kernel, fake, actual_output):
    torch = _require_torch()
    out_tensors = list(_trial_output_tensors(actual_output).values())
    if len(out_tensors) != 1 or kernel.dim() != 2:
        return None
    target_tensor = out_tensors[0]
    call_args = [arg for arg in list(args) + list((kwargs or {}).values())
                 if isinstance(arg, torch.Tensor)]
    if not call_args:
        return None
    source = residual = None
    if len(call_args) == 1:
        source = call_args[0]
    elif len(call_args) == 2 and call_args[0].dim() and call_args[1].dim():
        first, second = call_args
        if (int(second.shape[-1]) == int(kernel.shape[1])
                and (*second.shape[:-1], int(kernel.shape[0])) == tuple(target_tensor.shape)
                and tuple(first.shape) == tuple(target_tensor.shape)
                and int(kernel.shape[0]) != int(kernel.shape[1])):
            source, residual = second, first
        elif (int(first.shape[-1]) in (int(kernel.shape[0]), int(kernel.shape[1]))
              and tuple(second.shape) == tuple(target_tensor.shape)):
            source, residual = first, second
    if source is None or source.dim() == 0:
        return None
    if source.is_complex() or bool(source.is_quantized):
        return None
    if int(source.shape[-1]) not in (int(kernel.shape[0]), int(kernel.shape[1])):
        return None
    x_f32 = source.detach().to(torch.float32)
    fake_f32 = fake.detach().to(device=x_f32.device, dtype=torch.float32)
    bias = getattr(owner, "bias", None)
    bias_f32 = None
    if (isinstance(bias, torch.Tensor) and bias.dim() == 1
            and int(bias.shape[0]) == int(kernel.shape[0])):
        bias_f32 = bias.detach().to(device=x_f32.device, dtype=torch.float32)
    try:
        with torch.no_grad():
            if int(source.shape[-1]) == int(kernel.shape[1]):
                output = torch.nn.functional.linear(x_f32, fake_f32, bias_f32)
            else:
                output = x_f32 @ fake_f32
                if bias_f32 is not None:
                    output = output + bias_f32
            if residual is not None:
                output = output + residual.detach().to(
                    device=x_f32.device, dtype=torch.float32)
            return map_tree(
                actual_output,
                lambda leaf: output if isinstance(leaf, torch.Tensor) else leaf)
    except RuntimeError:
        return None
def _fakequant_check(capsule, fn_or_module, contract, original_float_weight,
                     actual_output):
    if contract is None:
        return {"status": _EVAL_UNSUPPORTED, "reason": "explicit_fakequant_contract_required"}
    if not isinstance(contract, FakeQuantContract):
        raise ValueError("fakequant_contract must be a FakeQuantContract")
    torch = _require_torch()
    module = fn_or_module if isinstance(fn_or_module, torch.nn.Module) else None
    if module is None:
        return {"status": _EVAL_UNSUPPORTED, "reason": "module_with_weight_required",
                "contract": asdict(contract)}
    _restore_rng(capsule.rng_state)
    args, kwargs = _prepare_state(capsule, module)
    _restore_generators(capsule.rng_state, args, kwargs, module)
    found = _find_kernel_weight(module)
    if found is None:
        return {"status": _EVAL_UNSUPPORTED, "reason": "module_weight_required",
                "contract": asdict(contract)}
    name, kernel = found
    owner = _owner_module(module, name)
    if owner is None:
        return {"status": _EVAL_UNSUPPORTED, "reason": "module_weight_required",
                "contract": asdict(contract)}
    source_weight = kernel
    if original_float_weight is not None:
        source_weight = (original_float_weight.detach().to(device=kernel.device, dtype=torch.float32)
                         if hasattr(original_float_weight, "detach")
                         else torch.as_tensor(original_float_weight, device=kernel.device, dtype=torch.float32))
    fake = fake_quantize(source_weight, contract)
    replacement = fake.to(device=kernel.device, dtype=kernel.dtype)
    independent = _independent_linear_output(
        owner, args, kwargs, kernel, fake, actual_output)
    if independent is not None:
        fakequant_output = independent
    elif original_float_weight is not None:
        return {"status": _EVAL_UNSUPPORTED, "reason": "independent_reference_required",
                "contract": asdict(contract)}
    else:
        _restore_rng(capsule.rng_state)
        saved_parameter = owner._parameters.get(name)
        saved_buffer = owner._buffers.get(name)
        try:
            if name in owner._parameters:
                owner._parameters[name] = torch.nn.Parameter(
                    replacement, requires_grad=bool(kernel.requires_grad))
            else:
                owner._buffers[name] = replacement
            with torch.no_grad():
                fakequant_output = _run_with_state(
                    module, args, kwargs, capsule.execution_state, module)
        finally:
            if name in owner._parameters:
                owner._parameters[name] = saved_parameter
            elif name in owner._buffers:
                owner._buffers[name] = saved_buffer
    reference = _trial_output_tensors(actual_output)
    fakequant = _trial_output_tensors(fakequant_output)
    if set(reference) != set(fakequant):
        output_metric = {"status": "output_structure_mismatch"}
    else:
        per_path = [_metric_pair(path, reference[path], fakequant[path])
                    for path in sorted(reference)]
        output_metric = {
            "status": "ok",
            "max_abs_err": max((item["max_abs_err"] for item in per_path), default=0.0),
            "rel_l2_err": max((item["rel_l2_err"] for item in per_path), default=0.0),
            "sample_count": sum(item["sample_count"] for item in per_path),
        }
    result = {
        "status": "ok",
        "contract": asdict(contract),
        "fakequant_vs_kernel": output_metric,
        "weight_fakequant_err": _metric_pair("weight", kernel, fake),
        "kernel_sample_count": int(kernel.numel()),
    }
    if original_float_weight is None:
        result["inherent_loss_status"] = "original_weights_unavailable"
    else:
        if hasattr(original_float_weight, "detach"):
            original = original_float_weight.detach().to(torch.float32)
        else:
            original = torch.as_tensor(original_float_weight, dtype=torch.float32)
        result["inherent_loss_status"] = "computed_from_explicit_original"
        result["float_vs_fakequant"] = _metric_pair(
            "weight", original, fake_quantize(original, contract))
    return result
def _distributed_mismatch_result(capsule, det_report, repeat, *, expected_rank=None,
                                 expected_world_size=None, observed=None,
                                 expected_trace=None, observed_trace=None):
    result = {
        "status": "error", "reason": "distributed_topology_mismatch",
        "expected_distributed_rank": expected_rank,
        "expected_distributed_world_size": expected_world_size,
        "observed_distributed_rank": observed[0] if observed else None,
        "observed_distributed_world_size": observed[1] if observed else None,
        "repeat": int(repeat),
        "outputs": reconstruct_from_snapshot(capsule.post).get("outputs"),
        "outputs_are_captured": True, "jitter_repeat": _jitter_repeat([]),
        "reference_check": {"status": _EVAL_UNSUPPORTED,
                            "reason": "distributed_topology_mismatch"},
        "fakequant_check": {"status": _EVAL_UNSUPPORTED,
                            "reason": "distributed_topology_mismatch"},
        "verdict": "error",
        "det_lock": det_report, "det_lock_warning": det_report.get("warning"),
    }
    if expected_trace is not None:
        result["expected_collective_trace"] = [list(item) for item in expected_trace]
        result["observed_collective_trace"] = [
            list(item) for item in (observed_trace or ())]
    return result


def replay_call(capsule_or_path, fn_or_module=None, *, repeat=1, det_lock=False,
                reference_fn=None, reference_atol=1e-3, reference_rtol=1e-3,
                fakequant_contract=None, original_float_weight=None,
                mode: str | None = None, quant_bits: int | None = None):
    """Replay a captured call from its pre-call storage state."""
    torch = _require_torch()
    if int(repeat) < 1:
        raise ValueError("repeat must be >= 1")
    if quant_bits is not None and fakequant_contract is None:
        fakequant_contract = FakeQuantContract(
            bits=int(quant_bits), symmetric=True, scale_axis=0)
    reference_requested = bool(reference_fn is not None or mode == "reference_check")
    fakequant_requested = bool(fakequant_contract is not None)
    auto_reference = (mode == "reference_check" and reference_fn is None)
    det_report = _apply_det_lock(torch, det_lock)
    capsule = _load_capsule(capsule_or_path)
    if capsule.execution_state.get("distributed_initialized"):
        observed = _distributed_topology(torch)
        expected_rank = capsule.execution_state.get("distributed_rank")
        expected_world_size = capsule.execution_state.get("distributed_world_size")
        mismatch = (
            observed is None
            or (expected_rank is not None and observed[0] != expected_rank)
            or (expected_world_size is not None
                and observed[1] != expected_world_size))
        if mismatch:
            return _distributed_mismatch_result(
                capsule, det_report, repeat,
                expected_rank=expected_rank,
                expected_world_size=expected_world_size,
                observed=observed)
    fn = _resolve_callable(capsule, fn_or_module)
    module = fn if isinstance(fn, torch.nn.Module) else None
    if "module" in capsule.pre.roots and module is None:
        raise ValueError("captured module state requires fn_or_module=<matching nn.Module>")
    previous_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(bool(capsule.execution_state.get("grad_enabled", True)))
    expected_trace = normalized_collective_trace(
        capsule.execution_state.get("collective_trace"))
    trials = _TrialWindow()
    accumulator = JitterAccumulator()
    timings = []
    actual_post = None
    actual_output = None
    try:
        for trial in range(int(repeat)):
            _restore_rng(capsule.rng_state)
            args, kwargs = _prepare_state(capsule, module)
            _restore_generators(capsule.rng_state, args, kwargs, module)
            _restore_rng(capsule.rng_state)
            _sync_devices()
            started = time.perf_counter()
            trial_trace = []
            with trace_collectives(torch, trial_trace):
                output = _run_with_state(
                    fn, args, kwargs, capsule.execution_state, module)
            _sync_devices()
            timings.append((time.perf_counter() - started) * 1000.0)
            observed_trace = normalized_collective_trace(trial_trace)
            if ((expected_trace or _distributed_initialized(torch))
                    and observed_trace != expected_trace):
                return _distributed_mismatch_result(
                    capsule, det_report, repeat,
                    expected_rank=capsule.execution_state.get("distributed_rank"),
                    expected_world_size=capsule.execution_state.get(
                        "distributed_world_size"),
                    observed=_distributed_topology(torch),
                    expected_trace=expected_trace, observed_trace=observed_trace)
            values = _trial_output_values(output)
            accumulator.update(values)
            trials.append(values)
            if len(trials) > 2:
                trials.pop(0)
            if trial == int(repeat) - 1:
                actual_post = capture_state_snapshot(
                    args=args, kwargs=kwargs, outputs=output, module=module)
                actual_output = reconstruct_from_snapshot(actual_post).get("outputs")
    finally:
        torch.set_grad_enabled(previous_grad)
    trials.accumulator = accumulator
    warm = timings[1:]
    if auto_reference:
        reference_fn = _auto_fp32_reference(
            fn, module, actual_output, actual_post)
    reference_result = _reference_check(
        capsule, module, reference_fn, actual_post, reference_atol, reference_rtol)
    fakequant_result = _fakequant_check(
        capsule, fn, fakequant_contract, original_float_weight, actual_output)
    if reference_result.get("status") in {"pass", "fail"}:
        reference_result["verdict"] = (
            "pass" if reference_result["status"] == "pass" else "real_numerical_bug")
    if fakequant_result.get("status") == "ok":
        fakequant_result["verdict"] = _fakequant_verdict(
            fakequant_result, reference_atol)
    result = {
        "status": "ok",
        "repeat": int(repeat),
        "cold_start_ms": timings[0] if timings else None,
        "warm_replay_p50_ms": statistics.median(warm) if warm else None,
        "warm_replay_p95_ms": _percentile(warm, 0.95) if warm else None,
        "measurements_ms": timings,
        "det_lock": det_report,
        "det_lock_warning": det_report.get("warning"),
        "jitter_repeat": _jitter_repeat(trials),
        "reference_check": reference_result,
        "fakequant_check": fakequant_result,
        "verdict": _replay_verdict(
            reference_result, fakequant_result, reference_atol,
            reference_requested=reference_requested,
            fakequant_requested=fakequant_requested),
        "outputs": actual_output,
    }
    return result
