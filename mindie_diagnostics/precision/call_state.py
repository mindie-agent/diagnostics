"""RNG, execution-mode and distributed-collective capture for call replay.

[Role]
  Snapshot and restore CPU/CUDA/NPU RNG streams and explicit generators,
  record the caller's grad/autocast/training/distributed execution state, and
  trace distributed collective calls so replay can detect topology changes.

[Invariants]
  * RNG restoration targets the current process/device and never fabricates
    state that was not captured.
  * Collective tracing only wraps ``torch.distributed`` while active and always
    restores the exact original callables in ``finally``.
  * The recorded trace is a list of ``(op_name, group_rank, group_world_size)``
    tuples; it never contains tensor payloads.

[Timeouts & Locks]
  Synchronous host-side bookkeeping.  No sleeps, subprocesses or locks.

[Side-Effect Policy]
  ``_apply_det_lock`` may set documented HCCL environment variables.  Tracing
  temporarily monkey-patches ``torch.distributed`` functions.
"""
from __future__ import annotations
import inspect
import os
import sys
from contextlib import contextmanager
_DET_LOCK_ENV = {
    "CLOSE_MATMUL_K_SHIFT": "1",
    "HCCL_DETERMINISTIC": "true",
    "LCCL_DETERMINISTIC": "1",
}
_COLLECTIVE_OPS = (
    "all_reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter",
    "reduce_scatter_tensor", "broadcast", "reduce", "all_to_all",
    "all_to_all_single", "barrier",
)
def _require_torch():
    from .storage_graph import _require_torch as getter
    return getter()
def _capture_rng():
    torch = _require_torch()
    state = {"cpu": torch.random.get_rng_state()}
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    npu = getattr(torch, "npu", None)
    if npu is not None:
        try:
            if npu.is_available() and hasattr(npu, "get_rng_state"):
                state["npu"] = npu.get_rng_state()
                if hasattr(npu, "get_rng_state_all"):
                    state["npu_all"] = npu.get_rng_state_all()
        except (RuntimeError, OSError):
            pass
    return state
def _restore_rng(state):
    torch = _require_torch()
    if not state:
        return
    if state.get("cpu") is not None:
        torch.random.set_rng_state(state["cpu"].to("cpu"))
    if "cuda" in state and hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.to("cpu") for item in state["cuda"]])
    npu = getattr(torch, "npu", None)
    if npu is not None and "npu" in state and npu.is_available():
        try:
            npu.set_rng_state(state["npu"].to("cpu"))
            if "npu_all" in state and hasattr(npu, "set_rng_state_all"):
                npu.set_rng_state_all([item.to("cpu") for item in state["npu_all"]])
        except (RuntimeError, OSError):
            pass
def _generator_map(args, kwargs, module):
    torch = _require_torch()
    from .storage_graph import flatten_tree_with_paths, path_to_string
    result = {}
    def collect(prefix, value):
        for path, leaf in flatten_tree_with_paths(value):
            if isinstance(leaf, torch.Generator):
                result[path_to_string((prefix,) + tuple(path))] = leaf
    if args is not None:
        collect("args", args)
    if kwargs is not None:
        collect("kwargs", kwargs)
    if module is not None:
        for module_name, submodule in module.named_modules(remove_duplicate=False):
            prefix = f"{module_name}." if module_name else ""
            for name, value in vars(submodule).items():
                if isinstance(value, torch.Generator):
                    result[f"module.{prefix}{name}"] = value
    return result
def _capture_generators(args, kwargs, module):
    return {path: generator.get_state()
            for path, generator in _generator_map(args, kwargs, module).items()}
def _restore_generators(state, args, kwargs, module):
    saved = state.get("generators") if state else None
    if not saved:
        return
    live = _generator_map(args, kwargs, module)
    for path, generator_state in saved.items():
        generator = live.get(path)
        if generator is not None:
            generator.set_state(generator_state)
def _distributed_initialized(torch):
    distributed = getattr(torch, "distributed", None)
    if distributed is None or not hasattr(distributed, "is_initialized"):
        return False
    try:
        return bool(distributed.is_initialized())
    except (RuntimeError, OSError):
        return False
def _distributed_topology(torch):
    distributed = getattr(torch, "distributed", None)
    if not _distributed_initialized(torch):
        return None
    try:
        return int(distributed.get_rank()), int(distributed.get_world_size())
    except (RuntimeError, OSError, AttributeError):
        return None
def _backend_initialized(torch, name):
    backend = getattr(torch, name, None)
    getter = getattr(backend, "is_initialized", None)
    if getter is None:
        return False
    try:
        return bool(getter())
    except (RuntimeError, OSError):
        return False
def _apply_det_lock(torch, det_lock):
    report = {
        "applied": bool(det_lock),
        "distributed_initialized_before": _distributed_initialized(torch),
        "cuda_initialized_before": _backend_initialized(torch, "cuda"),
        "npu_initialized_before": _backend_initialized(torch, "npu"),
    }
    if (report["distributed_initialized_before"] or report["cuda_initialized_before"]
            or report["npu_initialized_before"]):
        report["warning"] = "backend_already_initialized"
    if det_lock:
        for key, value in _DET_LOCK_ENV.items():
            os.environ[key] = value
        report["effective"] = dict(_DET_LOCK_ENV)
    return report
def _capture_execution_state(fn_or_module):
    torch = _require_torch()
    state = {
        "grad_enabled": bool(torch.is_grad_enabled()),
        "inference_mode": bool(torch.is_inference_mode_enabled()),
        "autocast": False,
        "autocast_device_type": None,
        "autocast_dtype": None,
        "training": None,
        "submodule_training": None,
        "torch_version": torch.__version__,
        "torch_npu_version": None,
        "backend_env": {key: os.environ.get(key) for key in _DET_LOCK_ENV},
        "distributed_initialized": _distributed_initialized(torch),
        "distributed_rank": None,
        "distributed_world_size": None,
    }
    topology = _distributed_topology(torch)
    if topology is not None:
        state["distributed_rank"], state["distributed_world_size"] = topology
    try:
        torch_npu = __import__("torch_npu")
        state["torch_npu_version"] = getattr(torch_npu, "__version__", None)
    except ImportError:
        pass
    if isinstance(fn_or_module, torch.nn.Module):
        state["training"] = bool(fn_or_module.training)
        state["submodule_training"] = {
            name: bool(submodule.training)
            for name, submodule in fn_or_module.named_modules(remove_duplicate=False)
        }
    try:
        state["autocast"] = bool(torch.is_autocast_enabled())
    except TypeError:
        pass
    for device_type in ("npu", "cuda", "cpu"):
        getter = getattr(torch, "get_autocast_dtype", None)
        if getter is None:
            break
        try:
            if torch.is_autocast_enabled(device_type):
                state["autocast"] = True
                state["autocast_device_type"] = device_type
                state["autocast_dtype"] = str(getter(device_type))
                break
        except (TypeError, RuntimeError):
            continue
    return state
def _callable_metadata(fn_or_module):
    torch = _require_torch()
    metadata = {
        "is_module": isinstance(fn_or_module, torch.nn.Module),
        "type": type(fn_or_module).__name__,
        "module": getattr(fn_or_module, "__module__", None),
        "qualname": getattr(fn_or_module, "__qualname__", None),
        "name": getattr(fn_or_module, "__name__", None),
        "torch_version": torch.__version__,
        "distributed_initialized": _distributed_initialized(torch),
    }
    if metadata["is_module"]:
        metadata["class_module"] = type(fn_or_module).__module__
        metadata["class_qualname"] = type(fn_or_module).__qualname__
        metadata["training"] = bool(fn_or_module.training)
        metadata["extra_repr"] = _bounded_repr(fn_or_module)
    else:
        metadata["import_path"] = (
            f"{metadata['module']}:{metadata['qualname']}"
            if metadata["module"] and metadata["qualname"] and "<" not in metadata["qualname"]
            else None
        )
        metadata["repr"] = _bounded_repr(fn_or_module)
    return metadata
def _bounded_repr(value, limit=512):
    try:
        text = repr(value)
    except Exception:  # pragma: no cover - exotic callables
        return None
    return text if len(text) <= limit else text[:limit] + "..."
def _collective_group(fn, args, kwargs):
    group = kwargs.get("group")
    if group is not None:
        return group
    try:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
        return bound.arguments.get("group")
    except (TypeError, ValueError):
        return None
def _record_collective_group(distributed, name, group, trace):
    try:
        rank = int(distributed.get_rank(group))
        world = int(distributed.get_world_size(group))
    except (RuntimeError, OSError, TypeError, AttributeError):
        rank = world = None
    trace.append((name, rank, world))
def _record_collective(distributed, name, fn, args, kwargs, trace):
    _record_collective_group(
        distributed, name, _collective_group(fn, args, kwargs), trace)
@contextmanager
def trace_collectives(torch, trace):
    """Record ``(op, group_rank, group_world_size)`` while a call executes."""
    distributed = getattr(torch, "distributed", None)
    if distributed is None or not _distributed_initialized(torch):
        yield
        return
    distributed_c10d = getattr(distributed, "distributed_c10d", None)
    targets = [t for t in (distributed, distributed_c10d) if t is not None]
    originals = []
    code_map = {}
    wrapped = set()
    try:
        for target in targets:
            for name in _COLLECTIVE_OPS:
                fn = getattr(target, name, None)
                if not callable(fn):
                    continue
                for candidate in (fn, getattr(fn, "__wrapped__", None)):
                    code = getattr(candidate, "__code__", None)
                    if code is not None and code not in code_map:
                        code_map[code] = (name, fn)
                if id(fn) in wrapped:
                    continue
                wrapped.add(id(fn))
                originals.append((target, name, fn))
                profiling = getattr(fn, "__code__", None) is not None
                def wrapper(*args, __fn=fn, __name=name,
                            __profiled=profiling, **kwargs):
                    if not __profiled:
                        _record_collective(distributed, __name, __fn, args, kwargs, trace)
                    return __fn(*args, **kwargs)
                setattr(target, name, wrapper)
        def profile(frame, event, _arg):
            if event != "call" or frame.f_code not in code_map:
                return
            parent = frame.f_back
            while parent is not None:
                if parent.f_code in code_map:
                    return
                parent = parent.f_back
            name, fn = code_map[frame.f_code]
            args = frame.f_locals.get("args")
            kwargs = frame.f_locals.get("kwargs")
            if isinstance(args, tuple) and isinstance(kwargs, dict):
                group = _collective_group(fn, args, kwargs)
            else:
                group = frame.f_locals.get("group")
            _record_collective_group(distributed, name, group, trace)
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            yield
        finally:
            sys.setprofile(previous)
    finally:
        for target, name, fn in originals:
            setattr(target, name, fn)
def normalized_collective_trace(trace):
    return [tuple(item) for item in (trace or ())]
