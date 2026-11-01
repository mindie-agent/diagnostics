"""Automatic true-FP32 reference execution for ``call_repro``.

[Role]
  Build the ``reference_check`` callable used when ``mode='reference_check'``
  is requested without an explicit ``reference_fn``.

[Invariants]
  * ``torch`` remains a lazy import; package import stays dependency-free.
  * Floating inputs and module parameters, buffers and tensor attributes
    execute as ``float32`` while preserving storage aliasing and container
    identity across args, kwargs and module state.
  * Mixed-dtype views sharing one storage with a half/BF16 tensor report
    ``unsupported`` (``mixed_dtype_storage_alias``) rather than silently
    breaking aliasing.
  * ``Tensor.to(float16|bfloat16)``, ``.bfloat16()``, ``.half()`` and active
    autocast contexts inside the current thread execute as ``float32``.
  * Concurrent threads never clobber ``torch.Tensor.to`` or leak state.

[Timeouts & Locks]
  Synchronous caller-controlled execution; one short lock guards the refcounted
  method-patch installation/removal across threads.

[Side-Effect Policy]
  Temporarily replaces module parameter/buffer/attribute entries and intercepts
  half-precision casts on the current thread only, restoring all state in
  ``finally``.
"""
from __future__ import annotations

import contextlib
import functools
import threading

from .storage_graph import _module_attribute_items, _require_torch, _storage_identity
from .tree_utils import (
    _child_items,
    _classify,
    _rebuild,
    flatten_tree_with_paths,
    path_to_string,
)

_PATCH_LOCK = threading.Lock()
_PATCH_ACTIVE = 0
_LOCAL = threading.local()
_ORIG_TO = None
_ORIG_FLOAT = None
_ORIG_BF16 = None
_ORIG_HALF = None


class _UnsupportedAutoReferenceError(RuntimeError):
    """Raised when automatic FP32 conversion cannot preserve storage semantics."""


def _transform_tree(value, leaf_fn, path=(), canon_path=None, memo=None,
                    canonical_paths=None):
    """Transform a PyTree tracking both current and canonical structural paths."""
    if memo is None:
        memo = {}
    if canon_path is None:
        canon_path = path
    kind, _ = _classify(value)
    if kind in {"leaf", "tensor"}:
        return leaf_fn(path, canon_path, value)
    obj_id = id(value)
    if obj_id in memo:
        return memo[obj_id]
    if canonical_paths is not None and obj_id in canonical_paths:
        canon_path = canonical_paths[obj_id]
    if kind == "dict":
        memo[obj_id] = value
        for key, child in list(value.items()):
            value[key] = _transform_tree(
                child, leaf_fn, path + (key,), canon_path + (key,),
                memo, canonical_paths)
        return value
    if kind == "list":
        memo[obj_id] = value
        for idx, child in enumerate(list(value)):
            value[idx] = _transform_tree(
                child, leaf_fn, path + (idx,), canon_path + (idx,),
                memo, canonical_paths)
        return value
    if kind in {"dataclass", "slots", "object"}:
        memo[obj_id] = value
        for name, child in _child_items(value, kind):
            object.__setattr__(
                value,
                name,
                _transform_tree(
                    child, leaf_fn, path + (name,), canon_path + (name,),
                    memo, canonical_paths),
            )
        return value
    items = [
        (key, _transform_tree(
            child, leaf_fn, path + (key,), canon_path + (key,),
            memo, canonical_paths))
        for key, child in _child_items(value, kind)
    ]
    rebuilt = _rebuild(value, kind, items)
    if canonical_paths is not None and obj_id in canonical_paths:
        canonical_paths.setdefault(id(rebuilt), canonical_paths[obj_id])
    memo[obj_id] = rebuilt
    return rebuilt


class _FP32StorageMapper:
    """Convert half/bf16 views to FP32 and back while preserving shared storages."""

    def __init__(self, actual_post=None):
        self.torch = _require_torch()
        self.storages: dict[tuple[str, int, object], tuple] = {}
        self.fp32_cdata_to_orig: dict[int, object] = {}
        self.restored_storages: dict[tuple[str, int, object], object] = {}
        self.tensor_memo: dict[tuple[int, bool], object] = {}
        self.reverse_tensor_memo: dict[int, object] = {}
        self.dtypes_by_path: dict[tuple, object] = {}
        self.post_dtypes_by_str: dict[str, object] = {}
        self.storage_real_dtype: dict[int, object] = {}
        self.canonical_paths: dict[int, tuple] = {}
        self.convert_container_memo: dict[int, object] = {}
        self.restore_container_memo: dict[int, object] = {}
        if actual_post is not None:
            for view in getattr(actual_post, "views", ()):
                dt_name = getattr(view, "dtype_str", "").rsplit(".", 1)[-1]
                dt = getattr(self.torch, dt_name, None)
                if isinstance(dt, self.torch.dtype) and dt.is_floating_point:
                    self.post_dtypes_by_str[view.tree_path] = dt

    def _target_dtype(self, current_path, canon_path):
        if self.post_dtypes_by_str:
            dtype = self.post_dtypes_by_str.get(path_to_string(current_path))
            if dtype is not None:
                return dtype
            if canon_path != current_path:
                dtype = self.post_dtypes_by_str.get(path_to_string(canon_path))
                if dtype is not None:
                    return dtype
        dtype = self.dtypes_by_path.get(current_path)
        if dtype is None and canon_path != current_path:
            dtype = self.dtypes_by_path.get(canon_path)
        return dtype

    def _record_tree(self, value, path, stack):
        torch = self.torch
        kind, _ = _classify(value)
        if kind in {"leaf", "tensor"}:
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                if value.layout == torch.strided:
                    self.storage_real_dtype[_storage_identity(value.untyped_storage())] = (
                        value.dtype
                    )
                if value.dtype != torch.float32:
                    self.dtypes_by_path[path] = value.dtype
            return
        obj_id = id(value)
        if obj_id in stack:
            return
        self.canonical_paths.setdefault(obj_id, path)
        stack.add(obj_id)
        try:
            for key, child in _child_items(value, kind):
                self._record_tree(child, path + (key,), stack)
        finally:
            stack.discard(obj_id)

    def record_inputs_and_attrs(self, args, kwargs, attr_items):
        self._record_tree(args, ("args",), set())
        self._record_tree(kwargs, ("kwargs",), set())
        for key, _, _, attr_val in attr_items:
            self._record_tree(attr_val, (f"module.attributes.{key}",), set())

    def check_mixed_dtype_aliases(self, args, kwargs, module):
        torch = self.torch
        dtypes_by_storage: dict[tuple[str, int], set] = {}

        def visit(value):
            for _, leaf in flatten_tree_with_paths(value):
                if isinstance(leaf, torch.Tensor) and leaf.layout == torch.strided:
                    storage = leaf.untyped_storage()
                    if int(storage.nbytes()) == 0:
                        continue
                    key = (str(leaf.device), _storage_identity(storage))
                    dtypes_by_storage.setdefault(key, set()).add(leaf.dtype)

        visit(args)
        visit(kwargs)
        if module is not None:
            for submodule in module.modules():
                for table_name in ("_parameters", "_buffers"):
                    for tensor in getattr(submodule, table_name).values():
                        if tensor is not None:
                            visit(tensor)
            for _, _, _, attr_val in _module_attribute_items(module):
                visit(attr_val)
        for dtypes in dtypes_by_storage.values():
            has_narrow_float = any(
                dt.is_floating_point and dt != torch.float32 for dt in dtypes)
            if has_narrow_float and len(dtypes) > 1:
                raise _UnsupportedAutoReferenceError("mixed_dtype_storage_alias")

    def convert_tensor(self, tensor, as_parameter=False):
        torch = self.torch
        if (not isinstance(tensor, torch.Tensor)
                or not tensor.is_floating_point()
                or tensor.dtype == torch.float32):
            return tensor
        memo_key = (id(tensor), bool(as_parameter))
        if memo_key in self.tensor_memo:
            return self.tensor_memo[memo_key]
        if tensor.layout != torch.strided:
            converted = tensor.detach().to(torch.float32)
            if as_parameter:
                converted = torch.nn.Parameter(
                    converted, requires_grad=bool(tensor.requires_grad))
            self.tensor_memo[memo_key] = converted
            self.reverse_tensor_memo[id(converted)] = tensor
            return converted
        storage = tensor.untyped_storage()
        skey = (str(tensor.device), _storage_identity(storage), tensor.dtype)
        entry = self.storages.get(skey)
        if entry is None:
            itemsize = int(tensor.element_size())
            numel = int(storage.nbytes()) // max(itemsize, 1)
            orig_flat = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
            orig_flat.set_(storage, 0, (numel,), (1,))
            fp32_flat = orig_flat.detach().to(torch.float32)
            fp32_initial = fp32_flat.clone()
            entry = (orig_flat, fp32_flat, fp32_initial)
            self.storages[skey] = entry
            fp32_cdata = _storage_identity(fp32_flat.untyped_storage())
            self.fp32_cdata_to_orig[fp32_cdata] = orig_flat
            self.storage_real_dtype[fp32_cdata] = tensor.dtype
        _, fp32_flat, _ = entry
        view = torch.empty(0, dtype=torch.float32, device=tensor.device)
        view.set_(
            fp32_flat.untyped_storage(),
            int(tensor.storage_offset()),
            tuple(int(dim) for dim in tensor.shape),
            tuple(int(stride) for stride in tensor.stride()),
        )
        if as_parameter:
            view = torch.nn.Parameter(
                view, requires_grad=bool(tensor.requires_grad))
        elif tensor.requires_grad:
            view.requires_grad_(True)
        self.tensor_memo[memo_key] = view
        self.reverse_tensor_memo[id(view)] = tensor
        return view

    def convert_tree(self, value, path=()):
        return _transform_tree(
            value,
            lambda _path, _canon, leaf: self.convert_tensor(leaf, False),
            path=path,
            memo=self.convert_container_memo,
            canonical_paths=self.canonical_paths,
        )

    def restore_tree(self, value, path=()):
        return _transform_tree(
            value,
            lambda p, cp, leaf: self.restore_tensor(
                leaf, self._target_dtype(p, cp), as_parameter=False),
            path=path,
            memo=self.restore_container_memo,
            canonical_paths=self.canonical_paths,
        )

    def restore_tensor(self, tensor, target_dtype, as_parameter=False):
        torch = self.torch
        if not isinstance(tensor, torch.Tensor):
            return tensor
        orig = self.reverse_tensor_memo.get(id(tensor))
        if orig is not None:
            if target_dtype is None or orig.dtype == target_dtype:
                return orig
            if tensor.dtype == target_dtype:
                cloned = tensor.clone()
                if as_parameter:
                    return torch.nn.Parameter(
                        cloned, requires_grad=bool(tensor.requires_grad))
                return cloned
        if (target_dtype is None
                or not tensor.is_floating_point()
                or tensor.dtype == target_dtype):
            return tensor
        if tensor.layout != torch.strided or tensor.dtype != torch.float32:
            casted = _ORIG_TO(tensor, target_dtype) if _ORIG_TO else tensor.to(target_dtype)
            if as_parameter:
                return torch.nn.Parameter(
                    casted, requires_grad=bool(tensor.requires_grad))
            return casted
        fp32_storage = tensor.untyped_storage()
        fp32_cdata = _storage_identity(fp32_storage)
        orig_flat = self.fp32_cdata_to_orig.get(fp32_cdata)
        if orig_flat is not None and orig_flat.dtype == target_dtype:
            narrow_storage = orig_flat.untyped_storage()
        else:
            rkey = (str(tensor.device), fp32_cdata, target_dtype)
            narrow_storage = self.restored_storages.get(rkey)
            if narrow_storage is None:
                numel = int(fp32_storage.nbytes()) // 4
                fp32_flat = torch.empty(0, dtype=torch.float32, device=tensor.device)
                fp32_flat.set_(fp32_storage, 0, (numel,), (1,))
                narrow_flat = (
                    _ORIG_TO(fp32_flat.detach(), target_dtype)
                    if _ORIG_TO else fp32_flat.detach().to(target_dtype))
                narrow_storage = narrow_flat.untyped_storage()
                self.restored_storages[rkey] = narrow_storage
        view = torch.empty(0, dtype=target_dtype, device=tensor.device)
        view.set_(
            narrow_storage,
            int(tensor.storage_offset()),
            tuple(int(dim) for dim in tensor.shape),
            tuple(int(stride) for stride in tensor.stride()),
        )
        if as_parameter:
            return torch.nn.Parameter(
                view, requires_grad=bool(tensor.requires_grad))
        if tensor.requires_grad and (
                target_dtype.is_floating_point or target_dtype.is_complex):
            view.requires_grad_(True)
        return view

    def convert_module(self, module, attr_items):
        torch = self.torch
        originals = []
        attr_originals = []
        if module is None:
            return originals, attr_originals
        for submodule in module.modules():
            for table_name in ("_parameters", "_buffers"):
                table = getattr(submodule, table_name)
                is_param = table_name == "_parameters"
                for name, tensor in list(table.items()):
                    if (tensor is None or not tensor.is_floating_point()
                            or tensor.dtype == torch.float32):
                        continue
                    converted = self.convert_tensor(tensor, as_parameter=is_param)
                    originals.append((table, name, tensor, converted, is_param))
                    table[name] = converted
        seen_attrs: set[tuple[int, str]] = set()
        for key, owner, attr_name, attr_val in attr_items:
            dedup_key = (id(owner), attr_name)
            if dedup_key in seen_attrs:
                continue
            seen_attrs.add(dedup_key)
            attr_path = (f"module.attributes.{key}",)
            attr_originals.append((owner, attr_name, attr_path))
            object.__setattr__(
                owner, attr_name, self.convert_tree(attr_val, attr_path))
        return originals, attr_originals

    def sync_and_restore(self, cast_args, cast_kwargs, originals,
                         attr_originals, module=None):
        torch = self.torch
        with torch.no_grad():
            for orig_flat, fp32_flat, fp32_initial in self.storages.values():
                if orig_flat.numel() == 0:
                    continue
                both_nan = torch.isnan(fp32_flat) & torch.isnan(fp32_initial)
                changed = (fp32_flat != fp32_initial) & (~both_nan)
                if bool(changed.any().item()):
                    updated = (
                        _ORIG_TO(fp32_flat, orig_flat.dtype)
                        if _ORIG_TO else fp32_flat.to(orig_flat.dtype))
                    orig_flat.copy_(torch.where(changed, updated, orig_flat))
            self.restore_tree(cast_args, ("args",))
            self.restore_tree(cast_kwargs, ("kwargs",))
            seen_attrs: set[tuple[int, str]] = set()
            for owner, attr_name, attr_path in attr_originals:
                seen_attrs.add((id(owner), attr_name))
                if hasattr(owner, attr_name):
                    current_val = getattr(owner, attr_name)
                    object.__setattr__(
                        owner,
                        attr_name,
                        self.restore_tree(current_val, attr_path),
                    )
            if module is not None:
                for key, owner, attr_name, current_val in _module_attribute_items(module):
                    dedup_key = (id(owner), attr_name)
                    if dedup_key in seen_attrs:
                        continue
                    seen_attrs.add(dedup_key)
                    attr_path = (f"module.attributes.{key}",)
                    object.__setattr__(
                        owner,
                        attr_name,
                        self.restore_tree(current_val, attr_path),
                    )
            for table, name, tensor, converted, is_param in reversed(originals):
                current = table.get(name)
                if current is converted or current is None:
                    table[name] = tensor
                else:
                    table[name] = self.restore_tensor(
                        current, tensor.dtype, as_parameter=is_param)


def _real_dtype(tensor):
    mapper = getattr(_LOCAL, "mapper", None)
    if mapper is None or tensor.layout != mapper.torch.strided:
        return tensor.dtype
    return mapper.storage_real_dtype.get(
        _storage_identity(tensor.untyped_storage()), tensor.dtype)


def _finalize_cast(tensor, out, target_dtype):
    torch = _require_torch()
    if not isinstance(out, torch.Tensor):
        return out
    mapper = getattr(_LOCAL, "mapper", None)
    src_dtype = _real_dtype(tensor)
    if (target_dtype is not None
            and src_dtype != target_dtype
            and tensor.layout == out.layout == torch.strided
            and _storage_identity(out.untyped_storage()) == _storage_identity(tensor.untyped_storage())):
        out = out.clone()
    if mapper is not None and target_dtype is not None and out.layout == torch.strided:
        mapper.storage_real_dtype[_storage_identity(out.untyped_storage())] = target_dtype
    return out


def _fp32_to(tensor, *args, **kwargs):
    if getattr(_LOCAL, "depth", 0) <= 0:
        return _ORIG_TO(tensor, *args, **kwargs)
    torch = _require_torch()
    target_dtype = None
    args = list(args)
    for index, arg in enumerate(args):
        if isinstance(arg, torch.dtype):
            target_dtype = arg
            if arg in (torch.bfloat16, torch.float16):
                args[index] = torch.float32
        elif isinstance(arg, torch.Tensor):
            target_dtype = _real_dtype(arg)
            if arg.dtype in (torch.bfloat16, torch.float16):
                args[index] = _ORIG_TO(arg, torch.float32)
    if isinstance(kwargs.get("dtype"), torch.dtype):
        target_dtype = kwargs["dtype"]
        if target_dtype in (torch.bfloat16, torch.float16):
            kwargs["dtype"] = torch.float32
    out = _ORIG_TO(tensor, *args, **kwargs)
    return _finalize_cast(tensor, out, target_dtype)


def _fp32_float(tensor, *args, **kwargs):
    if getattr(_LOCAL, "depth", 0) <= 0:
        return _ORIG_FLOAT(tensor, *args, **kwargs)
    out = _ORIG_FLOAT(tensor, *args, **kwargs)
    return _finalize_cast(tensor, out, _require_torch().float32)


def _fp32_bf16(tensor, *args, **kwargs):
    if getattr(_LOCAL, "depth", 0) <= 0:
        return _ORIG_BF16(tensor, *args, **kwargs)
    torch = _require_torch()
    out = _ORIG_TO(tensor, torch.float32, *args, **kwargs)
    return _finalize_cast(tensor, out, torch.bfloat16)


def _fp32_half(tensor, *args, **kwargs):
    if getattr(_LOCAL, "depth", 0) <= 0:
        return _ORIG_HALF(tensor, *args, **kwargs)
    torch = _require_torch()
    out = _ORIG_TO(tensor, torch.float32, *args, **kwargs)
    return _finalize_cast(tensor, out, torch.float16)


def _make_dtype_tracker_mode(torch, mapper):
    base = getattr(getattr(torch, "overrides", None), "TorchFunctionMode", None)
    if base is None:
        return contextlib.nullcontext()

    class _TrackerMode(base):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            if func in (_ORIG_TO, _ORIG_FLOAT, _ORIG_BF16, _ORIG_HALF):
                return func(*args, **(kwargs or {}))
            call_kwargs = dict(kwargs) if kwargs else {}
            explicit_dtype = call_kwargs.get("dtype")
            if (isinstance(explicit_dtype, torch.dtype)
                    and explicit_dtype in (torch.bfloat16, torch.float16)):
                call_kwargs["dtype"] = torch.float32
            in_dtypes = [
                _real_dtype(leaf)
                for _, leaf in flatten_tree_with_paths((args, kwargs))
                if isinstance(leaf, torch.Tensor) and leaf.is_floating_point()
            ]
            if (isinstance(explicit_dtype, torch.dtype)
                    and explicit_dtype.is_floating_point):
                inferred = explicit_dtype
            elif in_dtypes:
                inferred = functools.reduce(torch.promote_types, in_dtypes)
            else:
                inferred = torch.get_default_dtype()
            out = func(*args, **call_kwargs)
            if mapper is not None:
                for _, leaf in flatten_tree_with_paths(out):
                    if (isinstance(leaf, torch.Tensor)
                            and leaf.is_floating_point()
                            and leaf.layout == torch.strided):
                        cdata = _storage_identity(leaf.untyped_storage())
                        mapper.storage_real_dtype.setdefault(cdata, inferred)
            return out

    return _TrackerMode()


@contextlib.contextmanager
def _fp32_execution_scope(mapper=None):
    global _PATCH_ACTIVE, _ORIG_TO, _ORIG_FLOAT, _ORIG_BF16, _ORIG_HALF
    torch = _require_torch()
    with _PATCH_LOCK:
        if _PATCH_ACTIVE == 0:
            _ORIG_TO = torch.Tensor.to
            _ORIG_FLOAT = getattr(torch.Tensor, "float", None)
            _ORIG_BF16 = getattr(torch.Tensor, "bfloat16", None)
            _ORIG_HALF = getattr(torch.Tensor, "half", None)
            try:
                torch.Tensor.to = _fp32_to
                if _ORIG_FLOAT is not None:
                    torch.Tensor.float = _fp32_float
                if _ORIG_BF16 is not None:
                    torch.Tensor.bfloat16 = _fp32_bf16
                if _ORIG_HALF is not None:
                    torch.Tensor.half = _fp32_half
            except (AttributeError, TypeError):
                pass
        _PATCH_ACTIVE += 1
    prev_mapper = getattr(_LOCAL, "mapper", None)
    _LOCAL.mapper = mapper
    _LOCAL.depth = getattr(_LOCAL, "depth", 0) + 1
    try:
        with contextlib.ExitStack() as stack:
            for dev_type in ("cpu", "cuda", "npu"):
                try:
                    stack.enter_context(
                        torch.autocast(device_type=dev_type, enabled=False))
                except (RuntimeError, ValueError, TypeError):
                    pass
            stack.enter_context(_make_dtype_tracker_mode(torch, mapper))
            yield
    finally:
        _LOCAL.depth = max(0, getattr(_LOCAL, "depth", 1) - 1)
        _LOCAL.mapper = prev_mapper
        with _PATCH_LOCK:
            _PATCH_ACTIVE = max(0, _PATCH_ACTIVE - 1)
            if _PATCH_ACTIVE == 0 and _ORIG_TO is not None:
                try:
                    torch.Tensor.to = _ORIG_TO
                    if _ORIG_FLOAT is not None:
                        torch.Tensor.float = _ORIG_FLOAT
                    if _ORIG_BF16 is not None:
                        torch.Tensor.bfloat16 = _ORIG_BF16
                    if _ORIG_HALF is not None:
                        torch.Tensor.half = _ORIG_HALF
                except (AttributeError, TypeError):
                    pass


def _match_output_dtypes(output, actual_output, mapper=None):
    torch = _require_torch()
    if actual_output is None:
        return output
    dtypes_by_path = {
        tuple(path): leaf.dtype
        for path, leaf in flatten_tree_with_paths(actual_output)
        if isinstance(leaf, torch.Tensor)
    }

    def restore(path, _canon_path, leaf):
        if not isinstance(leaf, torch.Tensor):
            return leaf
        dtype = dtypes_by_path.get(path)
        if dtype is None or leaf.dtype == dtype:
            return leaf
        if mapper is not None:
            return mapper.restore_tensor(leaf, dtype, as_parameter=False)
        return _ORIG_TO(leaf, dtype) if _ORIG_TO is not None else leaf.to(dtype)

    return _transform_tree(output, restore)


def _fakequant_verdict(check, atol):
    if check.get("fakequant_vs_kernel", {}).get("status") not in {None, "ok"}:
        return "real_numerical_bug"
    if float(check.get("fakequant_vs_kernel", {}).get("max_abs_err", 0.0)) > atol:
        return "real_numerical_bug"
    if float(check.get("float_vs_fakequant", {}).get("max_abs_err", 0.0)) > atol:
        return "inherent_quantization_error"
    return "pass"


def _replay_verdict(reference_check, fakequant_check, atol, *,
                    reference_requested=False, fakequant_requested=False):
    if (reference_check.get("status") == "error"
            or fakequant_check.get("status") == "error"):
        return "error"
    if reference_check.get("status") == "fail":
        return "real_numerical_bug"
    if fakequant_check.get("status") == "ok":
        fq_verdict = _fakequant_verdict(fakequant_check, atol)
        if fq_verdict != "pass":
            return fq_verdict
    if ((reference_requested and reference_check.get("status") == "unsupported")
            or (fakequant_requested and fakequant_check.get("status") == "unsupported")):
        return "unsupported"
    return "pass"


def _auto_fp32_reference(fn, module, actual_output, actual_post=None):
    """Return a callable that executes ``fn`` as a true FP32 reference."""
    torch = _require_torch()

    def reference(*args, **kwargs):
        target_module = module if module is not None else (
            fn if isinstance(fn, torch.nn.Module) else None)
        attr_items = (
            list(_module_attribute_items(target_module))
            if target_module is not None else []
        )
        mapper = _FP32StorageMapper(actual_post=actual_post)
        mapper.check_mixed_dtype_aliases(args, kwargs, target_module)
        mapper.record_inputs_and_attrs(args, kwargs, attr_items)
        cast_args = mapper.convert_tree(args, ("args",))
        cast_kwargs = mapper.convert_tree(kwargs, ("kwargs",))
        originals, attr_originals = mapper.convert_module(
            target_module, attr_items)
        try:
            with _fp32_execution_scope(mapper):
                output = fn(*cast_args, **cast_kwargs)
        finally:
            mapper.sync_and_restore(
                cast_args, cast_kwargs, originals, attr_originals,
                module=target_module)
        return _match_output_dtypes(output, actual_output, mapper=mapper)

    return reference
