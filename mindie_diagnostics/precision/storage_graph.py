"""Shared storage-graph snapshot, restore and reconstruction foundation.

[Role]
  Walk arbitrary PyTree/container call states, deduplicate every shared
  ``UntypedStorage`` across args, kwargs, outputs, parameters and buffers
  (including non-persistent buffers), and snapshot/restore exact tensor views.

[Invariants]
  * ``torch`` is imported only inside runtime functions; package import stays
    zero-dependency.
  * A storage appears once in ``StateSnapshot.storages``; tensor views retain
    their exact ``(dtype, shape, stride, storage_offset)``.
  * Restore copies saved bytes into live storage once per storage id, in place,
    under ``torch.no_grad()``: no tensor variable is rebound and overlapping
    views are never written independently.
  * Container identity is captured by object id; reconstruction shares one
    container instance for every occurrence of the same captured container.
  * Scalar leaves inside tuples / namedtuples / registered PyTrees are rebuilt
    so the parent container receives the restored primitive while tensors stay
    live.
  * Opaque NPU internal formats are rejected with ``ValueError`` instead of
    being silently reinterpreted as plain strided memory.
  * Reconstructed views of one storage share one freshly allocated storage.

[Timeouts & Locks]
  Pure synchronous local memory work.  No sleeps, subprocesses, network or
  cross-thread locks.  Callers own any device synchronization at capture time.

[Side-Effect Policy]
  Capture only reads live tensors.  Restore mutates existing storage in place.
  Reconstruct allocates new device tensors.  No files are written here.
"""
from __future__ import annotations
import dataclasses
import enum
from dataclasses import dataclass, field
from typing import Any
from .tree_utils import (
    _NAME_RE,  # noqa: F401 - retained import surface
    _child_items,
    _classify,
    _pytree_flatten,
    _pytree_unflatten_type,
    _rebuild,
    _slot_names,
    flatten_tree_with_paths,
    map_tree,
    path_to_string,
    walk_tree,
)
_TORCH = None
def _require_torch():
    """Import torch lazily so stdlib-only diagnostics commands stay usable."""
    global _TORCH
    if _TORCH is None:
        try:
            import torch  # noqa: WPS433 - deliberate lazy import
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "PyTorch is required for mindie_diagnostics.precision runtime primitives"
            ) from exc
        _TORCH = torch
    return _TORCH
_SUPPORTED_NPU_FORMATS = {0, 1, 2}  # ACL_FORMAT_NCHW / NHWC / ND
_MODULE_INTERNAL_ATTRS = {
    'training', '_parameters', '_buffers', '_non_persistent_buffers_set',
    '_backward_pre_hooks', '_backward_hooks', '_is_full_backward_hook',
    '_forward_hooks', '_forward_hooks_with_kwargs', '_forward_hooks_always_called',
    '_forward_pre_hooks', '_forward_pre_hooks_with_kwargs', '_state_dict_hooks',
    '_state_dict_pre_hooks', '_load_state_dict_pre_hooks',
    '_load_state_dict_post_hooks', '_modules', '_backend',
}
@dataclass(frozen=True)
class StorageNode:
    """One deduplicated untyped storage and its exact saved bytes."""
    storage_id: str
    nbytes: int
    raw_bytes: Any
    device_str: str
@dataclass(frozen=True)
class TensorViewSpec:
    """A tensor's identity inside the captured storage graph."""
    tree_path: str
    storage_id: str
    dtype_str: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    requires_grad: bool
    device_str: str
    npu_format: Any = None
    tensor_id: int | None = None
    is_conj: bool = False
    is_neg: bool = False
@dataclass
class StateSnapshot:
    """Immutable-by-convention capture of one call state."""
    roots: dict[str, Any] = field(default_factory=dict)
    storages: dict[str, StorageNode] = field(default_factory=dict)
    views: list[TensorViewSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    def views_for(self, storage_id: str) -> list[TensorViewSpec]:
        return [view for view in self.views if view.storage_id == storage_id]
@dataclass(frozen=True)
class _TensorRef:
    """Template placeholder retaining a tensor's tree path and identity."""
    tree_path: str
    tensor_id: int | None = None
@dataclass(frozen=True)
class _ContainerNode:
    """Captured container identity plus already-captured child templates."""
    obj_id: int
    kind: str
    klass: type
    items: list[tuple[Any, Any]]
    context: Any = None
# ---------------------------------------------------------------------------
# Capture and restore
def _storage_key(storage) -> str:
    cdata = getattr(storage, "_cdata", None)
    if cdata is not None:
        return f"{storage.device}:{int(cdata)}"
    return f"{storage.device}:{int(storage.data_ptr())}:{int(storage.nbytes())}"
def _storage_bytes(storage):
    torch = _require_torch()
    nbytes = int(storage.nbytes())
    view = torch.empty(0, dtype=torch.uint8, device=storage.device)
    view.set_(storage, 0, (nbytes,), (1,))
    return view.detach().to("cpu", copy=True)
def _npu_format(tensor):
    if tensor.device.type != "npu":
        return None
    try:
        import torch_npu  # noqa: WPS433 - optional Ascend extension
    except ImportError as exc:
        raise ValueError(
            "cannot validate NPU tensor format without torch_npu.get_npu_format") from exc
    getter = getattr(torch_npu, "get_npu_format", None)
    if getter is None:
        raise ValueError(
            "cannot validate NPU tensor format without torch_npu.get_npu_format")
    try:
        raw_format = getter(tensor)
    except Exception as exc:
        raise ValueError(f"cannot read NPU format for {type(tensor).__name__}") from exc
    if isinstance(raw_format, str):
        normalized = raw_format.upper().replace("ACL_FORMAT_", "").strip()
        named = {"ND": 2, "NCHW": 0, "NHWC": 1}
        if normalized not in named:
            raise ValueError(
                f"unsupported opaque NPU tensor format {raw_format!r}; "
                "capture requires npu_format_cast")
        fmt = named[normalized]
    else:
        try:
            fmt = int(getattr(raw_format, "value", raw_format))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cannot interpret NPU format {raw_format!r}") from exc
    supported = set(_SUPPORTED_NPU_FORMATS)
    for name in ("ACL_FORMAT_ND", "ACL_FORMAT_NCHW", "ACL_FORMAT_NHWC"):
        if hasattr(torch_npu, name):
            supported.add(int(getattr(torch_npu, name)))
    if fmt not in supported:
        raise ValueError(
            f"unsupported opaque NPU tensor format {fmt}; capture requires npu_format_cast"
        )
    return fmt
def _validate_tensor_layout(tensor):
    torch = _require_torch()
    if tensor.layout != torch.strided:
        raise ValueError(f"non-strided tensor layout {tensor.layout} is not supported")
    if tensor.device.type == "meta":
        raise ValueError("meta tensors have no storage bytes to capture")
    if tensor.is_quantized:
        raise ValueError("quantized tensors cannot be restored through raw storage bytes")
def _dtype_name(dtype) -> str:
    return str(dtype)
def _dtype_from_name(name: str):
    torch = _require_torch()
    short = name.rsplit(".", 1)[-1]
    dtype = getattr(torch, short, None)
    if dtype is None:
        raise ValueError(f"unknown dtype {name!r}")
    return dtype
def _record_view(snapshot: StateSnapshot, tensor, path: str):
    _validate_tensor_layout(tensor)
    npu_format = _npu_format(tensor)
    storage = tensor.untyped_storage()
    storage_id = _storage_key(storage)
    if storage_id not in snapshot.storages:
        snapshot.storages[storage_id] = StorageNode(
            storage_id=storage_id,
            nbytes=int(storage.nbytes()),
            raw_bytes=_storage_bytes(storage),
            device_str=str(storage.device),
        )
    tensor_id = id(tensor)
    snapshot.views.append(
        TensorViewSpec(
            tree_path=path,
            storage_id=storage_id,
            dtype_str=_dtype_name(tensor.dtype),
            shape=tuple(int(item) for item in tensor.shape),
            stride=tuple(int(item) for item in tensor.stride()),
            storage_offset=int(tensor.storage_offset()),
            requires_grad=bool(tensor.requires_grad),
            device_str=str(tensor.device),
            npu_format=npu_format,
            tensor_id=tensor_id,
            is_conj=bool(tensor.is_conj()),
            is_neg=bool(tensor.is_neg()),
        )
    )
    return _TensorRef(path, tensor_id=tensor_id)
def _capture_root(snapshot: StateSnapshot, root: str, value) -> Any:
    stack: set[int] = set()
    containers: dict[int, _ContainerNode] = {}
    def rebuild(node, path):
        kind, _ = _classify(node)
        if kind == "tensor":
            return _record_view(snapshot, node, path_to_string((root,) + path))
        if kind == "leaf":
            return node
        marker = id(node)
        if marker in stack:
            raise ValueError("cyclic container graph is not supported")
        cached = containers.get(marker)
        if cached is not None:
            return cached
        stack.add(marker)
        try:
            items = [(key, rebuild(child, path + (key,)))
                     for key, child in _child_items(node, kind)]
        finally:
            stack.discard(marker)
        context = None
        if kind == "pytree":
            context = _pytree_flatten(node)[1]
        template = _ContainerNode(
            obj_id=marker, kind=kind, klass=type(node), items=items, context=context)
        containers[marker] = template
        return template
    return rebuild(value, ())
def _is_scalar_state(value: Any) -> bool:
    torch = _require_torch()
    if isinstance(value, (bool, int, float, complex, str, bytes, type(None), enum.Enum)):
        return True
    if isinstance(value, (tuple, frozenset)):
        return all(_is_scalar_state(item) for item in value)
    return isinstance(value, (torch.dtype, torch.device, torch.Size, torch.layout))
def _module_attribute_items(module):
    """Yield every non-internal, non-module module attribute with its owner."""
    torch = _require_torch()
    for module_name, submodule in module.named_modules(remove_duplicate=False):
        prefix = f"{module_name}." if module_name else ""
        for name, value in vars(submodule).items():
            if name in _MODULE_INTERNAL_ATTRS or callable(value):
                continue
            if isinstance(value, (torch.nn.Module, torch.Generator)):
                continue
            yield prefix + name, submodule, name, value
def _scalar_values(module) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, _, _, value in _module_attribute_items(module):
        if _is_scalar_state(value):
            result[key] = value
    return result
def _capture_module(snapshot: StateSnapshot, module) -> dict[str, Any]:
    torch = _require_torch()
    params: dict[str, Any] = {}
    buffers: dict[str, Any] = {}
    nonpersistent: list[str] = []
    for module_name, submodule in module.named_modules(remove_duplicate=False):
        prefix = f"{module_name}." if module_name else ""
        for name, parameter in submodule._parameters.items():
            if parameter is None:
                continue
            path = f"module.parameters.{prefix}{name}"
            params[prefix + name] = _record_view(snapshot, parameter, path)
        for name, buffer in submodule._buffers.items():
            if buffer is None:
                continue
            path = f"module.buffers.{prefix}{name}"
            buffers[prefix + name] = _record_view(snapshot, buffer, path)
            if name in submodule._non_persistent_buffers_set:
                nonpersistent.append(path)
    attributes = {
        key: _capture_root(snapshot, f"module.attributes.{key}", value)
        for key, _, _, value in _module_attribute_items(module)
    }
    result = {
        "__kind__": "module",
        "module_name": type(module).__name__,
        "parameters": params,
        "buffers": buffers,
        "non_persistent_buffers": tuple(nonpersistent),
        "scalars": _scalar_values(module),
        "attributes": attributes,
    }
    if hasattr(module, "training"):
        result["training"] = bool(module.training)
    return result
def capture_state_snapshot(args=None, kwargs=None, outputs=None, module=None) -> StateSnapshot:
    """Capture shared storages and tensor views for the supplied state roots."""
    _require_torch()
    snapshot = StateSnapshot()
    if args is not None:
        snapshot.roots["args"] = _capture_root(snapshot, "args", args)
    if kwargs is not None:
        snapshot.roots["kwargs"] = _capture_root(snapshot, "kwargs", kwargs)
    if outputs is not None:
        snapshot.roots["outputs"] = _capture_root(snapshot, "outputs", outputs)
    if module is not None:
        snapshot.roots["module"] = _capture_module(snapshot, module)
    return snapshot
def _item_matches(template, live, kind) -> bool:
    live_items = _child_items(live, kind)
    if kind == "dict":
        template_map, live_map = dict(template.items), dict(live_items)
        if set(template_map) != set(live_map):
            return False
        return all(_structure_matches(value, live_map[key])
                   for key, value in template_map.items())
    if [key for key, _ in template.items] != [key for key, _ in live_items]:
        return False
    return all(_structure_matches(left, right)
               for (_, left), (_, right) in zip(template.items, live_items))
def _structure_matches(template, live) -> bool:
    """Compare container shape/keys while allowing scalar values to differ."""
    if isinstance(template, _TensorRef):
        return isinstance(live, _require_torch().Tensor)
    if isinstance(template, _ContainerNode):
        live_kind, _ = _classify(live)
        if live_kind != template.kind:
            return False
        if template.kind in {"dataclass", "object", "namedtuple", "pytree", "slots"}:
            if type(live) is not template.klass:
                return False
        return _item_matches(template, live, template.kind)
    template_kind, _ = _classify(template)
    live_kind, _ = _classify(live)
    if template_kind == "tensor":
        return live_kind == "tensor"
    if template_kind == "leaf":
        return live_kind == "leaf"
    if live_kind in {"tensor", "leaf"}:
        return False
    if template_kind != live_kind:
        return False
    if template_kind in {"dataclass", "object", "namedtuple", "pytree", "slots"}:
        if type(template) is not type(live):
            return False
    if template_kind == "dict":
        template_items = dict(_child_items(template, template_kind))
        live_items = dict(_child_items(live, live_kind))
        if set(template_items) != set(live_items):
            return False
        return all(_structure_matches(value, live_items[key])
                   for key, value in template_items.items())
    template_items = _child_items(template, template_kind)
    live_items = _child_items(live, live_kind)
    if [key for key, _ in template_items] != [key for key, _ in live_items]:
        return False
    return all(_structure_matches(left, right)
               for (_, left), (_, right) in zip(template_items, live_items))
def _walk_live_root(root: str, value) -> dict[str, Any]:
    result: dict[str, Any] = {}
    torch = _require_torch()
    for path, leaf in flatten_tree_with_paths(value, dedupe_containers=True):
        if isinstance(leaf, torch.Tensor):
            result[path_to_string((root,) + path)] = leaf
    return result
def _live_module_map(module) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for module_name, submodule in module.named_modules(remove_duplicate=False):
        prefix = f"{module_name}." if module_name else ""
        for name, parameter in submodule._parameters.items():
            if parameter is not None:
                result[f"module.parameters.{prefix}{name}"] = parameter
        for name, buffer in submodule._buffers.items():
            if buffer is not None:
                result[f"module.buffers.{prefix}{name}"] = buffer
    return result
def _validate_view(spec: TensorViewSpec, live) -> None:
    torch = _require_torch()
    if not isinstance(live, torch.Tensor):
        raise ValueError(f"{spec.tree_path} is no longer a tensor")
    _validate_tensor_layout(live)
    if _dtype_name(live.dtype) != spec.dtype_str:
        raise ValueError(f"{spec.tree_path} dtype changed")
    if tuple(int(item) for item in live.shape) != spec.shape:
        raise ValueError(f"{spec.tree_path} shape changed")
    if tuple(int(item) for item in live.stride()) != spec.stride:
        raise ValueError(f"{spec.tree_path} stride changed")
    if int(live.storage_offset()) != spec.storage_offset:
        raise ValueError(f"{spec.tree_path} storage offset changed")
    if str(live.device) != spec.device_str:
        raise ValueError(f"{spec.tree_path} device changed")
    if bool(live.is_conj()) != bool(spec.is_conj):
        raise ValueError(f"{spec.tree_path} conjugate flag changed")
    if bool(live.is_neg()) != bool(spec.is_neg):
        raise ValueError(f"{spec.tree_path} negative flag changed")
    if spec.npu_format is not None and _npu_format(live) != spec.npu_format:
        raise ValueError(f"{spec.tree_path} NPU format changed")
def _tree_root(path: str) -> str:
    """Return the root key of a rendered path (``args[0]`` -> ``args``)."""
    for index, char in enumerate(path):
        if char in ".[":
            return path[:index]
    return path
def _rebuild_container(node: _ContainerNode, items):
    values = [item for _, item in items]
    if node.kind == "namedtuple":
        return node.klass(*values)
    if node.kind == "tuple":
        try:
            return tuple.__new__(node.klass, values)
        except Exception:
            return tuple(values)
    if node.kind == "list":
        try:
            result = list.__new__(node.klass)
        except Exception:
            result = []
        result.extend(values)
        return result
    if node.kind == "pytree":
        return _pytree_unflatten_type(node.klass, values, node.context)
    if node.kind == "dict":
        try:
            result = node.klass()
        except Exception:
            result = {}
        try:
            result.update(items)
            return result
        except Exception:
            return dict(items)
    try:
        result = object.__new__(node.klass)
    except Exception:
        result = None
    if result is None:
        return _rebuild(type("Container", (), {})(), node.kind, items)
    for name, item in items:
        object.__setattr__(result, name, item)
    return result
def _copy_object_fields(live, rebuilt):
    """Copy rebuilt scalar fields onto a frozen/slotted live container."""
    if rebuilt is live:
        return live
    for name in getattr(live, "__dict__", ()):
        object.__setattr__(live, name, getattr(rebuilt, name))
    for name in _slot_names(type(live)):
        if hasattr(rebuilt, name):
            object.__setattr__(live, name, getattr(rebuilt, name))
    return live
def _restore_scalars_in_place(template, live):
    """Restore non-tensor leaves, returning rebuilt immutable roots."""
    if isinstance(template, _TensorRef):
        return live
    torch = _require_torch()
    if isinstance(template, _ContainerNode):
        live_kind, _ = _classify(live)
        if live_kind != template.kind:
            return live
        if template.kind in {"tuple", "namedtuple"}:
            live_items = _child_items(live, live_kind)
            restored = [
                _restore_scalars_in_place(child, right)
                for (_, child), (_, right) in zip(template.items, live_items)]
            if template.kind == "namedtuple":
                return template.klass(*restored)
            try:
                return tuple.__new__(template.klass, restored)
            except Exception:
                return tuple(restored)
        if template.kind == "pytree":
            try:
                live_children = _pytree_flatten(live)[0]
            except (TypeError, KeyError, RuntimeError):
                return live
            restored = [
                _restore_scalars_in_place(child, right)
                for (_, child), right in zip(template.items, live_children)]
            return _copy_object_fields(
                live,
                _pytree_unflatten_type(type(live), restored, template.context))
        if template.kind == "dict":
            for key, child in template.items:
                if key in live:
                    live[key] = _restore_scalars_in_place(child, live[key])
            return live
        if template.kind == "list":
            for key, child in template.items:
                if key < len(live):
                    live[key] = _restore_scalars_in_place(child, live[key])
            return live
        for key, child in template.items:
            if hasattr(live, key):
                object.__setattr__(
                    live, key, _restore_scalars_in_place(child, getattr(live, key)))
        return live
    kind, _ = _classify(template)
    if kind == "leaf":
        return live if isinstance(live, torch.Tensor) else template
    live_kind, _ = _classify(live)
    if live_kind != kind:
        return live
    items = _child_items(template, kind)
    if kind == "dict":
        for key, child in items:
            if key in live:
                live[key] = _restore_scalars_in_place(child, live[key])
        return live
    if kind == "list":
        for key, child in items:
            if key < len(live):
                live[key] = _restore_scalars_in_place(child, live[key])
        return live
    if kind in {"tuple", "namedtuple"}:
        live_items = _child_items(live, live_kind)
        restored = [
            _restore_scalars_in_place(child, right)
            for (_, child), (_, right) in zip(items, live_items)]
        if kind == "namedtuple":
            return type(template)(*restored)
        return tuple(restored)
    if kind == "pytree":
        try:
            live_children = _pytree_flatten(live)[0]
            context = _pytree_flatten(template)[1]
        except (TypeError, KeyError, RuntimeError):
            return live
        restored = [
            _restore_scalars_in_place(child, right)
            for (_, child), right in zip(items, live_children)]
        return _copy_object_fields(
            live, _pytree_unflatten_type(type(live), restored, context))
    for key, child in items:
        if hasattr(live, key):
            object.__setattr__(
                live, key, _restore_scalars_in_place(child, getattr(live, key)))
    return live
def _restore_module_attributes(snapshot: StateSnapshot, module) -> int:
    state = snapshot.roots.get("module") or {}
    attributes = state.get("attributes")
    if attributes is None:
        attributes = state.get("scalars")
    if not attributes:
        return 0
    restored = 0
    for key, submodule, name, value in _module_attribute_items(module):
        template = attributes.get(key)
        if template is None:
            continue
        result = _restore_scalars_in_place(template, value)
        if result is not value:
            setattr(submodule, name, result)
        restored += 1
    return restored
def restore_state_snapshot(snapshot: StateSnapshot, args=None, kwargs=None,
                           outputs=None, module=None, *, restore_module_scalars=True
                           ) -> dict[str, int]:
    """Restore saved bytes in place, once per unique storage id."""
    torch = _require_torch()
    if not isinstance(snapshot, StateSnapshot):
        raise ValueError("snapshot must be a StateSnapshot")
    provided = {"args": args, "kwargs": kwargs, "outputs": outputs, "module": module}
    if snapshot.roots and not any(value is not None for value in provided.values()):
        raise ValueError("no live state roots provided to restore")
    live: dict[str, Any] = {}
    scalar_restored = 0
    for root, value in provided.items():
        if value is None:
            continue
        if root not in snapshot.roots:
            raise ValueError(f"snapshot has no {root} root")
        if root == "module":
            if restore_module_scalars:
                scalar_restored += _restore_module_attributes(snapshot, value)
            live.update(_live_module_map(value))
            attributes = (snapshot.roots.get("module") or {}).get("attributes") or {}
            for key, _, _, attribute in _module_attribute_items(value):
                if key in attributes:
                    live.update(_walk_live_root(f"module.attributes.{key}", attribute))
        else:
            if not _structure_matches(snapshot.roots[root], value):
                raise ValueError(f"{root} structure differs from the snapshot")
            _restore_scalars_in_place(snapshot.roots[root], value)
            live.update(_walk_live_root(root, value))
    expected_paths = {
        view.tree_path for view in snapshot.views
        if _tree_root(view.tree_path) in {
            root for root, value in provided.items() if value is not None
        }
    }
    missing = expected_paths - set(live)
    if missing:
        raise ValueError(f"live state is missing tensor path(s): {sorted(missing)[:4]}")
    extra = set(live) - {view.tree_path for view in snapshot.views}
    if extra:
        raise ValueError(f"live state has untracked tensor path(s): {sorted(extra)[:4]}")
    by_path = {view.tree_path: view for view in snapshot.views}
    live_storages: dict[str, Any] = {}
    live_owner: dict[int, str] = {}
    for path, tensor in live.items():
        spec = by_path.get(path)
        if spec is None:
            raise ValueError(f"untracked tensor path {path}")
        if spec.storage_id not in snapshot.storages:
            raise ValueError(f"snapshot storage {spec.storage_id} is missing")
        _validate_view(spec, tensor)
        saved = snapshot.storages[spec.storage_id]
        storage = tensor.untyped_storage()
        if int(storage.nbytes()) != saved.nbytes:
            raise ValueError(f"{path} storage size {storage.nbytes()} != {saved.nbytes}")
        cdata = int(storage._cdata)
        previous = live_storages.get(spec.storage_id)
        if previous is not None and int(previous._cdata) != cdata:
            raise ValueError(
                f"multiple live storages claim snapshot storage {spec.storage_id}")
        owner = live_owner.get(cdata)
        if owner is not None and owner != spec.storage_id:
            raise ValueError(
                f"live storage aliases snapshot storages {owner} and {spec.storage_id}")
        live_owner[cdata] = spec.storage_id
        live_storages[spec.storage_id] = storage
    restored = 0
    with torch.no_grad():
        for storage_id, storage in live_storages.items():
            saved = snapshot.storages[storage_id]
            target = torch.empty(0, dtype=torch.uint8, device=storage.device)
            target.set_(storage, 0, (saved.nbytes,), (1,))
            source = saved.raw_bytes
            if str(source.device) != str(target.device):
                source = source.to(target.device)
            target.copy_(source)
            restored += 1
    return {"storages_restored": restored, "tensors_checked": len(live),
            "module_scalars_restored": scalar_restored}
# ---------------------------------------------------------------------------
# Reconstruction
def _allocate_storages(snapshot: StateSnapshot, device=None, overrides=None):
    torch = _require_torch()
    overrides = overrides or {}
    storages = {}
    used = {view.storage_id for view in snapshot.views}
    for storage_id in used:
        node = snapshot.storages.get(storage_id)
        if node is None:
            raise ValueError(f"snapshot storage {storage_id} is missing")
        if storage_id in overrides:
            storage = overrides[storage_id]
            if int(storage.nbytes()) != node.nbytes:
                raise ValueError(f"override storage {storage_id} has incompatible size")
            storages[storage_id] = storage
            continue
        target_device = device if device is not None else node.device_str
        raw = torch.empty(int(node.nbytes), dtype=torch.uint8, device=target_device)
        source = node.raw_bytes
        if str(source.device) != str(raw.device):
            source = source.to(raw.device)
        if node.nbytes:
            raw.copy_(source)
        storages[storage_id] = raw.untyped_storage()
    return storages
def _tensor_view_signature(tensor):
    return (
        int(tensor.untyped_storage()._cdata), _dtype_name(tensor.dtype),
        tuple(int(item) for item in tensor.shape),
        tuple(int(item) for item in tensor.stride()),
        int(tensor.storage_offset()), bool(tensor.requires_grad),
        bool(tensor.is_conj()), bool(tensor.is_neg()),
    )
def snapshot_tensor_map(snapshot: StateSnapshot, device=None, storage_overrides=None) -> dict[str, Any]:
    """Materialize every captured tensor view, sharing storages and identities."""
    torch = _require_torch()
    storages = _allocate_storages(snapshot, device=device, overrides=storage_overrides)
    result: dict[str, Any] = {}
    cache: dict[tuple, Any] = {}
    for view in snapshot.views:
        if view.npu_format is not None and int(view.npu_format) not in _SUPPORTED_NPU_FORMATS:
            raise ValueError(
                f"cannot reconstruct opaque NPU format {view.npu_format} for {view.tree_path}")
        key = (view.tensor_id, view.storage_id, view.dtype_str, view.shape, view.stride,
               view.storage_offset, view.requires_grad, view.device_str, view.npu_format,
               view.is_conj, view.is_neg)
        if view.tensor_id is not None and key in cache:
            result[view.tree_path] = cache[key]
            continue
        dtype = _dtype_from_name(view.dtype_str)
        tensor = torch.empty(0, dtype=dtype, device=device or view.device_str)
        tensor.set_(storages[view.storage_id], view.storage_offset, view.shape, view.stride)
        if view.requires_grad and (dtype.is_floating_point or dtype.is_complex):
            tensor.requires_grad_(True)
        if view.is_conj:
            tensor = tensor.conj()
        if view.is_neg:
            tensor = tensor._neg_view()
        if view.tensor_id is not None:
            cache[key] = tensor
        result[view.tree_path] = tensor
    return result
def _materialize_template(template, tensors, cache=None, overrides=None,
                          container_memo=None):
    if isinstance(template, _TensorRef):
        if overrides is not None and template.tree_path in overrides:
            tensor = overrides[template.tree_path]
        else:
            try:
                tensor = tensors[template.tree_path]
            except KeyError as exc:
                raise ValueError(
                    f"missing reconstructed tensor {template.tree_path}") from exc
        if cache is not None and template.tensor_id is not None:
            key = (template.tensor_id, _tensor_view_signature(tensor))
            cached = cache.get(key)
            if cached is not None:
                return cached
            cache[key] = tensor
        return tensor
    if cache is None:
        cache = {}
    if isinstance(template, dict) and template.get("__kind__") == "module":
        return {
            "__kind__": "module",
            "module_name": template.get("module_name"),
            "parameters": {
                name: _materialize_template(value, tensors, cache, overrides, container_memo)
                for name, value in template.get("parameters", {}).items()
            },
            "buffers": {
                name: _materialize_template(value, tensors, cache, overrides, container_memo)
                for name, value in template.get("buffers", {}).items()
            },
            "non_persistent_buffers": template.get("non_persistent_buffers", ()),
            "training": template.get("training"),
            "scalars": template.get("scalars", {}),
            "attributes": {
                name: _materialize_template(
                    value, tensors, cache, overrides, container_memo)
                for name, value in template.get("attributes", {}).items()
            },
        }
    if isinstance(template, _ContainerNode):
        if container_memo is not None and template.obj_id in container_memo:
            return container_memo[template.obj_id]
        items = [(key, _materialize_template(
            child, tensors, cache, overrides, container_memo))
            for key, child in template.items]
        result = _rebuild_container(template, items)
        if container_memo is not None:
            container_memo[template.obj_id] = result
        return result
    kind, _ = _classify(template)
    if kind in {"leaf", "tensor"}:
        return template
    items = [(key, _materialize_template(child, tensors, cache, overrides, container_memo))
             for key, child in _child_items(template, kind)]
    return _rebuild(template, kind, items)
def reconstruct_from_snapshot(snapshot: StateSnapshot, device=None, storage_overrides=None,
                              tensor_overrides=None) -> dict[str, Any]:
    """Rebuild captured roots on fresh storages, preserving all view aliases."""
    if not isinstance(snapshot, StateSnapshot):
        raise ValueError("snapshot must be a StateSnapshot")
    tensors = snapshot_tensor_map(snapshot, device=device, storage_overrides=storage_overrides)
    cache: dict[tuple, Any] = {}
    container_memo: dict[int, Any] = {}
    return {
        root: _materialize_template(
            template, tensors, cache, tensor_overrides, container_memo)
        for root, template in snapshot.roots.items()
    }
