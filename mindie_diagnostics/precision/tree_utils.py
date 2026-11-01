"""Generic PyTree/container walking and rebuilding helpers.

[Role]
  Classify arbitrary call-state values, enumerate their child items and rebuild
  same-shape containers.  Shared by the storage graph and its callers so the
  snapshot foundation stays within its module budget.

[Invariants]
  * ``torch`` is imported lazily; package import stays zero-dependency.
  * Cyclic containers are rejected, never recursed infinitely.
  * Rebuilding never runs a user ``__init__`` for plain object/slots nodes.

[Timeouts & Locks]
  Pure local in-memory traversal.  No sleeps, subprocesses or locks.

[Side-Effect Policy]
  No writes outside newly allocated containers.
"""
from __future__ import annotations
import re
from typing import Any
_TORCH = None
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
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
def _is_namedtuple(value: Any) -> bool:
    return isinstance(value, tuple) and hasattr(value, "_fields")
_PYTREE_NODES = None
def _pytree_nodes():
    global _PYTREE_NODES
    if _PYTREE_NODES is None:
        from torch.utils import _pytree  # noqa: WPS433 - optional, lazy
        _PYTREE_NODES = _pytree.SUPPORTED_NODES
    return _PYTREE_NODES
def _pytree_flatten(value):
    node = _pytree_nodes()[type(value)]
    flatten = getattr(node, "flatten_fn", None) or getattr(node, "flatten_func", None)
    if flatten is None:
        raise TypeError(f"registered PyTree {type(value).__name__} has no flatten function")
    return flatten(value)
def _pytree_unflatten(value, children, context):
    node = _pytree_nodes()[type(value)]
    unflatten = getattr(node, "unflatten_fn", None) or getattr(node, "unflatten_func", None)
    if unflatten is None:
        raise TypeError(f"registered PyTree {type(value).__name__} has no unflatten function")
    return unflatten(children, context)
def _pytree_unflatten_type(cls, children, context):
    node = _pytree_nodes()[cls]
    unflatten = getattr(node, "unflatten_fn", None) or getattr(node, "unflatten_func", None)
    if unflatten is None:
        raise TypeError(f"registered PyTree {cls.__name__} has no unflatten function")
    return unflatten(children, context)
def _slot_names(value_type) -> tuple[str, ...]:
    names: list[str] = []
    for base in getattr(value_type, "__mro__", (value_type,)):
        slots = getattr(base, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name not in {"__dict__", "__weakref__"} and not name.startswith("__"):
                if name not in names:
                    names.append(name)
    return tuple(names)
def _is_slots_object(value: Any) -> bool:
    value_type = type(value)
    if value_type.__module__ in {"builtins", "torch._C"}:
        return False
    return bool(_slot_names(value_type)) and not hasattr(value, "__dict__")
def _path_text(path: tuple[Any, ...]) -> str:
    out = ""
    for part in path:
        if isinstance(part, int) and not isinstance(part, bool):
            out += f"[{part}]"
        elif isinstance(part, str):
            out += ("." if out else "") + part if _NAME_RE.match(part) else f"[{part!r}]"
        else:
            out += f"[{part!r}]"
    return out
def path_to_string(path: tuple[Any, ...]) -> str:
    """Render a structural path tuple as a stable, readable tensor path."""
    return _path_text(tuple(path))
def _classify(value: Any):
    torch = _require_torch()
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return "leaf", None
    if isinstance(value, torch.Tensor):
        return "tensor", None
    if _is_namedtuple(value):
        return "namedtuple", None
    if isinstance(value, tuple):
        return "tuple", None
    if isinstance(value, list):
        return "list", None
    if isinstance(value, dict):
        return "dict", None
    value_type = type(value)
    if value_type in _pytree_nodes() and value_type not in (tuple, list, dict):
        return "pytree", None
    import dataclasses
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return "dataclass", None
    if isinstance(value, torch.nn.Module):
        # Modules are captured explicitly by capture_state_snapshot(); avoid
        # recursively walking hooks, autograd state and compiled internals.
        return "leaf", None
    if _is_slots_object(value):
        return "slots", None
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return "object", None
    return "leaf", None
def _child_items(value: Any, kind: str):
    if kind == "namedtuple":
        return [(name, getattr(value, name)) for name in value._fields]
    if kind in {"tuple", "list"}:
        return list(enumerate(value))
    if kind == "dict":
        return list(value.items())
    if kind == "dataclass":
        import dataclasses
        return [(item.name, getattr(value, item.name)) for item in dataclasses.fields(value)]
    if kind == "pytree":
        return list(enumerate(_pytree_flatten(value)[0]))
    if kind == "slots":
        return [(name, getattr(value, name)) for name in _slot_names(type(value))
                if hasattr(value, name)]
    if kind == "object":
        return list(vars(value).items())
    return []
def _new_container(value: Any, kind: str):
    cls = type(value)
    if kind == "dict":
        try:
            return cls()
        except Exception:
            return {}
    if kind == "list":
        try:
            return list.__new__(cls)
        except Exception:
            return []
    if kind == "tuple":
        try:
            return tuple.__new__(cls, ())
        except Exception:
            return ()
    if kind == "namedtuple":
        return None
    if kind == "pytree":
        return None
    try:
        return object.__new__(cls)
    except Exception:
        return None
def _rebuild(value: Any, kind: str, items: list[tuple[Any, Any]]) -> Any:
    if kind == "namedtuple":
        return type(value)(*[item for _, item in items])
    if kind == "tuple":
        values = [item for _, item in items]
        try:
            return tuple.__new__(type(value), values)
        except Exception:
            return tuple(values)
    if kind == "list":
        result = _new_container(value, kind)
        result.extend(item for _, item in items)
        return result
    if kind == "dict":
        result = _new_container(value, kind)
        try:
            result.update(items)
            return result
        except Exception:
            result = dict(items)
            return result
    if kind == "pytree":
        try:
            _, context = _pytree_flatten(value)
            return _pytree_unflatten(
                value, [item for _, item in items], context)
        except Exception as exc:
            raise TypeError(
                f"cannot rebuild registered PyTree {type(value).__name__}") from exc
    result = _new_container(value, kind)
    if result is None:
        import copy
        try:
            result = copy.copy(value)
        except Exception as exc:
            raise TypeError(f"cannot safely copy container {type(value).__name__}") from exc
    try:
        for name, item in items:
            object.__setattr__(result, name, item)
    except Exception as exc:
        raise TypeError(f"cannot restore slots of {type(value).__name__}") from exc
    return result
def walk_tree(value: Any, visit=None, dedupe_containers=False):
    """Yield ``(path_tuple, leaf)`` for every tensor or primitive leaf.
    Container nodes are traversed (tuple, namedtuple, list, dict, dataclass,
    custom ``__dict__`` object, and registered container adapters).  When
    ``visit`` is supplied the traversal is consumed and ``visit(path, leaf)``
    is called for each leaf; in that form the return value is ``None``.
    ``dedupe_containers`` yields only the first path of an aliased container,
    matching capture's container-identity deduplication.
    """
    _require_torch()
    seen: set[int] = set()
    def descend(node, path, stack):
        kind, _ = _classify(node)
        if kind in {"leaf", "tensor"}:
            yield tuple(path), node
            return
        marker = id(node)
        if marker in stack:
            raise ValueError("cyclic container graph is not supported")
        if dedupe_containers and marker in seen:
            return
        seen.add(marker)
        stack.add(marker)
        try:
            for key, child in _child_items(node, kind):
                yield from descend(child, path + (key,), stack)
        finally:
            stack.discard(marker)
    if visit is None:
        return descend(value, (), set())
    for path, leaf in descend(value, (), set()):
        visit(path, leaf)
    return None
def flatten_tree_with_paths(value: Any, dedupe_containers=False
                            ) -> list[tuple[tuple[Any, ...], Any]]:
    """Return every leaf with its structural path, in deterministic order."""
    return list(walk_tree(value, dedupe_containers=dedupe_containers))
def map_tree(value: Any, fn):
    """Return a same-shape container with ``fn(leaf)`` applied to every leaf."""
    _require_torch()
    def rebuild(node, stack):
        kind, _ = _classify(node)
        if kind in {"leaf", "tensor"}:
            return fn(node)
        marker = id(node)
        if marker in stack:
            raise ValueError("cyclic container graph is not supported")
        stack.add(marker)
        try:
            items = [(key, rebuild(child, stack)) for key, child in _child_items(node, kind)]
        finally:
            stack.discard(marker)
        return _rebuild(node, kind, items)
    return rebuild(value, set())
