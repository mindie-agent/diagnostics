"""Universal module-level lockstep differential tracer with optional healing.

[Role]
  Primitive 2 (``lockstep-diff``): hook every canonical ``nn.Module`` with
  kwargs-aware pre/post hooks, capture complete pre/post call storage states,
  align invocations by ``(canonical_module_id, invocation_ordinal, rank)``,
  compare tensor-tree paths against one or more baseline traces, and heal a
  non-overlapping module frontier back to baseline post-call state.

[Invariants]
  * Hooks use ``with_kwargs=True``; duplicate module aliases map to one
    canonical identity, while invocation ordinals distinguish repeated calls.
  * A baseline snapshot is immutable.  Healing restores baseline *post-call*
    bytes in place (inputs, kwargs, parameters, persistent and non-persistent
    buffers) and replaces newly allocated outputs with baseline-shaped views.
  * The healing frontier is strictly non-overlapping, so a child and its
    ancestor are never healed in the same trace.
  * Alignment stops at the first invocation-sequence divergence and reports
    ``status='control_flow_diverged'`` rather than fabricating comparisons.
  * Metrics include max-abs, relative-L2, ULP, nonfinite mismatch, sample
    count, jitter quantiles, and SJR; jitter requires at least three baselines.

[Timeouts & Locks]
  Synchronous tracing in the caller's process.  No sleeps, subprocesses,
  network, or cross-thread locks.  Snapshot capture synchronizes device bytes;
  this is intentionally outside the zero-sync watch primitive.

[Side-Effect Policy]
  Registers module hooks; removes them via ``close()``.  Healing mutates live
  storages and returns replacement outputs.  No files, network or env writes.
"""
from __future__ import annotations
import contextlib
from dataclasses import dataclass, field
from typing import Any
from .storage_graph import (
    _live_module_map, _require_torch, _restore_scalars_in_place, _scalar_values,
    _storage_key, _structure_matches, _tree_root, _walk_live_root,
    capture_state_snapshot, flatten_tree_with_paths, path_to_string,
    reconstruct_from_snapshot, restore_state_snapshot, snapshot_tensor_map,
)
from .metrics import (
    _aggregate_metrics, _jitter_delta, _missing_metric, _ordered_bits,
    _tensor_metrics, _ulp_floor, _ulp_max,
)
from .tree_utils import map_tree
def _evict_output_tensors(value):
    torch = _require_torch()
    return map_tree(
        value,
        lambda leaf: torch.empty((0,), dtype=leaf.dtype, device=leaf.device)
        if isinstance(leaf, torch.Tensor) else leaf)
def _flatten_tensors(value):
    torch = _require_torch()
    return [leaf for _, leaf in flatten_tree_with_paths(value)
            if isinstance(leaf, torch.Tensor)]
def _tensor_bytes(values):
    total = 0
    for tensor in values.values():
        try:
            total += int(tensor.numel()) * int(tensor.element_size())
        except (AttributeError, RuntimeError):
            continue
    return total
def _is_ancestor(ancestor: str, descendant: str) -> bool:
    if ancestor == descendant:
        return False
    if ancestor == "":
        return True
    return descendant.startswith(ancestor + ".")
def _assert_inference_mode(module, *values):
    torch = _require_torch()
    if not torch.is_grad_enabled():
        return
    for value in values:
        if value is None:
            continue
        for tensor in _flatten_tensors(value):
            if tensor.requires_grad:
                raise RuntimeError(
                    "LockstepDiffer(heal=True) requires inference/no_grad mode; "
                    "autograd is active")
    if module is not None:
        for parameter in module.parameters():
            if parameter.requires_grad:
                raise RuntimeError(
                    "LockstepDiffer(heal=True) requires inference/no_grad mode; "
                    "autograd is active")
def resolve_healing_frontier(module_paths, mode="leaf") -> set[str]:
    """Resolve candidate paths into a strict ancestor/descendant-free set."""
    if mode not in {"leaf", "top"}:
        raise ValueError("mode must be 'leaf' or 'top'")
    if isinstance(module_paths, str):
        module_paths = [module_paths]
    candidates = {str(path).strip().rstrip(".") for path in module_paths}
    if mode == "leaf":
        return {
            path for path in candidates
            if not any(_is_ancestor(path, other) for other in candidates if other != path)
        }
    return {
        path for path in candidates
        if not any(_is_ancestor(other, path) for other in candidates if other != path)
    }
@dataclass
class InvocationRecord:
    key: tuple[str, int, int]
    module_path: str
    invocation_ordinal: int
    rank: int
    aliases: tuple[str, ...]
    pre: Any
    post: Any
    output: Any
    metrics: list[dict[str, Any]] = field(default_factory=list)
    healed: bool = False
    heal_status: str | None = None
    heal_error: str | None = None
@dataclass
class Trace:
    name: str
    kind: str
    records: list[InvocationRecord] = field(default_factory=list)
    order: list[tuple[str, int, int]] = field(default_factory=list)
    result: Any = None
    error: BaseException | None = None
    status: str = "ok"
    divergence: dict[str, Any] | None = None
    def by_key(self) -> dict[tuple[str, int, int], InvocationRecord]:
        return {record.key: record for record in self.records}
class LockstepDiffer:
    """Hook, trace, compare and heal a module tree in lockstep."""
    def __init__(self, model, *, rank=0, heal=False, heal_after_compare=None,
                 frontier=None, healing_frontier=None, frontier_mode="leaf",
                 max_records=None, jitter_ulp_floor=None,
                 max_retained_bytes=256 * 1024 * 1024):
        torch = _require_torch()
        if not isinstance(model, torch.nn.Module):
            raise ValueError("model must be a torch.nn.Module")
        self.model = model
        if rank is None:
            distributed = getattr(torch, "distributed", None)
            try:
                rank = (distributed.get_rank()
                        if distributed is not None and distributed.is_initialized() else 0)
            except (RuntimeError, OSError):
                rank = 0
        self.rank = int(rank)
        self.heal = bool(heal if heal_after_compare is None else heal_after_compare)
        self.frontier_mode = frontier_mode
        self.max_records = int(max_records) if max_records is not None else None
        self.jitter_ulp_floor = jitter_ulp_floor
        self.max_retained_bytes = max(0, int(max_retained_bytes))
        self._canonical: dict[int, str] = {}
        self._aliases: dict[int, list[str]] = {}
        self._alias_to_canonical: dict[str, str] = {}
        self._modules: list[Any] = []
        self._modules_by_id: dict[int, Any] = {}
        self._path_modules: dict[str, int] = {}
        self._build_module_map()
        candidates = frontier if frontier is not None else healing_frontier
        if candidates is None:
            candidates = list(self._canonical.values())
        elif isinstance(candidates, str):
            candidates = [candidates]
        candidates = [self._canonicalize_candidate(path) for path in candidates]
        self.frontier = self._resolve_frontier(candidates, mode=frontier_mode)
        self.healing_frontier = self.frontier
        self._initial_model_state = None
        self.baselines: dict[str, Trace] = {}
        self.baseline_traces: list[Trace] = []
        self.target: Trace | None = None
        self.reference_name: str | None = None
        self._active: Trace | None = None
        self._ordinals: dict[int, int] = {}
        self._pending: dict[int, list[dict[str, Any]]] = {}
        self._expected_index = 0
        self._baseline_order: list[tuple[str, int, int]] = []
        self._baseline_records: dict[tuple[str, int, int], InvocationRecord] = {}
        self._values_cache: dict[int, dict[str, Any]] = {}
        self._values_cache_bytes = 0
        self._cache_order: list[int] = []
        self._jitter_elementwise: dict[tuple[Any, str], Any] = {}
        self._jitter_summary: dict[tuple[Any, str], tuple[float, float]] = {}
        self._volatile_scalars: dict[str, Any] = {}
        self._volatile_tracked = False
        self._active_pre_scalars: dict[str, Any] | None = None
        self._hooks: list[Any] = []
        self._report: dict[str, Any] | None = None
        self._install_hooks()
    # -- module identity ---------------------------------------------------
    def _build_module_map(self):
        torch = _require_torch()
        seen: set[int] = set()
        for name, module in self.model.named_modules(remove_duplicate=True):
            self._canonical[id(module)] = name
            if id(module) not in seen:
                seen.add(id(module))
                self._modules.append(module)
                self._modules_by_id[id(module)] = module
        for name, module in self.model.named_modules(remove_duplicate=False):
            self._aliases.setdefault(id(module), []).append(name)
        for module_id, aliases in self._aliases.items():
            canonical = self._canonical[module_id]
            self._path_modules[canonical] = module_id
            for alias in aliases:
                self._alias_to_canonical[alias] = canonical
                self._path_modules[alias] = module_id
    def _module_id_for_path(self, path):
        module_id = self._path_modules.get(path)
        if module_id is not None:
            return module_id
        best_alias = None
        for alias in self._path_modules:
            if path.startswith(alias + ".") and (best_alias is None or len(alias) > len(best_alias)):
                best_alias = alias
        return self._path_modules.get(best_alias) if best_alias is not None else None
    def _instance_is_ancestor(self, left, right):
        if left == right:
            return False
        left_id = self._module_id_for_path(left)
        right_id = self._module_id_for_path(right)
        if left_id is not None and left_id == right_id:
            return False
        if left_id is not None and right_id is not None:
            left_module = self._modules_by_id.get(left_id)
            right_module = self._modules_by_id.get(right_id)
            if (left_module is not None and right_module is not None
                    and any(item is right_module for item in left_module.modules())):
                return True
        left_aliases = self._aliases.get(left_id, [left])
        right_aliases = self._aliases.get(right_id, [right])
        return any(_is_ancestor(alias, other)
                   for alias in left_aliases for other in right_aliases)
    def _resolve_frontier(self, candidates, mode="leaf"):
        if mode not in {"leaf", "top"}:
            raise ValueError("mode must be 'leaf' or 'top'")
        candidates = {str(path).strip().rstrip(".") for path in candidates}
        if mode == "leaf":
            return {p for p in candidates if not any(self._instance_is_ancestor(p, o) for o in candidates if o != p)}
        return {p for p in candidates if not any(self._instance_is_ancestor(o, p) for o in candidates if o != p)}
    def _canonicalize_candidate(self, path):
        path = str(path).strip().rstrip(".")
        best_alias = None
        for alias in self._alias_to_canonical:
            if path == alias or path.startswith(alias + "."):
                if best_alias is None or len(alias) > len(best_alias):
                    best_alias = alias
        if best_alias is None:
            return path
        canonical = self._alias_to_canonical[best_alias]
        return canonical + path[len(best_alias):]
    def _install_hooks(self):
        for module in self._modules:
            pre = module.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
            post = module.register_forward_hook(self._post_hook, with_kwargs=True)
            self._hooks.extend((pre, post))
    def close(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.close()
    # -- trace lifecycle ---------------------------------------------------
    def begin_baseline(self, name=None, reset_model_state=True):
        if self._active is not None:
            raise RuntimeError("a trace is already active")
        if self._initial_model_state is None:
            self._initial_model_state = capture_state_snapshot(module=self.model)
        elif reset_model_state:
            restore_state_snapshot(
                self._initial_model_state, module=self.model,
                restore_module_scalars=False)
            self._restore_volatile_scalars()
        self._active_pre_scalars = _scalar_values(self.model)
        name = name or f"A{len(self.baselines) + 1}"
        trace = Trace(name=name, kind="baseline")
        self._active = trace
        self._ordinals = {}
        self._pending = {}
        self._expected_index = 0
        if self.reference_name is not None and self.reference_name in self.baselines:
            reference = self.baselines[self.reference_name]
            self._baseline_order = list(reference.order)
            self._baseline_records = reference.by_key()
        else:
            self._baseline_order = []
            self._baseline_records = {}
        return trace
    def end_baseline(self):
        trace = self._active
        if trace is None or trace.kind != "baseline":
            raise RuntimeError("no baseline trace is active")
        if trace.status == "ok" and trace.error is not None:
            trace.status = "error"
        self.baselines[trace.name] = trace
        self.baseline_traces.append(trace)
        if self.reference_name is None:
            self.reference_name = trace.name
        if not self._volatile_tracked and self._active_pre_scalars is not None:
            after = _scalar_values(self.model)
            for key, value in self._active_pre_scalars.items():
                if key not in after or after[key] != value:
                    self._volatile_scalars[key] = value
            self._volatile_tracked = True
        self._active_pre_scalars = None
        self._active = None
        self._clear_pending()
        if trace.name != self.reference_name:
            for record in trace.records:
                self._record_baseline_jitter(record)
            self._enforce_snapshot_retention()
        return trace
    def begin_target(self, name="B", reset_model_state=True):
        if self._active is not None:
            raise RuntimeError("a trace is already active")
        if not self.baselines:
            raise RuntimeError("at least one baseline trace is required")
        if self._initial_model_state is not None and reset_model_state:
            restore_state_snapshot(
                self._initial_model_state, module=self.model,
                restore_module_scalars=False)
            self._restore_volatile_scalars()
        reference = self.baselines[self.reference_name]
        trace = Trace(name=name, kind="target")
        self._active = trace
        self._ordinals = {}
        self._pending = {}
        self._expected_index = 0
        self._baseline_order = list(reference.order)
        self._baseline_records = reference.by_key()
        return trace
    def end_target(self):
        trace = self._active
        if trace is None or trace.kind != "target":
            raise RuntimeError("no target trace is active")
        if trace.error is not None and trace.status == "ok":
            trace.status = "error"
        if trace.status == "ok" and self._expected_index != len(self._baseline_order):
            trace.status = "control_flow_diverged"
            trace.divergence = {"reason": "target_invocation_count_mismatch",
                                "expected": len(self._baseline_order),
                                "observed": self._expected_index}
        self.target = trace
        self._active = None
        self._clear_pending()
        self._report = self._build_report()
        self._release_caches()
        for trace in self.baseline_traces + [self.target]:
            for record in trace.records:
                record.pre = None
        self._enforce_snapshot_retention(include_target=True)
        return self._report
    def _restore_volatile_scalars(self):
        if not self._volatile_scalars:
            return
        for module_name, submodule in self.model.named_modules(remove_duplicate=False):
            prefix = f"{module_name}." if module_name else ""
            for name in list(self._volatile_scalars):
                if name.startswith(prefix) and name[len(prefix):] in vars(submodule):
                    setattr(submodule, name[len(prefix):],
                            self._volatile_scalars[name])
    def _release_caches(self):
        self._values_cache.clear()
        self._cache_order.clear()
        self._values_cache_bytes = 0
        self._jitter_elementwise.clear()
    def _enforce_retention(self):
        while (self._values_cache_bytes > self.max_retained_bytes
               and self._cache_order):
            record_id = self._cache_order.pop(0)
            evicted = self._values_cache.pop(record_id, None)
            if evicted:
                self._values_cache_bytes -= _tensor_bytes(evicted)
    def _retained_snapshot_bytes(self):
        traces = self.baseline_traces + ([self.target] if self.target else [])
        return sum(
            sum(int(node.nbytes) for node in record.post.storages.values())
            for trace in traces for record in trace.records
            if record.post is not None)
    def _retained_output_bytes(self):
        torch = _require_torch()
        traces = self.baseline_traces + ([self.target] if self.target else [])
        storages = {}
        for trace in traces:
            for root in [trace.result] + [r.output for r in trace.records]:
                for _, leaf in flatten_tree_with_paths(root, dedupe_containers=True):
                    if isinstance(leaf, torch.Tensor):
                        st = leaf.untyped_storage()
                        storages[(str(leaf.device), int(st.data_ptr()))] = int(st.nbytes())
        return sum(storages.values())
    def _evict_snapshot(self, record):
        record.pre = None
        if record.post is not None:
            record.post.storages.clear()
    def _enforce_snapshot_retention(self, include_target=False):
        traces = list(self.baseline_traces)
        if include_target and self.target is not None:
            traces.append(self.target)
        for trace in traces:
            if self._retained_snapshot_bytes() <= self.max_retained_bytes:
                break
            if not include_target and trace.name == self.reference_name:
                continue
            for record in trace.records:
                self._evict_snapshot(record)
        for trace in traces:
            if (self._retained_snapshot_bytes() + self._retained_output_bytes()
                    <= self.max_retained_bytes):
                return
            if not include_target and trace.name == self.reference_name:
                continue
            for record in trace.records:
                record.output = _evict_output_tensors(record.output)
            trace.result = _evict_output_tensors(trace.result)
    def _clear_pending(self):
        for stack in self._pending.values():
            stack.clear()
        self._pending = {}
    def run_baseline(self, fn, *args, name=None, reset_model_state=True, **kwargs):
        trace = self.begin_baseline(name, reset_model_state=reset_model_state)
        try:
            trace.result = fn(*args, **kwargs)
        except BaseException as exc:
            trace.error = exc
            trace.status = "error"
            raise
        finally:
            self.end_baseline()
        return trace
    def run_target(self, fn, *args, name="B", reset_model_state=True, **kwargs):
        self.begin_target(name, reset_model_state=reset_model_state)
        try:
            self._active.result = fn(*args, **kwargs)
        except BaseException as exc:
            self._active.error = exc
            self._active.status = "error"
            raise
        finally:
            report = self.end_target()
        return report
    def run_baselines(self, fn, count=1, *args, prefix="A", reset_model_state=True, **kwargs):
        traces = []
        args_snap = capture_state_snapshot(args=args, kwargs=kwargs)
        restore_args = args if "args" in args_snap.roots else None
        restore_kwargs = kwargs if "kwargs" in args_snap.roots else None
        for index in range(int(count)):
            restore_state_snapshot(
                args_snap, args=restore_args, kwargs=restore_kwargs)
            traces.append(self.run_baseline(
                fn, *args, name=f"{prefix}{index + 1}",
                reset_model_state=reset_model_state, **kwargs))
        return traces
    def start_baseline(self, name=None):
        return self.begin_baseline(name)
    def start_target(self, name="B"):
        return self.begin_target(name)
    def compare(self):
        report = self.report
        if self.target is not None:
            self._enforce_snapshot_retention(include_target=True)
        return report
    @contextlib.contextmanager
    def baseline(self, name=None, reset_model_state=True):
        self.begin_baseline(name, reset_model_state=reset_model_state)
        try:
            yield self._active
        finally:
            self.end_baseline()
    @contextlib.contextmanager
    def target_trace(self, name="B", reset_model_state=True):
        self.begin_target(name, reset_model_state=reset_model_state)
        try:
            yield self._active
        finally:
            self.end_target()
    def run(self, fn, *args, kind="baseline", name=None, reset_model_state=True, **kwargs):
        if kind == "baseline":
            return self.run_baseline(
                fn, *args, name=name, reset_model_state=reset_model_state, **kwargs)
        if kind == "target":
            return self.run_target(
                fn, *args, name=name or "B",
                reset_model_state=reset_model_state, **kwargs)
        raise ValueError("kind must be 'baseline' or 'target'")
    # -- hooks -------------------------------------------------------------
    def _pre_hook(self, module, args, kwargs):
        active = self._active
        if active is None:
            return None
        if self.heal:
            _assert_inference_mode(module, args, kwargs)
        if self.max_records is not None and len(active.records) >= self.max_records:
            active.status = "record_limit"
            return None
        module_id = id(module)
        ordinal = self._ordinals.get(module_id, 0) + 1
        self._ordinals[module_id] = ordinal
        canonical = self._canonical.get(module_id)
        if canonical is None:
            return None
        pre = capture_state_snapshot(args=args, kwargs=kwargs, module=module)
        self._pending.setdefault(module_id, []).append({
            "key": (canonical, ordinal, self.rank), "args": args,
            "kwargs": kwargs, "pre": pre})
        return None
    def _post_hook(self, module, args, kwargs, output):
        active = self._active
        if active is None:
            return None
        if self.heal:
            _assert_inference_mode(module, args, kwargs, output)
        stack = self._pending.get(id(module))
        if not stack:
            return None
        entry = stack.pop()
        post = capture_state_snapshot(args=args, kwargs=kwargs, outputs=output, module=module)
        record = InvocationRecord(
            key=entry["key"], module_path=entry["key"][0],
            invocation_ordinal=entry["key"][1], rank=entry["key"][2],
            aliases=tuple(self._aliases.get(id(module), ())),
            pre=entry["pre"], post=post, output=output)
        active.records.append(record)
        active.order.append(record.key)
        if active.kind == "baseline" and self.reference_name is None:
            return None
        if active.status != "ok":
            return None
        if (self._expected_index >= len(self._baseline_order)
                or record.key != self._baseline_order[self._expected_index]):
            active.status = "control_flow_diverged"
            expected = (self._baseline_order[self._expected_index]
                        if self._expected_index < len(self._baseline_order) else None)
            active.divergence = {
                "reason": "invocation_sequence_mismatch",
                "expected": expected,
                "observed": record.key,
                "ordinal": self._expected_index,
            }
            return None
        baseline_record = self._baseline_records.get(record.key)
        if baseline_record is None:
            active.status = "control_flow_diverged"
            active.divergence = {"reason": "missing_baseline_invocation", "observed": record.key}
            return None
        if active.kind == "target":
            record.metrics = self._compare_records(baseline_record, record)
        replacement = None
        replace_output = False
        if self.heal and record.module_path in self.frontier:
            replacement, replace_output = self._heal(
                baseline_record, record, args, kwargs, module, output, active)
            record.healed = replace_output or record.heal_status == "restored_in_place"
            if record.heal_status == "rejected_incompatible_output":
                active.status = "heal_rejected"
        self._expected_index += 1
        baseline_record.pre = None
        record.pre = None
        return replacement if replace_output else None
    # -- comparison --------------------------------------------------------
    def _values_for(self, record):
        cached = self._values_cache.get(id(record))
        if cached is None:
            torch = _require_torch()
            try:
                with torch.no_grad():
                    cached = snapshot_tensor_map(record.post, device="cpu")
            except (ValueError, KeyError, RuntimeError):
                return {}
            self._values_cache[id(record)] = cached
            self._cache_order.append(id(record))
            self._values_cache_bytes += _tensor_bytes(cached)
            self._enforce_retention()
        return cached
    def _jitter_for(self, key, path, reference):
        torch = _require_torch()
        values = []
        for name, trace in self.baselines.items():
            if name == self.reference_name:
                continue
            record = trace.by_key().get(key)
            if record is None:
                continue
            tensor = self._values_for(record).get(path)
            if tensor is not None and tuple(tensor.shape) == tuple(reference.shape):
                values.append(tensor)
        return values
    def _record_baseline_jitter(self, record):
        torch = _require_torch()
        ref_record = self._baseline_records.get(record.key)
        if ref_record is None:
            return
        ref_vals = self._values_for(ref_record)
        cur_vals = self._values_for(record)
        for path, ref_tensor in ref_vals.items():
            cur_tensor = cur_vals.get(path)
            if cur_tensor is None:
                continue
            delta = _jitter_delta(ref_tensor, cur_tensor)
            if delta is None or not delta.numel():
                continue
            slot = (record.key, path)
            prev = self._jitter_elementwise.get(slot)
            elem = delta if prev is None else torch.maximum(prev, delta)
            self._jitter_elementwise[slot] = elem
            self._jitter_summary[slot] = (
                float(elem.max().item()), float(elem.median().item()))
    def _compare_records(self, baseline_record, target_record):
        base = self._values_for(baseline_record)
        target = self._values_for(target_record)
        metrics = []
        for path in sorted(set(base) | set(target)):
            if path not in base:
                metrics.append(_missing_metric(path, "untracked_in_baseline", target[path]))
                continue
            if path not in target:
                metrics.append(_missing_metric(path, "missing_in_target", base[path]))
                continue
            summary = self._jitter_summary.get((baseline_record.key, path))
            jitter = ([] if summary is not None
                      else self._jitter_for(baseline_record.key, path, base[path]))
            metrics.append(_tensor_metrics(
                path, base[path], target[path], jitter, self.jitter_ulp_floor,
                jitter_summary=summary))
        return metrics
    # -- healing -----------------------------------------------------------
    def _heal(self, baseline_record, record, args, kwargs, module, target_output,
              active=None):
        torch = _require_torch()
        try:
            _assert_inference_mode(module, args, kwargs, target_output)
            with torch.no_grad():
                if not _outputs_compatible(baseline_record, target_output):
                    record.heal_status = "rejected_incompatible_output"
                    return None, False
                target_tensors = _flatten_tensors(target_output)
                target_pre = record.pre
                has_new = any(_storage_key(tensor.untyped_storage())
                              not in target_pre.storages for tensor in target_tensors)
                if "outputs" not in baseline_record.post.roots:
                    restore_state_snapshot(baseline_record.post, args=args,
                                           kwargs=kwargs, module=module)
                    if target_output is None:
                        record.heal_status = "restored_in_place"
                        return None, False
                    record.heal_status = "baseline_has_no_outputs"
                    return None, False
                if target_output is None or not target_tensors:
                    restore_state_snapshot(baseline_record.post, args=args,
                                           kwargs=kwargs, module=module)
                    overrides = self._baseline_storage_overrides(
                        baseline_record, args, kwargs, module)
                    rebuilt = reconstruct_from_snapshot(
                        baseline_record.post, storage_overrides=overrides,
                        tensor_overrides=self._output_identity_overrides(
                            self.model, baseline_record))
                    _bind_module_aliases(module, baseline_record, rebuilt.get("module"))
                    record.heal_status = "restored_and_output_rebuilt"
                    return rebuilt.get("outputs"), True
                if not has_new:
                    try:
                        restore_state_snapshot(baseline_record.post, args=args, kwargs=kwargs,
                                               outputs=target_output, module=module)
                        healed = _restore_scalars_in_place(
                            baseline_record.post.roots["outputs"], target_output)
                        record.heal_status = "restored_in_place"
                        return (None, False) if healed is target_output else (healed, True)
                    except ValueError:
                        pass
                restore_state_snapshot(baseline_record.post, args=args,
                                       kwargs=kwargs, module=module)
                overrides = self._baseline_storage_overrides(
                    baseline_record, args, kwargs, module)
                rebuilt = reconstruct_from_snapshot(
                    baseline_record.post, storage_overrides=overrides,
                    tensor_overrides=self._output_identity_overrides(
                        self.model, baseline_record))
                _bind_module_aliases(module, baseline_record, rebuilt.get("module"))
                record.heal_status = "restored_and_output_rebuilt"
                return rebuilt.get("outputs"), True
        except (ValueError, RuntimeError) as exc:
            record.heal_status = "failed"
            record.heal_error = type(exc).__name__
            return None, False
    @staticmethod
    def _output_identity_overrides(module, baseline_record):
        if module is None:
            return {}
        live = {}
        for _, submodule in module.named_modules(remove_duplicate=False):
            for table in (submodule._parameters, submodule._buffers):
                for value in table.values():
                    if value is not None:
                        live[id(value)] = value
        overrides = {}
        for view in baseline_record.post.views:
            if _tree_root(view.tree_path) != "outputs" or view.tensor_id is None:
                continue
            tensor = live.get(view.tensor_id)
            if tensor is not None:
                overrides[view.tree_path] = tensor
        return overrides
    @staticmethod
    def _baseline_storage_overrides(baseline_record, args, kwargs, module):
        live: dict[str, Any] = {}
        if args is not None:
            live.update(_walk_live_root("args", args))
        if kwargs is not None:
            live.update(_walk_live_root("kwargs", kwargs))
        live.update(_live_module_map(module))
        overrides = {}
        for view in baseline_record.post.views:
            tensor = live.get(view.tree_path)
            node = baseline_record.post.storages.get(view.storage_id)
            if tensor is None or node is None:
                continue
            storage = tensor.untyped_storage()
            if int(storage.nbytes()) == int(node.nbytes):
                overrides[view.storage_id] = storage
        return overrides
    # -- reporting ---------------------------------------------------------
    def _build_report(self):
        trace = self.target
        if trace is None:
            return {"status": "target_missing", "invocations": [], "metrics": []}
        invocations, flat = [], []
        for record in trace.records:
            item = {"module_path": record.module_path, "canonical_module_id": record.module_path,
                    "invocation_ordinal": record.invocation_ordinal, "rank": record.rank,
                    "key": list(record.key), "aliases": list(record.aliases),
                    "metrics": record.metrics, "tensors": record.metrics,
                    "healed": record.healed, "heal_status": record.heal_status,
                    "heal_error": record.heal_error, **_aggregate_metrics(record.metrics),
                    "metrics_by_path": {m.get("tree_path"): m for m in record.metrics}}
            invocations.append(item)
            for metric in record.metrics:
                flat.append({"module_path": record.module_path,
                             "invocation_ordinal": record.invocation_ordinal,
                             "rank": record.rank, **metric})
        return {"status": trace.status, "heal": self.heal, "frontier_mode": self.frontier_mode,
                "healing_frontier": sorted(self.frontier), "rank": self.rank,
                "baseline_names": list(self.baselines), "baseline_count": len(self.baselines),
                "reference": self.reference_name, "jitter_available": len(self.baselines) >= 3,
                "baseline_pair_count": max(0, len(self.baselines) - 1),
                "invocations": invocations, "metrics": flat,
                "module_order": [list(key) for key in trace.order],
                "divergence": trace.divergence, "first_divergence": trace.divergence,
                "error_type": type(trace.error).__name__ if trace.error is not None else None}
    @property
    def report(self):
        if self._report is None:
            self._report = self._build_report()
        return self._report
def _outputs_compatible(baseline_record, target_output) -> bool:
    if "outputs" not in baseline_record.post.roots:
        return target_output is None
    if not _structure_matches(baseline_record.post.roots["outputs"], target_output):
        return False
    expected = {v.tree_path: v for v in baseline_record.post.views if _tree_root(v.tree_path) == "outputs"}
    live = {path_to_string(("outputs",) + tuple(p)): leaf
            for p, leaf in (flatten_tree_with_paths(target_output, dedupe_containers=True)
                            if target_output is not None else ())
            if isinstance(leaf, _require_torch().Tensor)}
    return set(expected) == set(live) and all(
        str(live[p].dtype) == v.dtype_str and tuple(live[p].shape) == v.shape
        for p, v in expected.items())
def _bind_module_aliases(module, baseline_record, rebuilt_module):
    """Bind only genuinely missing module buffers/parameters from a rebuild."""
    if module is None or not isinstance(rebuilt_module, dict):
        return
    output_views = [view for view in baseline_record.post.views
                    if _tree_root(view.tree_path) == "outputs"]
    storage_ids = {view.storage_id for view in output_views}
    tensor_ids = {view.tensor_id for view in output_views if view.tensor_id is not None}
    by_path = {view.tree_path: view for view in baseline_record.post.views}
    for module_name, submodule in module.named_modules(remove_duplicate=False):
        prefix = f"{module_name}." if module_name else ""
        for table_name, table in (("parameters", submodule._parameters),
                                  ("buffers", submodule._buffers)):
            for name in list(table):
                if table_name == "parameters" or table.get(name) is not None:
                    # Never replace a live nn.Parameter with a plain Tensor.
                    continue
                key = f"{prefix}{name}"
                value = rebuilt_module.get(table_name, {}).get(key)
                spec = by_path.get(f"module.{table_name}.{key}")
                if value is None or spec is None:
                    continue
                if spec.storage_id in storage_ids or (
                        spec.tensor_id is not None and spec.tensor_id in tensor_ids):
                    table[name] = value
