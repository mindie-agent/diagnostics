"""Zero-sync NPU HBM ring-buffer oscilloscope and stride-aware watchpoint.

[Role]
  Primitive 1 (``npu-watch``): pre-allocate a fixed device ring buffer, record
  universal tensor telemetry with device-side epoch/index math, latch the
  instant NaN/Inf/threshold predicates fire, and detect byte changes over the
  exact physical elements of arbitrary strided tensor views.

[Invariants]
  * ``watch()`` performs no ``.item()``, no ``bool(tensor)`` truth test and no
    device-to-host copy.  Epoch/slot/latch decisions execute on device.
  * Valid-mask masking happens before every reduction, with identity values
    that remove padded NaNs; inactive rows never contaminate active statistics.
  * ``completion_seq`` is published last; latch snapshots are claimed via
    ``latch_triggered`` and published via ``snapshot_complete`` last, so later
    steps cannot overwrite a latched crime scene.
  * The memory watchpoint hashes only bytes at the tensor's real strided
    offsets.  ``views_overlap`` intersects physical byte runs, never bounding
    ``[min, max]`` intervals, so disjoint even/odd columns do not alias.

[Timeouts & Locks]
  Watch calls are bounded device kernels on the caller's current stream.  No
  host synchronization, sleeps, subprocesses or locks.  ``read_records`` and
  ``read_latch`` are explicit post-mortem host synchronizations.

[Side-Effect Policy]
  Allocates fixed device buffers at construction.  Watch calls mutate only
  those buffers (and replace ``MemoryWatchpoint``'s in-process digest records).
  No files, network or environment writes.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from .storage_graph import _require_torch, _storage_bytes, _storage_key  # noqa: F401

_HASH_CHUNK_BYTES = 1024 * 1024


def _same_device(a, b) -> bool:
    torch = _require_torch()
    left, right = torch.device(a), torch.device(b)
    if left.type != right.type:
        return False

    def resolved_index(device):
        if device.index is not None:
            return int(device.index)
        if device.type == "cpu":
            return 0
        backend = getattr(torch, device.type, None)
        getter = getattr(backend, "current_device", None)
        if getter is not None:
            try:
                return int(getter())
            except (RuntimeError, OSError, TypeError):
                return None
        return None

    return resolved_index(left) == resolved_index(right)


def _is_compiling() -> bool:
    torch = _require_torch()
    for owner in (getattr(torch, "compiler", None), getattr(torch, "_dynamo", None)):
        getter = getattr(owner, "is_compiling", None) if owner is not None else None
        if getter is not None:
            try:
                if getter():
                    return True
            except (RuntimeError, TypeError):
                continue
    return False


def _event_like(value) -> bool:
    torch = _require_torch()
    event_type = getattr(torch, "Event", None)
    if event_type is not None and isinstance(value, event_type):
        return True
    module = str(getattr(type(value), "__module__", ""))
    if module.startswith(("torch", "torch_npu")):
        return True
    return any(callable(getattr(value, name, None))
               for name in ("record", "query", "synchronize"))


def _current_stream(device):
    torch = _require_torch()
    backend = getattr(torch, device.type, None)
    getter = getattr(backend, "current_stream", None)
    if getter is None:
        return None
    try:
        return getter(device)
    except (TypeError, RuntimeError, OSError):
        try:
            return getter()
        except (TypeError, RuntimeError, OSError):
            return None


def _current_stream_id(device):
    stream = _current_stream(device)
    if stream is None:
        return None
    for name in (f"{device.type}_stream", "stream_id", "cuda_stream", "npu_stream"):
        value = getattr(stream, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _storage_view(storage):
    torch = _require_torch()
    nbytes = int(storage.nbytes())
    view = torch.empty(0, dtype=torch.uint8, device=storage.device)
    view.set_(storage, 0, (nbytes,), (1,))
    return view


def _as_device_tensor(value, *, dtype, device):
    torch = _require_torch()
    if isinstance(value, torch.Tensor):
        if not _same_device(value.device, device):
            raise ValueError("mask/length tensors must live on the watched device")
        return value.to(dtype=dtype)
    if isinstance(value, (int, float, bool)):
        return torch.full((), value, dtype=dtype, device=device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _mask_for(tensor, valid_mask, valid_length):
    torch = _require_torch()
    shape = tuple(tensor.shape)
    mask = None
    if valid_mask is not None:
        mask = _as_device_tensor(valid_mask, dtype=torch.bool, device=tensor.device)
        if tuple(mask.shape) != shape:
            try:
                if tuple(mask.shape) == shape[:1] and len(shape) > 1:
                    mask = mask.reshape(shape[:1] + (1,) * (len(shape) - 1))
                mask = torch.broadcast_to(mask, shape)
            except RuntimeError as exc:
                raise ValueError("valid_mask is not broadcastable to the watched tensor") from exc
    if valid_length is not None:
        if len(shape) == 0:
            raise ValueError("valid_length requires a tensor with dimension 0")
        length = _as_device_tensor(valid_length, dtype=torch.int64, device=tensor.device)
        if int(length.numel()) == 1:
            length = length.reshape(())
        index = torch.arange(shape[0], dtype=torch.int64, device=tensor.device)
        if length.dim() == 0:
            index = index.reshape((shape[0],) + (1,) * (len(shape) - 1))
            length_mask = index < length
        else:
            try:
                length = length.reshape((shape[0],) + (1,) * (len(shape) - 1))
                length_mask = index.reshape((shape[0],) + (1,) * (len(shape) - 1)) < length
            except RuntimeError as exc:
                raise ValueError("valid_length must be scalar or have length shape[0]") from exc
        mask = length_mask if mask is None else mask & length_mask
    if mask is not None and tuple(mask.shape) != shape:
        try:
            mask = torch.broadcast_to(mask, shape)
        except RuntimeError as exc:
            raise ValueError("mask is not broadcastable to the watched tensor") from exc
    return mask


def _scalar(value, dtype, device):
    torch = _require_torch()
    return torch.full((), value, dtype=dtype, device=device)


class WatchRingBuffer:
    """Fixed-size, graph-capture-safe telemetry ring with an atomic latch."""

    def __init__(self, capacity=64, max_snapshot_numel=65536, device="cpu",
                 probe_id="default", graph_id=0, stream_id=0):
        torch = _require_torch()
        if int(capacity) < 1:
            raise ValueError("capacity must be >= 1")
        if int(max_snapshot_numel) < 0:
            raise ValueError("max_snapshot_numel must be >= 0")
        self.capacity = int(capacity)
        self.max_snapshot_numel = int(max_snapshot_numel)
        self.device = torch.device(device)
        self.probe_id = probe_id
        self.graph_id = graph_id
        self.stream_id = stream_id
        self.epoch_counter = torch.full((), -1, dtype=torch.int64, device=self.device)
        self.records_f32 = torch.zeros((self.capacity, 4), dtype=torch.float32, device=self.device)
        self.records_i64 = torch.zeros((self.capacity, 6), dtype=torch.int64, device=self.device)
        self.completion_seq = torch.full((self.capacity,), -1, dtype=torch.int64, device=self.device)
        self.latch_triggered = torch.zeros((), dtype=torch.int64, device=self.device)
        self.snapshot_complete = torch.zeros((), dtype=torch.int64, device=self.device)
        self.latched_slot = torch.full((), -1, dtype=torch.int64, device=self.device)
        self.latched_epoch = torch.full((), -1, dtype=torch.int64, device=self.device)
        self.latched_record_f32 = torch.zeros((4,), dtype=torch.float32, device=self.device)
        self.latched_record_i64 = torch.zeros((6,), dtype=torch.int64, device=self.device)
        self._watch_stream = None
        self.latched_pre_snapshot = torch.zeros(
            (self.max_snapshot_numel,), dtype=torch.float32, device=self.device)
        self.latched_post_snapshot = torch.zeros(
            (self.max_snapshot_numel,), dtype=torch.float32, device=self.device)
        self.pre_stage_snapshot = torch.zeros(
            (self.max_snapshot_numel,), dtype=torch.float32, device=self.device)
        self.pre_stage_numel = torch.zeros((), dtype=torch.int64, device=self.device)
        # Private fixed scratch avoids per-step allocations in graph replay.
        self._post_stage = torch.zeros(
            (self.max_snapshot_numel,), dtype=torch.float32, device=self.device)
        self._post_stage_numel = torch.zeros((), dtype=torch.int64, device=self.device)
        self._ptr_scratch = torch.zeros((), dtype=torch.int64, device=self.device)
        self.latched_pre_numel = torch.zeros((), dtype=torch.int64, device=self.device)
        self.latched_post_numel = torch.zeros((), dtype=torch.int64, device=self.device)

    def _bind_or_check_stream(self):
        """Bind the ring to one stream, or reject cross-stream writers."""
        if self.device.type not in {"npu", "cuda"} or _is_compiling():
            return None
        stream_id = _current_stream_id(self.device)
        if self._watch_stream is None:
            self._watch_stream = stream_id
        elif stream_id != self._watch_stream:
            raise RuntimeError(
                "WatchRingBuffer is single-writer and bound to a different stream")
        return stream_id

    def _wait_for_producer_event(self, producer_event):
        if producer_event is None:
            return
        wait = getattr(producer_event, "wait", None)
        if callable(wait):
            stream = _current_stream(self.device)
            try:
                wait(stream)
            except TypeError:
                wait()
            return
        stream = _current_stream(self.device)
        wait_event = getattr(stream, "wait_event", None)
        if callable(wait_event) and _event_like(producer_event):
            wait_event(producer_event)
            return
        raise TypeError(
            "producer_event must provide wait() or a real event for the "
            "current stream's wait_event()")

    def advance_step(self):
        """Advance the logical step once, entirely on device."""
        self._bind_or_check_stream()
        self.epoch_counter.add_(1)
        return self

    def capture_pre(self, tensor):
        """Stage up to ``max_snapshot_numel`` pre-op values for a later latch."""
        tensor = self._check_tensor(tensor)
        self._bind_or_check_stream()
        torch = _require_torch()
        flat = tensor.detach().reshape(-1)
        count = min(int(flat.numel()), self.max_snapshot_numel)
        self.pre_stage_snapshot.zero_()
        if count:
            # copy_ converts dtype on device without an extra cast allocation.
            self.pre_stage_snapshot[:count].copy_(flat[:count])
        self.pre_stage_numel.fill_(count)
        return self

    def _check_tensor(self, tensor):
        torch = _require_torch()
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("watch() requires a torch.Tensor")
        if not _same_device(tensor.device, self.device):
            raise ValueError("watched tensor must be on the ring buffer device")
        if tensor.layout != torch.strided:
            raise ValueError("only strided tensors can be watched")
        return tensor

    @staticmethod
    def _flatten_stage(flat, stage, count_tensor):
        count = min(int(flat.numel()), stage.numel())
        stage.zero_()
        if count:
            stage[:count].copy_(flat[:count])
        count_tensor.fill_(count)

    def watch(self, tensor, valid_mask=None, valid_length=None, latch_threshold=None,
              latch_on_nan_inf=True, advance_epoch=False, producer_event=None):
        """Record one telemetry row without host synchronization."""
        torch = _require_torch()
        tensor = self._check_tensor(tensor)
        self._bind_or_check_stream()
        self._wait_for_producer_event(producer_event)
        if advance_epoch:
            self.epoch_counter.add_(1)
        epoch = torch.clamp(self.epoch_counter, min=0)
        slot_1d = torch.remainder(epoch, self.capacity).reshape(1)
        slot = slot_1d.reshape(())
        epoch_1d = epoch.reshape(1)
        x_f32 = tensor.detach().to(torch.float32)
        mask = _mask_for(tensor, valid_mask, valid_length)
        empty_tensor = int(tensor.numel()) == 0

        if empty_tensor:
            zero_f = torch.zeros((), dtype=torch.float32, device=self.device)
            zero_i = torch.zeros((), dtype=torch.int64, device=self.device)
            min_v = max_v = l2_v = abs_v = zero_f
            nan_count = inf_count = valid_count = zero_i
            argmax = zero_i
        elif mask is None:
            min_v = x_f32.min()
            max_v = x_f32.max()
            l2_v = torch.sqrt((x_f32 * x_f32).sum())
            abs_v = x_f32.abs().max()
            nan_count = torch.isnan(x_f32).sum(dtype=torch.int64)
            inf_count = torch.isinf(x_f32).sum(dtype=torch.int64)
            valid_count = _scalar(tensor.numel(), torch.int64, self.device)
            argmax = torch.argmax(x_f32.reshape(-1))
        else:
            zero_f = torch.zeros((), dtype=torch.float32, device=self.device)
            inf_f = torch.full((), float("inf"), dtype=torch.float32, device=self.device)
            neg_inf_f = torch.full((), -float("inf"), dtype=torch.float32, device=self.device)
            false = torch.zeros((), dtype=torch.bool, device=self.device)
            valid_count = mask.sum(dtype=torch.int64)
            empty = valid_count == 0
            l2_v = torch.sqrt((torch.where(mask, x_f32, zero_f) ** 2).sum())
            min_v = torch.where(mask, x_f32, inf_f).min()
            max_v = torch.where(mask, x_f32, neg_inf_f).max()
            abs_v = torch.where(mask, x_f32.abs(), zero_f).max()
            nan_masked = torch.where(mask, torch.isnan(x_f32), false)
            inf_masked = torch.where(mask, torch.isinf(x_f32), false)
            nan_count = nan_masked.sum(dtype=torch.int64)
            inf_count = inf_masked.sum(dtype=torch.int64)
            finite_or_posinf_mask = mask & (~torch.isnan(x_f32)) & (x_f32 > -float("inf"))
            cand_finite = torch.argmax(
                torch.where(finite_or_posinf_mask, x_f32, neg_inf_f).reshape(-1))
            cand_any_valid = torch.argmax(mask.to(torch.int64).reshape(-1))
            argmax = torch.where(
                finite_or_posinf_mask.any(), cand_finite, cand_any_valid)
            min_v = torch.where(empty, torch.zeros_like(min_v), min_v)
            max_v = torch.where(empty, torch.zeros_like(max_v), max_v)
            l2_v = torch.where(empty, torch.zeros_like(l2_v), l2_v)
            abs_v = torch.where(empty, torch.zeros_like(abs_v), abs_v)
            nan_count = torch.where(empty, torch.zeros_like(nan_count), nan_count)
            inf_count = torch.where(empty, torch.zeros_like(inf_count), inf_count)
            argmax = torch.where(empty, torch.zeros_like(argmax), argmax)

        if _is_compiling():
            self._ptr_scratch.zero_()
        else:
            self._ptr_scratch.fill_(int(tensor.untyped_storage().data_ptr()))
        row_f32 = torch.stack((min_v, max_v, l2_v, abs_v)).unsqueeze(0)
        row_i64 = torch.stack((epoch, valid_count, nan_count, inf_count,
                               self._ptr_scratch, argmax)).unsqueeze(0)
        self.records_f32.index_copy_(0, slot_1d, row_f32)
        self.records_i64.index_copy_(0, slot_1d, row_i64)
        self.completion_seq.index_copy_(0, slot_1d, epoch_1d)

        flat_post = x_f32.reshape(-1)
        self._flatten_stage(flat_post, self._post_stage, self._post_stage_numel)
        trigger = torch.zeros((), dtype=torch.bool, device=self.device)
        if latch_on_nan_inf:
            trigger = trigger | (nan_count > 0) | (inf_count > 0)
        if latch_threshold is not None:
            threshold = latch_threshold
            if not isinstance(threshold, torch.Tensor):
                threshold = _scalar(threshold, torch.float32, self.device)
            elif not _same_device(threshold.device, self.device):
                raise ValueError("latch_threshold must live on the ring buffer device")
            trigger = trigger | (abs_v > threshold.reshape(()).to(torch.float32))
        should_latch = trigger & (self.latch_triggered == 0)
        pre_available = self.pre_stage_numel > 0
        pre_values = torch.where(pre_available, self.pre_stage_snapshot,
                                 torch.zeros_like(self.pre_stage_snapshot))
        # Claim first, fill crime-scene data, then publish completion last.
        self.latch_triggered.copy_(torch.where(should_latch, 1, self.latch_triggered))
        self.latched_pre_snapshot.copy_(
            torch.where(should_latch, pre_values, self.latched_pre_snapshot))
        self.latched_post_snapshot.copy_(
            torch.where(should_latch, self._post_stage, self.latched_post_snapshot))
        self.latched_pre_numel.copy_(
            torch.where(should_latch, self.pre_stage_numel, self.latched_pre_numel))
        self.latched_post_numel.copy_(
            torch.where(should_latch, self._post_stage_numel, self.latched_post_numel))
        self.latched_slot.copy_(torch.where(should_latch, slot, self.latched_slot))
        self.latched_epoch.copy_(torch.where(should_latch, epoch, self.latched_epoch))
        self.latched_record_f32.copy_(torch.where(
            should_latch, row_f32.reshape(-1), self.latched_record_f32))
        self.latched_record_i64.copy_(torch.where(
            should_latch, row_i64.reshape(-1), self.latched_record_i64))
        self.snapshot_complete.copy_(
            torch.where(should_latch, 1, self.snapshot_complete))
        self.pre_stage_numel.zero_()
        self._post_stage_numel.zero_()
        return None

    def read_records(self):
        """Post-mortem host read of completed ring slots."""
        f32 = self.records_f32.detach().cpu().tolist()
        i64 = self.records_i64.detach().cpu().tolist()
        completion = self.completion_seq.detach().cpu().tolist()
        epoch = int(self.epoch_counter.detach().cpu())
        rows = []
        for slot, sequence in enumerate(completion):
            if sequence < 0:
                continue
            row_f, row_i = f32[slot], i64[slot]
            rows.append({
                "slot": slot,
                "epoch": int(row_i[0]),
                "valid_count": int(row_i[1]),
                "nan_count": int(row_i[2]),
                "inf_count": int(row_i[3]),
                "storage_ptr": int(row_i[4]),
                "argmax": int(row_i[5]),
                "min": float(row_f[0]),
                "max": float(row_f[1]),
                "l2_norm": float(row_f[2]),
                "abs_max": float(row_f[3]),
                "completion_seq": int(sequence),
            })
        rows.sort(key=lambda item: item["epoch"])
        return {
            "probe_id": self.probe_id,
            "graph_id": self.graph_id,
            "stream_id": self.stream_id,
            "capacity": self.capacity,
            "epoch": epoch,
            "records": rows,
        }

    def read_latch(self):
        """Post-mortem host read of the latched snapshots."""
        slot = int(self.latched_slot.detach().cpu())
        pre_numel = int(self.latched_pre_numel.detach().cpu())
        post_numel = int(self.latched_post_numel.detach().cpu())
        triggered = int(self.latch_triggered.detach().cpu()) != 0
        complete = int(self.snapshot_complete.detach().cpu()) != 0
        pre = self.latched_pre_snapshot.detach().cpu().tolist() if triggered else []
        post = self.latched_post_snapshot.detach().cpu().tolist() if triggered else []
        pre_values = pre[:pre_numel]
        post_values = post[:post_numel]
        return {
            "probe_id": self.probe_id,
            "epoch": int(self.latched_epoch.detach().cpu()),
            "current_epoch": int(self.epoch_counter.detach().cpu()),
            "latch_triggered": triggered,
            "snapshot_complete": complete,
            "latched_slot": slot,
            "slot": slot,
            "pre_numel": pre_numel,
            "post_numel": post_numel,
            "pre_snapshot": pre_values,
            "post_snapshot": post_values,
            "latched_pre_snapshot": pre_values,
            "latched_post_snapshot": post_values,
            "latched_pre_numel": pre_numel,
            "latched_post_numel": post_numel,
        }


# ---------------------------------------------------------------------------
# Stride-aware memory watchpoint


@dataclass
class _WatchedRange:
    name: str
    digest: str
    storage_key: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: str
    itemsize: int
    nbytes: int
    min_byte: int
    max_byte: int
    runs: tuple[tuple[int, int], ...] = ()
    storage_ref: Any = None
    storage_cdata: int = 0
    data_ptr: int = 0


def _selected_byte_runs(tensor) -> list[tuple[int, int]]:
    """Return exact contiguous physical byte runs selected by a strided view."""
    torch = _require_torch()
    if tensor.layout != torch.strided:
        raise ValueError("only strided tensors have byte runs")
    shape = tuple(int(item) for item in tensor.shape)
    if any(item <= 0 for item in shape):
        return []
    itemsize = int(tensor.element_size())
    strides = [int(item) for item in tensor.stride()]
    base = int(tensor.storage_offset()) * itemsize
    if len(shape) == 0:
        return [(base, base + itemsize)]
    if tensor.is_contiguous():
        return [(base, base + math.prod(shape) * itemsize)]
    runs: list[tuple[int, int]] = []

    def descend(dim, offset):
        if len(runs) > 1_000_000:
            raise ValueError("tensor view expands to too many physical byte runs")
        if dim == len(shape) - 1:
            stride = strides[dim]
            if stride == 0:
                runs.append((offset, offset + itemsize))
            elif stride == 1:
                runs.append((offset, offset + shape[dim] * itemsize))
            else:
                for index in range(shape[dim]):
                    start = offset + index * stride * itemsize
                    runs.append((start, start + itemsize))
            return
        stride = strides[dim]
        if stride == 0:
            descend(dim + 1, offset)
            return
        for index in range(shape[dim]):
            descend(dim + 1, offset + index * stride * itemsize)

    descend(0, base)
    return runs


def _merge_runs(runs):
    if not runs:
        return []
    ordered = sorted(runs)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _digest_runs(storage, runs) -> str:
    view = _storage_view(storage)
    digest = hashlib.sha256()
    for start, end in runs:
        if end > int(view.numel()):
            raise ValueError("tensor view extends beyond its storage")
        # Copy only the selected device slice; never the entire allocation.
        cursor = start
        while cursor < end:
            chunk_end = min(end, cursor + _HASH_CHUNK_BYTES)
            chunk = view[cursor:chunk_end].detach().to("cpu", copy=True)
            digest.update(bytes(chunk.tolist()))
            cursor = chunk_end
    return digest.hexdigest()


def _runs_overlap(runs_a, runs_b) -> bool:
    merged_a = _merge_runs(runs_a)
    merged_b = _merge_runs(runs_b)
    left = right = 0
    while left < len(merged_a) and right < len(merged_b):
        a_start, a_end = merged_a[left]
        b_start, b_end = merged_b[right]
        if a_start < b_end and b_start < a_end:
            return True
        if a_end <= b_end:
            left += 1
        else:
            right += 1
    return False


class MemoryWatchpoint:
    """Hash exact strided bytes and answer true physical-overlap questions."""

    def __init__(self):
        self._watched: dict[str, _WatchedRange] = {}
        self._tensors: dict[str, Any] = {}
        self._history: list[dict[str, Any]] = []

    @staticmethod
    def views_overlap(tensor_a, tensor_b) -> bool:
        torch = _require_torch()
        if not isinstance(tensor_a, torch.Tensor) or not isinstance(tensor_b, torch.Tensor):
            raise ValueError("views_overlap requires two torch.Tensor values")
        if not _same_device(tensor_a.device, tensor_b.device):
            return False
        if int(tensor_a.numel()) == 0 or int(tensor_b.numel()) == 0:
            return False
        storage_a = tensor_a.untyped_storage()
        storage_b = tensor_b.untyped_storage()
        base_a = int(storage_a.data_ptr())
        base_b = int(storage_b.data_ptr())
        runs_a = [(base_a + start, base_a + end)
                  for start, end in _selected_byte_runs(tensor_a)]
        runs_b = [(base_b + start, base_b + end)
                  for start, end in _selected_byte_runs(tensor_b)]
        return _runs_overlap(runs_a, runs_b)

    @staticmethod
    def _refresh(record, tensor, storage, runs, digest):
        record.digest = digest
        record.storage_key = _storage_key(storage)
        record.storage_ref = storage
        record.storage_cdata = int(storage._cdata)
        record.data_ptr = int(storage.data_ptr())
        record.runs = tuple(runs)
        record.shape = tuple(int(item) for item in tensor.shape)
        record.stride = tuple(int(item) for item in tensor.stride())
        record.storage_offset = int(tensor.storage_offset())
        record.dtype = str(tensor.dtype)
        record.itemsize = int(tensor.element_size())
        record.nbytes = int(storage.nbytes())
        if runs:
            record.min_byte = min(start for start, _ in runs)
            record.max_byte = max(end for _, end in runs)
        else:
            record.min_byte = record.max_byte = (
                int(tensor.storage_offset()) * int(tensor.element_size()))

    def watch(self, name, tensor):
        torch = _require_torch()
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("watch() requires a torch.Tensor")
        if tensor.layout != torch.strided:
            raise ValueError("only strided tensors can be watched")
        storage_ref = tensor.untyped_storage()
        runs = _selected_byte_runs(tensor)
        digest = _digest_runs(storage_ref, runs)
        record = _WatchedRange(
            name=name,
            digest=digest,
            storage_key=_storage_key(storage_ref),
            shape=tuple(int(item) for item in tensor.shape),
            stride=tuple(int(item) for item in tensor.stride()),
            storage_offset=int(tensor.storage_offset()),
            dtype=str(tensor.dtype),
            itemsize=int(tensor.element_size()),
            nbytes=int(storage_ref.nbytes()),
            min_byte=min((start for start, _ in runs), default=0),
            max_byte=max((end for _, end in runs), default=0),
            runs=tuple(runs),
            storage_ref=storage_ref,
            storage_cdata=int(storage_ref._cdata),
            data_ptr=int(storage_ref.data_ptr()),
        )
        self._watched[name] = record
        self._tensors[name] = tensor
        return record

    def check(self, step=None, module_name=None, op_name=None, active_streams=1):
        changes = []
        rebounds = []
        for name, previous in list(self._watched.items()):
            current = self._tensors[name]
            storage = current.untyped_storage()
            try:
                runs = _selected_byte_runs(current)
                current_digest = _digest_runs(storage, runs)
            except (ValueError, RuntimeError):
                runs, current_digest = [], None
            layout_changed = (
                tuple(int(item) for item in current.shape) != previous.shape
                or tuple(int(item) for item in current.stride()) != previous.stride
                or int(current.storage_offset()) != previous.storage_offset
                or str(current.dtype) != previous.dtype
                or int(current.element_size()) != previous.itemsize)
            rebound = int(storage._cdata) != previous.storage_cdata or layout_changed
            if rebound:
                try:
                    old_digest = (
                        _digest_runs(previous.storage_ref, previous.runs)
                        if previous.storage_ref is not None else previous.digest)
                except (ValueError, RuntimeError, TypeError):
                    old_digest = None
                original_changed = old_digest != previous.digest
                entry = {
                    "name": name, "step": step, "module_name": module_name,
                    "op_name": op_name, "changed": bool(original_changed), "rebound": True,
                    "digest_before": previous.digest,
                    "digest_after": current_digest if current_digest is not None else old_digest,
                    "shape": tuple(int(item) for item in current.shape),
                    "stride": tuple(int(item) for item in current.stride()),
                    "storage_offset": int(current.storage_offset()),
                    "storage_cdata_before": previous.storage_cdata,
                    "storage_cdata_after": int(storage._cdata),
                    "data_ptr_before": previous.data_ptr,
                    "data_ptr_after": int(storage.data_ptr()),
                }
                if original_changed:
                    changes.append(entry)
                else:
                    rebounds.append(entry)
                self._refresh(
                    previous, current, storage, runs,
                    current_digest if current_digest is not None else previous.digest)
                continue
            if current_digest == previous.digest:
                continue
            changes.append({
                "name": name,
                "step": step,
                "module_name": module_name,
                "op_name": op_name,
                "changed": True,
                "digest_before": previous.digest,
                "digest_after": current_digest,
                "shape": tuple(int(item) for item in current.shape),
                "stride": tuple(int(item) for item in current.stride()),
                "storage_offset": int(current.storage_offset()),
                "min_byte": previous.min_byte,
                "max_byte": previous.max_byte,
            })
            self._refresh(previous, current, storage, runs, current_digest)
        if changes:
            self._history.extend(changes)
        smallest = None
        if self._history:
            earliest = min(self._history, key=lambda item: _step_order(item.get("step")))
            smallest = {
                "step": earliest.get("step"),
                "module_name": earliest.get("module_name"),
                "op_name": earliest.get("op_name"),
                "name": earliest.get("name"),
            }
        exact = bool(active_streams == 1 and (module_name is not None or op_name is not None))
        return {
            "changed": bool(changes),
            "rebound": bool(rebounds),
            "rebound_views": rebounds,
            "step": step,
            "module_name": module_name,
            "op_name": op_name,
            "smallest_interval": smallest,
            "smallest_observed_interval": smallest,
            "interval": smallest,
            "attribution_confidence": (
                "exact_single_stream_interval" if exact else "coarse_interval_only"
            ),
            "changes": changes,
            "watched": len(self._watched),
        }


def _step_order(step):
    if isinstance(step, (int, float)) and not isinstance(step, bool):
        return (0, float(step))
    if step is None:
        return (2, 0.0)
    return (1, 0.0)


def views_overlap(tensor_a, tensor_b) -> bool:
    """Module-level alias for ``MemoryWatchpoint.views_overlap``."""
    return MemoryWatchpoint.views_overlap(tensor_a, tensor_b)
