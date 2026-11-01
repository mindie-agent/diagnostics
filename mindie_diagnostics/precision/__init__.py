"""Universal three-primitive NPU precision debugging toolkit.

[Role]
  Public surface for the shared storage-graph foundation plus the three
  universal primitives: ``npu-watch``, ``lockstep-diff`` and ``call-repro``.
  Optional deterministic stimulus and safetensors MoE slimming helpers are
  re-exported here.

[Invariants]
  * Importing this package must not import ``torch``; runtime entry points use
    the lazy ``_require_torch`` helper in ``storage_graph``.
  * Every public name is owned by a module in this package; no logic is copied.
  * State snapshots are explicit runtime objects, never import-time state.

[Timeouts & Locks]
  Import-time only.  All bounded work belongs to the owning modules.

[Side-Effect Policy]
  None at import time.  Callers opt in to capture, replay or checkpoint writes.
"""
from .storage_graph import (
    StateSnapshot,
    StorageNode,
    TensorViewSpec,
    _require_torch,
    capture_state_snapshot,
    flatten_tree_with_paths,
    map_tree,
    path_to_string,
    reconstruct_from_snapshot,
    restore_state_snapshot,
    snapshot_tensor_map,
    walk_tree,
)
from .npu_watch import MemoryWatchpoint, WatchRingBuffer, views_overlap
from .lockstep_diff import LockstepDiffer, resolve_healing_frontier
from .call_repro import (
    CallCapsule,
    FakeQuantContract,
    capture_call,
    capsule_summary,
    fake_quantize,
    replay_call,
)
from .stimulus import build_deterministic_stimulus, slim_moe_checkpoint

__all__ = [
    "StateSnapshot", "StorageNode", "TensorViewSpec", "_require_torch",
    "capture_state_snapshot", "restore_state_snapshot", "reconstruct_from_snapshot",
    "snapshot_tensor_map", "walk_tree", "map_tree", "flatten_tree_with_paths",
    "path_to_string", "WatchRingBuffer", "MemoryWatchpoint", "views_overlap",
    "LockstepDiffer", "resolve_healing_frontier", "CallCapsule",
    "FakeQuantContract", "capture_call", "replay_call", "capsule_summary",
    "fake_quantize", "build_deterministic_stimulus", "slim_moe_checkpoint",
]
