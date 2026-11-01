"""Deterministic integer stimulus and pure-Python safetensors MoE slimming.

[Role]
  Optional utility 4 (``stimulus-and-slim``): produce reproducible integer
  prompt token ids for baseline/target lockstep runs, and slim a routed-expert
  checkpoint without PyTorch by rewriting safetensors headers and byte ranges.

[Invariants]
  * Stimulus tokens depend only on ``(vocab_size, batch_size, seq_len, seed,
    num_steps)``; no tokenizer, locale, clock or global RNG is consulted.
  * ``slim_moe_checkpoint`` keeps ``num_hidden_layers`` unchanged, filters
    ``experts.<idx>.`` keys with ``idx >= target``, and slices only a routed
    expert dimension stored as dimension 0 (a byte-contiguous slice).  A
    routed expert dimension in another position is rejected, never corrupted.
  * Safetensors output is a valid little-endian length-prefixed JSON header
    plus streamed byte ranges; shard index weight maps drop removed keys.
  * No torch import, no NumPy, no network and no subprocess.

[Timeouts & Locks]
  Linear file scans and 1 MiB streaming copies.  No sleeps or locks.  Callers
  own any filesystem concurrency; output files are written under the caller's
  provided directory.

[Side-Effect Policy]
  ``build_deterministic_stimulus`` is pure.  ``slim_moe_checkpoint`` creates
  ``output_dir`` and writes checkpoint shards, ``config.json`` and any index.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import struct
from pathlib import Path
from typing import Any

_EXPERT_INDEX_RE = re.compile(r"(?:^|\.)experts\.(\d+)(?:\.|$)")
_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}
_EXPERT_CONFIG_KEYS = (
    "n_routed_experts", "num_routed_experts", "routed_expert_num",
    "num_experts", "num_local_experts", "moe_num_experts",
)
_COPY_CHUNK = 1024 * 1024


def build_deterministic_stimulus(vocab_size=32000, batch_size=2, seq_len=16,
                                 seed=42, num_steps=1):
    """Return reproducible integer prompt ids and per-step schedules."""
    for name, value in (("vocab_size", vocab_size), ("batch_size", batch_size),
                        ("seq_len", seq_len), ("num_steps", num_steps)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    state = int(seed) & 0xFFFFFFFF

    def next_token():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state % int(vocab_size)

    steps = []
    for step in range(int(num_steps)):
        token_ids = [[next_token() for _ in range(int(seq_len))]
                     for _ in range(int(batch_size))]
        steps.append({"step": step, "token_ids": token_ids})
    return {
        "prompt_token_ids": steps[0]["token_ids"],
        "steps": steps,
        "vocab_size": int(vocab_size),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "seed": int(seed),
        "num_steps": int(num_steps),
    }


def _read_header(path: Path):
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path} is too short to be safetensors")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len > size - 8 or header_len > 256 * 1024 * 1024:
            raise ValueError(f"{path} has an invalid safetensors header length")
        raw = handle.read(header_len)
    try:
        header = json.loads(raw.decode("utf-8").rstrip(" "))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} has an invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path} safetensors header is not an object")
    return header, 8 + header_len, size


def _dtype_size(dtype: str) -> int:
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported safetensors dtype {dtype!r}") from exc


def _shape_numel(shape) -> int:
    total = 1
    for item in shape:
        total *= int(item)
    return total


def _infer_current_experts(config, headers) -> int | None:
    if isinstance(config, dict):
        for key in _EXPERT_CONFIG_KEYS:
            value = config.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    maximum = None
    shape_candidates = []
    for header in headers:
        for name, info in header.items():
            if name == "__metadata__":
                continue
            match = _EXPERT_INDEX_RE.search(name)
            if match:
                index = int(match.group(1)) + 1
                maximum = index if maximum is None else max(maximum, index)
            elif _is_routed_expert_name(name) and info.get("shape"):
                shape_candidates.append(int(info["shape"][0]))
    if shape_candidates:
        inferred = max(shape_candidates)
        maximum = inferred if maximum is None else max(maximum, inferred)
    return maximum


def _is_routed_expert_name(name: str) -> bool:
    lowered = name.lower()
    if any(marker in lowered for marker in
           ("self_attn", "attn", "attention", "shared")):
        return False
    if "experts" in lowered:
        return True
    dense_mlp = ("gate_proj", "up_proj", "down_proj")
    if any(marker in lowered for marker in dense_mlp):
        return False
    router_markers = ("mlp.gate.", "moe.gate.", ".router.",
                      "block_sparse_moe.gate.", "e_score_correction_bias")
    return any(marker in lowered for marker in router_markers)


def _slice_shape(name, shape, current_experts, target_experts):
    """Return ``(new_shape, slice_dim)`` for one tensor, or None to keep."""
    if _EXPERT_INDEX_RE.search(name):
        # Never slice inside an individual expert; it is either kept whole or
        # removed by the expert-index filter.
        return None
    if not shape or current_experts is None or not _is_routed_expert_name(name):
        return None
    shape = [int(item) for item in shape]
    if shape[0] == current_experts:
        return [target_experts] + shape[1:], 0
    if current_experts in shape[1:]:
        raise ValueError(
            f"{name} stores routed experts in a non-leading dimension; "
            "byte-range slicing would corrupt the tensor")
    return None


def _copy_range(source, destination, offset, length):
    source.seek(offset)
    remaining = int(length)
    while remaining > 0:
        chunk = source.read(min(_COPY_CHUNK, remaining))
        if not chunk:
            raise ValueError("safetensors shard ended before its declared byte range")
        destination.write(chunk)
        remaining -= len(chunk)


def _process_shard(path: Path, output_path: Path, current_experts, target_experts,
                   filtered_names, sliced_names, kept_keys):
    header, data_start, file_size = _read_header(path)
    metadata = header.get("__metadata__")
    new_header: dict[str, Any] = {}
    if metadata is not None:
        new_header["__metadata__"] = metadata
    entries = []
    for name in sorted(key for key in header if key != "__metadata__"):
        info = header[name]
        if not isinstance(info, dict):
            raise ValueError(f"{path}: tensor {name} has an invalid descriptor")
        dtype = str(info.get("dtype"))
        shape = [int(item) for item in info.get("shape", [])]
        offsets = info.get("data_offsets") or [0, 0]
        if len(offsets) != 2:
            raise ValueError(f"{path}: tensor {name} has invalid data offsets")
        start, end = int(offsets[0]), int(offsets[1])
        if start < 0 or end < start or data_start + end > file_size:
            raise ValueError(f"{path}: tensor {name} byte range is out of bounds")
        match = _EXPERT_INDEX_RE.search(name)
        if match and "shared" not in name.lower() and int(match.group(1)) >= target_experts:
            filtered_names.append(name)
            continue
        sliced = _slice_shape(name, shape, current_experts, target_experts)
        itemsize = _dtype_size(dtype)
        if sliced is None:
            new_shape, slice_dim = shape, None
            copy_length = end - start
        else:
            new_shape, slice_dim = sliced
            copy_length = _shape_numel(new_shape) * itemsize
            if copy_length > end - start:
                raise ValueError(f"{path}: sliced tensor {name} exceeds source bytes")
            sliced_names.append(name)
        entries.append((name, dtype, new_shape, start, copy_length, slice_dim))
        kept_keys.add(name)
    cursor = 0
    for name, dtype, shape, start, length, _ in entries:
        new_header[name] = {
            "dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + length],
        }
        cursor += length
    header_bytes = json.dumps(new_header, separators=(",", ":"),
                              ensure_ascii=False).encode("utf-8")
    padding = (-len(header_bytes)) % 8
    header_bytes += b" " * padding
    with path.open("rb") as source, output_path.open("wb") as destination:
        destination.write(struct.pack("<Q", len(header_bytes)))
        destination.write(header_bytes)
        for name, dtype, shape, start, length, _ in entries:
            if length:
                _copy_range(source, destination, data_start + start, length)
    return len(entries), cursor


def _update_config(config_path: Path, output_path: Path, current_experts,
                   target_experts, added_keys):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.json must contain an object")
    before_layers = config.get("num_hidden_layers")
    found = False
    for key in _EXPERT_CONFIG_KEYS:
        if key in config:
            config[key] = target_experts
            found = True
    if not found:
        config["n_routed_experts"] = target_experts
        added_keys.append("n_routed_experts")
    if isinstance(config.get("num_experts_per_tok"), int):
        config["num_experts_per_tok"] = min(config["num_experts_per_tok"], target_experts)
    if isinstance(config.get("n_group"), int):
        group = int(config["n_group"])
        if group <= 0 or group > target_experts or target_experts % group != 0:
            group = math.gcd(group, int(target_experts))
        group = max(1, group)
        topk_group = config.get("topk_group", group)
        try:
            topk_group = int(topk_group)
        except (TypeError, ValueError):
            topk_group = group
        topk_group = min(max(1, topk_group), group)
        selected = next(
            (config[key] for key in
             ("num_experts_per_tok", "num_selected_experts", "top_k")
             if isinstance(config.get(key), int)), None)
        if selected is not None:
            selected = min(int(selected), int(target_experts))
            while (group > 1
                   and topk_group * (target_experts // group) < selected):
                needed = math.ceil(selected / (target_experts // group))
                if needed <= group:
                    topk_group = needed
                    break
                group = max(
                    (divisor for divisor in range(1, group)
                     if target_experts % divisor == 0), default=1)
                topk_group = min(topk_group, group)
        config["n_group"] = group
        config["topk_group"] = topk_group
    if "num_hidden_layers" in config:
        config["num_hidden_layers"] = before_layers
    output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    return before_layers, config.get("num_hidden_layers")


def slim_moe_checkpoint(input_dir, output_dir, target_routed_experts):
    """Slim routed experts in a safetensors checkpoint and update its config."""
    source_dir = Path(input_dir)
    target_dir = Path(output_dir)
    target = int(target_routed_experts)
    if target < 1:
        raise ValueError("target_routed_experts must be >= 1")
    if source_dir.resolve() == target_dir.resolve():
        raise ValueError("input_dir and output_dir must differ")
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    shards = sorted(source_dir.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"no .safetensors files found under {source_dir}")
    headers = [_read_header(path)[0] for path in shards]
    config_path = source_dir / "config.json"
    config = None
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    current = _infer_current_experts(config, headers)
    if current is not None and target > current:
        raise ValueError(
            f"target_routed_experts={target} exceeds available {current} experts")
    if current is None:
        current = target
    target_dir.mkdir(parents=True, exist_ok=True)
    filtered_names: list[str] = []
    sliced_names: list[str] = []
    kept_keys: set[str] = set()
    shard_summaries = []
    for path in shards:
        output_path = target_dir / path.name
        tensors, data_bytes = _process_shard(
            path, output_path, current, target, filtered_names, sliced_names, kept_keys)
        shard_summaries.append({
            "name": path.name, "tensors": tensors, "data_bytes": data_bytes,
        })
    added_keys: list[str] = []
    layers_before = layers_after = None
    if config is not None:
        layers_before, layers_after = _update_config(
            config_path, target_dir / "config.json", current, target, added_keys)
    index_updated = None
    for index_path in sorted(source_dir.glob("*.index.json")):
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if isinstance(weight_map, dict):
            kept = {name: shard for name, shard in weight_map.items() if name in kept_keys}
            removed = len(weight_map) - len(kept)
            index["weight_map"] = kept
            metadata = index.get("metadata")
            if isinstance(metadata, dict) and "total_size" in metadata:
                shard_names = {str(shard) for shard in kept.values()}
                metadata["total_size"] = sum(
                    (target_dir / name).stat().st_size
                    for name in shard_names if (target_dir / name).exists())
            (target_dir / index_path.name).write_text(
                json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            index_updated = {"name": index_path.name, "removed_keys": removed,
                             "kept_keys": len(kept)}
    copied = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".safetensors" or path.name == "config.json":
            continue
        if path.name.endswith(".index.json"):
            continue
        destination = target_dir / path.name
        shutil.copy2(path, destination)
        copied.append(path.name)
    return {
        "status": "ok",
        "input_dir": str(source_dir),
        "output_dir": str(target_dir),
        "target_routed_experts": target,
        "current_routed_experts": int(current),
        "num_hidden_layers_before": layers_before,
        "num_hidden_layers_after": layers_after,
        "config_updated": config is not None,
        "added_config_keys": added_keys,
        "shards": shard_summaries,
        "filtered_expert_keys": len(filtered_names),
        "sliced_tensor_names": sliced_names,
        "index": index_updated,
        "copied_files": copied,
    }
