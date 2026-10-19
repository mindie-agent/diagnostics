"""Read-only safetensors header inspection for weight accounting.

[Role]
Parse only the 8-byte little-endian header length, the JSON header, and file
metadata.  Tensor payload bytes are never read, written, converted, or moved.
The manifest is consumed by the P3 weight source in ``memory_profile.py``.

[Invariants]
- A header is accepted only when its declared length fits the actual file.
- Tensor byte ranges come from ``data_offsets``; overlapping ranges are an
  error, and ranges beyond the payload are an error.
- Unknown dtypes are reported with ``bytes_per_element=None``; they are never
  silently treated as two bytes.
- Per-device sizes are rule-based sharding inference for col/row/expert
  parallel tensors.  They are labelled as inference, not hardware measurement.

[Timeouts & Locks]
No subprocess, network, lock, sleep, or unbounded read.  At most
``max_files`` files and ``max_header_bytes`` per header are examined.

[Side-Effect Policy]
The model directory is opened read-only.  This module only returns data; its
CLI writes JSON to stdout and never modifies model files.
"""

from __future__ import annotations

import argparse
import json
import math
import stat
import struct
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "mindie-weight-inspection/1"
DEFAULT_MAX_HEADER_BYTES = 100_000_000
DEFAULT_MAX_FILES = 4096
DEFAULT_MAX_TENSORS = 1_000_000

DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "U32": 4,
    "I32": 4,
    "U64": 8,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2FNUZ": 1,
}

SHARD_COL_PARALLEL = "col_parallel"
SHARD_ROW_PARALLEL = "row_parallel"
SHARD_EXPERT_PARALLEL = "expert_parallel"
SHARD_REPLICATED = "replicated"

CATEGORY_DISPLAY = {
    "embedding": "Embedding",
    "lm_head": "LM Head",
    "attn_q": "Attention Q",
    "attn_k": "Attention K",
    "attn_v": "Attention V",
    "attn_o": "Attention O",
    "attn_conv": "Attention Conv",
    "attn_kv_compress": "Attention KV Compress",
    "attn_other": "Attention Other",
    "linear_attn_qkv": "Linear Attn QKV",
    "linear_attn_proj": "Linear Attn Proj (Z/O)",
    "linear_attn_conv": "Linear Attn Conv1d",
    "linear_attn_param": "Linear Attn Param",
    "ffn": "FFN (Dense)",
    "moe_expert": "MoE Experts",
    "moe_shared_expert": "MoE Shared Expert",
    "moe_gate": "MoE Router Gate",
    "moe_other": "MoE Other",
    "norm": "LayerNorm / RMSNorm",
    "vision": "Vision Encoder",
    "mtp_expert": "MTP Experts",
    "mtp_shared_expert": "MTP Shared Expert",
    "mtp_gate": "MTP Router Gate",
    "mtp_attn": "MTP Attention",
    "mtp_fc": "MTP FC Proj",
    "mtp_norm": "MTP Norm",
    "mtp_other": "MTP Other",
    "quant_param": "Quant Scales/ZP",
    "other": "Other",
}

CATEGORY_ORDER = [
    "embedding", "lm_head",
    "attn_q", "attn_k", "attn_v", "attn_o",
    "attn_conv", "attn_kv_compress", "attn_other",
    "linear_attn_qkv", "linear_attn_proj", "linear_attn_conv", "linear_attn_param",
    "ffn",
    "moe_expert", "moe_shared_expert", "moe_gate", "moe_other",
    "norm", "vision",
    "mtp_expert", "mtp_shared_expert", "mtp_gate", "mtp_attn", "mtp_fc", "mtp_norm", "mtp_other",
    "quant_param", "other",
]


class SafetensorsError(ValueError):
    """A safetensors header or range is structurally invalid."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _header_path(path: str | Path) -> Path:
    return Path(path)


def read_safetensors_header(
    path: str | Path,
    *,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    with_sizes: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], int, int]:
    """Read a safetensors JSON header without reading tensor payload.

    By default this returns the header dict, matching the reference skill's
    interface.  ``with_sizes=True`` also returns ``(header, header_bytes,
    payload_bytes)`` so callers can validate tensor ranges.  Raises
    ``SafetensorsError`` for short files, impossible lengths, invalid JSON, or
    a non-object header.
    """
    if not _is_int(max_header_bytes) or max_header_bytes <= 0:
        raise ValueError("max_header_bytes must be a positive integer")
    file_path = _header_path(path)
    try:
        size = file_path.stat().st_size
        if not stat.S_ISREG(file_path.stat().st_mode):
            raise SafetensorsError(f"not a regular file: {file_path}")
        with file_path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise SafetensorsError(f"truncated length prefix: {file_path}")
            (header_len,) = struct.unpack("<Q", prefix)
            if header_len == 0:
                raise SafetensorsError(f"empty header: {file_path}")
            if header_len > max_header_bytes:
                raise SafetensorsError(
                    f"header length {header_len} exceeds bound {max_header_bytes}: {file_path}"
                )
            if header_len > size - 8:
                raise SafetensorsError(
                    f"header length {header_len} exceeds file payload size {size - 8}: {file_path}"
                )
            raw = handle.read(header_len)
    except OSError as exc:
        raise SafetensorsError(f"cannot read {file_path}: {exc}") from exc
    if len(raw) != header_len:
        raise SafetensorsError(f"truncated header JSON: {file_path}")
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsError(f"invalid header JSON in {file_path}: {exc}") from exc
    if not isinstance(header, dict):
        raise SafetensorsError(f"header is not a JSON object: {file_path}")
    if with_sizes:
        return header, header_len + 8, size - header_len - 8
    return header


def parse_tensor_info(
    header: Mapping[str, Any],
    *,
    payload_bytes: int | None = None,
) -> list[dict[str, Any]]:
    """Validate a header and return tensor metadata sorted by data offset."""
    if not isinstance(header, Mapping):
        raise SafetensorsError("header must be a mapping")
    metadata = header.get("__metadata__")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise SafetensorsError("__metadata__ must be a mapping when present")

    tensors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name:
            raise SafetensorsError("tensor name must be a non-empty string")
        if name in seen:
            raise SafetensorsError(f"duplicate tensor name: {name}")
        seen.add(name)
        if not isinstance(info, Mapping):
            raise SafetensorsError(f"tensor {name!r} metadata is not a mapping")
        dtype = info.get("dtype")
        if not isinstance(dtype, str) or not dtype:
            raise SafetensorsError(f"tensor {name!r} has no dtype")
        shape = info.get("shape")
        if not isinstance(shape, list) or any(not _is_int(d) or d < 0 for d in shape):
            raise SafetensorsError(f"tensor {name!r} has invalid shape")
        offsets = info.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not _is_int(v) or v < 0 for v in offsets)
            or offsets[1] < offsets[0]
        ):
            raise SafetensorsError(f"tensor {name!r} has invalid data_offsets")
        start, end = offsets
        if payload_bytes is not None and end > payload_bytes:
            raise SafetensorsError(
                f"tensor {name!r} ends at {end}, past payload size {payload_bytes}"
            )
        numel = 1
        for dim in shape:
            numel *= dim
        bpe = DTYPE_SIZES.get(dtype)
        expected = None if bpe is None else numel * bpe
        tensors.append({
            "name": name,
            "dtype": dtype,
            "shape": list(shape),
            "numel": numel,
            "data_offsets": [start, end],
            "byte_size": end - start,
            "bytes_per_element": bpe,
            "expected_byte_size": expected,
            "size_matches_dtype": expected is None or expected == end - start,
            "source": "safetensors_header",
            "unit": "byte",
        })

    ordered = sorted(tensors, key=lambda item: (item["data_offsets"][0], item["name"]))
    previous_end = 0
    for index, item in enumerate(ordered):
        start, end = item["data_offsets"]
        if index and start < previous_end:
            raise SafetensorsError(
                f"overlapping data_offsets: {item['name']!r} starts at {start}, "
                f"previous range ends at {previous_end}"
            )
        previous_end = end
    return ordered


def classify_tensor(name: str) -> str:
    """Classify a tensor name into a component category."""
    lowered = name.lower()
    if "visual" in lowered or "vision" in lowered or "vit." in lowered or "image_encoder" in lowered:
        return "vision"
    if "mtp" in lowered:
        if ".experts." in lowered and "shared_expert" not in lowered:
            return "mtp_expert"
        if "shared_expert" in lowered:
            return "mtp_shared_expert"
        if ".gate." in lowered or "shared_expert_gate" in lowered:
            return "mtp_gate"
        if "norm" in lowered:
            return "mtp_norm"
        if ".self_attn." in lowered or ".attention." in lowered:
            return "mtp_attn"
        if ".fc." in lowered:
            return "mtp_fc"
        return "mtp_other"
    if "embed_tokens" in lowered or "wte." in lowered:
        return "embedding"
    if "lm_head" in lowered:
        return "lm_head"
    if ".linear_attn." in lowered:
        if "in_proj_qkv" in lowered:
            return "linear_attn_qkv"
        if "in_proj_z" in lowered or "out_proj" in lowered:
            return "linear_attn_proj"
        if "conv" in lowered:
            return "linear_attn_conv"
        return "linear_attn_param"
    if ".self_attn." in lowered or ".attention." in lowered or ".attn." in lowered:
        if "q_proj" in lowered or "q_a_proj" in lowered or "q_b_proj" in lowered:
            return "attn_q"
        if "k_proj" in lowered or "k_a_proj" in lowered or "k_b_proj" in lowered:
            return "attn_k"
        if "v_proj" in lowered or "v_a_proj" in lowered or "v_b_proj" in lowered:
            return "attn_v"
        if "o_proj" in lowered or "out_proj" in lowered:
            return "attn_o"
        if "conv" in lowered:
            return "attn_conv"
        if "kv_a_layernorm" in lowered or "kv_b_proj" in lowered:
            return "attn_kv_compress"
        return "attn_other"
    if ".mlp." in lowered or ".feed_forward." in lowered or ".ffn." in lowered:
        if "expert" in lowered:
            return "moe_shared_expert" if "shared_expert" in lowered else "moe_expert"
        if "gate" in lowered and not any(token in lowered for token in ("proj", "up", "down")):
            return "moe_gate"
        return "ffn"
    if "block_sparse_moe" in lowered or "moe" in lowered:
        if "expert" in lowered:
            return "moe_shared_expert" if "shared" in lowered else "moe_expert"
        if "gate" in lowered:
            return "moe_gate"
        return "moe_other"
    if "layernorm" in lowered or "layer_norm" in lowered or "norm" in lowered or "ln_" in lowered:
        return "norm"
    if "scale" in lowered or "zero_point" in lowered or "quant_" in lowered:
        return "quant_param"
    return "other"


def classify_shard_strategy(name: str, category: str) -> str:
    """Rule-based shard inference; this is not a hardware measurement."""
    if category in ("norm", "quant_param", "moe_gate", "attn_conv", "attn_other"):
        return SHARD_REPLICATED
    if category == "moe_expert":
        return SHARD_EXPERT_PARALLEL
    if category in ("attn_q", "attn_k", "attn_v", "attn_kv_compress"):
        return SHARD_COL_PARALLEL
    if category == "attn_o":
        return SHARD_ROW_PARALLEL
    if category == "linear_attn_qkv":
        return SHARD_COL_PARALLEL
    if category == "linear_attn_proj":
        return SHARD_ROW_PARALLEL if "out_proj" in name.lower() else SHARD_COL_PARALLEL
    if category in ("linear_attn_conv", "linear_attn_param"):
        return SHARD_COL_PARALLEL
    if category in ("ffn", "moe_shared_expert"):
        lowered = name.lower()
        if "gate_proj" in lowered or "up_proj" in lowered or "w1" in lowered or "w3" in lowered:
            return SHARD_COL_PARALLEL
        if "down_proj" in lowered or "w2" in lowered:
            return SHARD_ROW_PARALLEL
        return SHARD_COL_PARALLEL
    if category in ("embedding", "lm_head"):
        return SHARD_COL_PARALLEL
    if category == "vision":
        lowered = name.lower()
        if "norm" in lowered or "patch_embed" in lowered or "pos_embed" in lowered:
            return SHARD_REPLICATED
        if "qkv" in lowered or "linear_fc1" in lowered or "gate_up" in lowered:
            return SHARD_COL_PARALLEL
        if "proj.weight" in lowered or "proj.bias" in lowered or "linear_fc2" in lowered or "down_proj" in lowered:
            return SHARD_ROW_PARALLEL
        return SHARD_COL_PARALLEL
    if category == "mtp_expert":
        return SHARD_EXPERT_PARALLEL
    if category == "mtp_shared_expert":
        return SHARD_ROW_PARALLEL if "down_proj" in name.lower() or "w2" in name.lower() else SHARD_COL_PARALLEL
    if category in ("mtp_gate", "mtp_norm"):
        return SHARD_REPLICATED
    if category == "mtp_attn":
        return SHARD_ROW_PARALLEL if "o_proj" in name.lower() or "out_proj" in name.lower() else SHARD_COL_PARALLEL
    if category in ("mtp_fc", "mtp_other"):
        return SHARD_COL_PARALLEL
    return SHARD_REPLICATED


def compute_per_device_bytes(
    tensors: Iterable[Mapping[str, Any]],
    *,
    tp: int = 1,
    dp: int = 1,
    enable_expert_parallel: bool = False,
) -> dict[str, Any]:
    """Sum inferred per-device bytes for already-classified tensors."""
    if not _is_int(tp) or tp <= 0:
        raise ValueError("tp must be a positive integer")
    if not _is_int(dp) or dp <= 0:
        raise ValueError("dp must be a positive integer")
    ep = tp * dp if enable_expert_parallel else tp
    total = 0.0
    per_device = 0.0
    categories: dict[str, dict[str, Any]] = {}
    for item in tensors:
        size = item.get("byte_size", 0)
        if not isinstance(size, (int, float)) or isinstance(size, bool) or not math.isfinite(float(size)):
            raise ValueError(f"invalid byte_size for {item.get('name')!r}")
        size = float(size)
        category = str(item.get("category") or classify_tensor(str(item.get("name", ""))))
        strategy = str(item.get("shard_strategy") or classify_shard_strategy(str(item.get("name", "")), category))
        if strategy in (SHARD_COL_PARALLEL, SHARD_ROW_PARALLEL):
            device_size = size / tp
        elif strategy == SHARD_EXPERT_PARALLEL:
            device_size = size / ep
        else:
            device_size = size
        total += size
        per_device += device_size
        bucket = categories.setdefault(category, {
            "total_bytes": 0.0,
            "per_device_bytes": 0.0,
            "tensor_count": 0,
            "dtypes": set(),
            "shard_strategies": set(),
        })
        bucket["total_bytes"] += size
        bucket["per_device_bytes"] += device_size
        bucket["tensor_count"] += 1
        bucket["dtypes"].add(str(item.get("dtype", "?")))
        bucket["shard_strategies"].add(strategy)
    ordered = []
    for key in CATEGORY_ORDER:
        if key not in categories:
            continue
        bucket = categories[key]
        ordered.append({
            "category": key,
            "display_name": CATEGORY_DISPLAY.get(key, key),
            "total_bytes": bucket["total_bytes"],
            "total_gib": bucket["total_bytes"] / (1024 ** 3),
            "per_device_bytes": bucket["per_device_bytes"],
            "per_device_gib": bucket["per_device_bytes"] / (1024 ** 3),
            "tensor_count": bucket["tensor_count"],
            "dtypes": sorted(bucket["dtypes"]),
            "shard_strategies": sorted(bucket["shard_strategies"]),
        })
    return {
        "source": "safetensors_file_headers",
        "sharding": "rule_based_inference",
        "total_bytes": total,
        "total_gib": total / (1024 ** 3),
        "per_device_bytes": per_device,
        "per_device_gib": per_device / (1024 ** 3),
        "tp": tp,
        "dp": dp,
        "ep": ep,
        "categories": ordered,
    }


def inspect_weights(
    model_dir: str | Path,
    *,
    tp: int = 1,
    dp: int = 1,
    enable_expert_parallel: bool = False,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    max_files: int = DEFAULT_MAX_FILES,
    max_tensors: int = DEFAULT_MAX_TENSORS,
    include_tensors: bool = True,
) -> dict[str, Any]:
    """Inspect every top-level ``*.safetensors`` file in read-only mode."""
    root = Path(model_dir)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "source": "safetensors_file_headers",
        "read_only": True,
        "model_dir": str(root),
        "unit": "byte",
        "tp": tp,
        "dp": dp,
        "enable_expert_parallel": bool(enable_expert_parallel),
        "files": [],
        "tensors": [],
        "errors": [],
        "limitations": [],
    }
    if not root.is_dir():
        result["error"] = f"model directory does not exist: {root}"
        return result
    files = sorted(root.glob("*.safetensors"))
    if not files:
        result["error"] = "no safetensors files found"
        return result
    if len(files) > max_files:
        result["limitations"].append({
            "code": "file_count_bounded",
            "message": f"examined the first {max_files} of {len(files)} safetensors files",
        })
        files = files[:max_files]

    all_tensors: list[dict[str, Any]] = []
    total_header_bytes = 0
    total_payload_bytes = 0
    for file_path in files:
        entry: dict[str, Any] = {"name": file_path.name}
        try:
            header, header_bytes, payload_bytes = read_safetensors_header(
                file_path, max_header_bytes=max_header_bytes, with_sizes=True
            )
            tensors = parse_tensor_info(header, payload_bytes=payload_bytes)
            if len(all_tensors) + len(tensors) > max_tensors:
                raise SafetensorsError(
                    f"tensor count exceeds bound {max_tensors} at {file_path.name}"
                )
            for item in tensors:
                item["source_file"] = file_path.name
                item["category"] = classify_tensor(item["name"])
                item["shard_strategy"] = classify_shard_strategy(item["name"], item["category"])
                if item["bytes_per_element"] is None:
                    result["limitations"].append({
                        "code": "unknown_dtype",
                        "message": f"unknown dtype {item['dtype']!r} for {item['name']!r}",
                        "tensor": item["name"],
                    })
                elif not item["size_matches_dtype"]:
                    result["limitations"].append({
                        "code": "dtype_size_mismatch",
                        "message": (
                            f"{item['name']!r} declares {item['byte_size']} bytes, "
                            f"shape/dtype implies {item['expected_byte_size']}"
                        ),
                        "tensor": item["name"],
                    })
            all_tensors.extend(tensors)
            entry.update({
                "header_bytes": header_bytes,
                "payload_bytes": payload_bytes,
                "tensor_count": len(tensors),
                "status": "ok",
            })
            total_header_bytes += header_bytes
            total_payload_bytes += payload_bytes
        except (OSError, SafetensorsError) as exc:
            entry.update({"status": "error", "error": str(exc)})
            result["errors"].append({"file": file_path.name, "message": str(exc)})
        result["files"].append(entry)

    if not all_tensors:
        result["error"] = "no readable safetensors headers"
        return result

    per_device = compute_per_device_bytes(
        all_tensors, tp=tp, dp=dp, enable_expert_parallel=enable_expert_parallel
    )
    result.update({
        "status": "ok" if not result["errors"] else "partial",
        "num_safetensors_files": len(files),
        "readable_files": sum(1 for item in result["files"] if item.get("status") == "ok"),
        "total_tensors": len(all_tensors),
        "total_bytes": per_device["total_bytes"],
        "total_gib": per_device["total_gib"],
        "per_device_bytes": per_device["per_device_bytes"],
        "per_device_gib": per_device["per_device_gib"],
        "header_bytes": total_header_bytes,
        "payload_bytes": total_payload_bytes,
        "categories": per_device["categories"],
        "sharding": per_device["sharding"],
        "tensors": all_tensors if include_tensors else [],
    })
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("model_dir")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--no-tensors", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = inspect_weights(
            args.model_dir,
            tp=args.tp,
            dp=args.dp,
            enable_expert_parallel=args.enable_expert_parallel,
            include_tensors=not args.no_tensors,
        )
    except ValueError as exc:
        result = {"schema_version": SCHEMA_VERSION, "status": "error", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") in ("ok", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DTYPE_SIZES",
    "SafetensorsError",
    "classify_shard_strategy",
    "classify_tensor",
    "compute_per_device_bytes",
    "inspect_weights",
    "parse_tensor_info",
    "read_safetensors_header",
]
