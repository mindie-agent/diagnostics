"""P4 reference-only weight estimation from model config.

[Role]
  Estimate per-device parameter/byte counts from an explicit model config for
  P4 reference only.

[Invariants]
  * Output is labeled reference-only and never used as a measurement.
  * Missing hidden/layer/vocab inputs return an explicit error.
  * No I/O, hardware probing, or runtime provider invocation.

[Timeouts & Locks]
  Pure arithmetic over the supplied config mapping.

[Side-Effect Policy]
  No filesystem/environment/process/global mutation.
"""

from __future__ import annotations

from typing import Any, Mapping

CONFIG_DTYPE_BYTES = {
    "float64": 8, "double": 8, "float32": 4, "float": 4, "fp32": 4,
    "float16": 2, "half": 2, "fp16": 2, "bfloat16": 2, "bf16": 2,
    "int8": 1, "uint8": 1, "fp8": 1,
}


def estimate_weights_from_config(config: Mapping[str, Any], *, tp: int = 1, dp: int = 1,
                                 enable_expert_parallel: bool = False) -> dict[str, Any]:
    """P4 fallback estimate; explicitly reference-only."""
    if not isinstance(config, Mapping):
        return {"error": "config must be a mapping"}
    text = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else config
    hidden = text.get("hidden_size") or 0
    layers = text.get("num_hidden_layers") or 0
    vocab = text.get("vocab_size") or 0
    if not (hidden and layers and vocab):
        return {"error": "missing hidden_size, num_hidden_layers, or vocab_size"}
    intermediate = text.get("intermediate_size") or hidden * 4
    moe_intermediate = text.get("moe_intermediate_size") or intermediate
    shared_intermediate = text.get("shared_expert_intermediate_size") or 0
    experts = text.get("num_experts") or 0
    heads = text.get("num_attention_heads") or 1
    kv_heads = text.get("num_key_value_heads") or heads
    head_dim = text.get("head_dim") or hidden // heads
    dtype = str(text.get("dtype") or config.get("torch_dtype") or "bfloat16")
    bpe = CONFIG_DTYPE_BYTES.get(dtype.lower(), 2)
    tie = bool(text.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))
    embed = vocab * hidden
    lm_head = 0 if tie else vocab * hidden
    attention = layers * (hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim + heads * head_dim * hidden)
    norm = layers * hidden * 4
    if experts > 1:
        expert_params = layers * experts * 3 * hidden * moe_intermediate
        shared = layers * 3 * hidden * shared_intermediate if shared_intermediate else 0.0
        router = layers * hidden * experts
        ff_params = expert_params + shared + router
    else:
        expert_params = shared = router = 0.0
        ff_params = layers * 3 * hidden * intermediate
    total_params = embed + lm_head + attention + norm + ff_params
    ep = tp * dp if enable_expert_parallel else tp
    if experts > 1:
        per_device = ((embed + lm_head + attention + norm) / tp + expert_params / ep
                      + shared / tp + router)
    else:
        per_device = total_params / tp
    return {"source": "config_theory", "confidence": "reference_only", "dtype": dtype,
            "bytes_per_element": bpe, "total_params": total_params, "total_bytes": total_params * bpe,
            "per_device_bytes": per_device * bpe, "is_moe": bool(experts > 1), "num_experts": experts}


__all__ = ["CONFIG_DTYPE_BYTES", "estimate_weights_from_config"]
