"""Deterministic semantic-block and execution-engine classification.

[Role]
  Map every kernel/communication event to exactly one of the six event
  blocks (Attention, MoE, Comm, Norm, Copy_Layout, Unclassified) and to a
  separate execution engine (aic / aiv / mix / aicpu / hccl / host).  The
  engine is a hardware execution fact, never a resource-bottleneck verdict.

[Invariants]
  * Classification is pure and order-independent; every event receives a
    block, an engine, and a human-readable mapping rule.
  * ``execution_engine`` and ``bound_family`` are orthogonal.  ``bound_family``
    stays ``unknown`` unless a caller later binds measured PMU/wait evidence;
    an AIC/AIV engine string alone never becomes ``cube_bound``/``vector_bound``.
  * Fused MoE communication kernels are MoE, not Comm, so their AIV work is
    not counted twice as exposed communication.
  * A missing/blank name is Unclassified, never a fabricated category.
  * Token lists below are the compact, source-verified subset of the old
    ``kernel_signatures.yaml``; they use no unverified database columns.

[Timeouts & Locks]
  Pure string matching, O(len(text)); no locks.

[Side-Effect Policy]
  No I/O and no mutation outside the returned frozen record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EVENT_BLOCKS: tuple[str, ...] = (
    "Attention",
    "MoE",
    "Comm",
    "Norm",
    "Copy_Layout",
    "Unclassified",
)
ALL_BLOCKS: tuple[str, ...] = EVENT_BLOCKS + ("Host_Gap",)
EXECUTION_ENGINES: tuple[str, ...] = ("aic", "aiv", "mix", "aicpu", "hccl", "host", "unknown")
MAPPING_RULES = {
    "Attention": "op_name_prefix+layer_scope",
    "MoE": "grouped_matmul+dispatch_combine",
    "Comm": "hccl_communication_op",
    "Norm": "rmsnorm+layernorm",
    "Copy_Layout": "transdata+cast+d2d",
    "Host_Gap": "stream_idle_gap",
    "Unclassified": "residual_unmapped",
}

_ATTENTION = (
    "flashattention", "fusedinferattentionscore", "pagedattention",
    "sparseattn", "sparseflash", "mla", "mlaprolog", "mlapreprocess",
    "lightningindex", "compressor", "kvrmsnormropecache", "attention",
    "causalconv", "gdn", "mamba", "deltanet", "qkv", "sharedkv",
)
_MOE = (
    "groupedmatmul", "groupedmm", "gmm", "moe", "expert", "dispatch",
    "combine", "gating", "router", "topk", "swiglu",
)
_FUSED_MOE_COMM = (
    "dispatchffncombine", "ffncombine", "dispatchgmmcombine",
    "moedistribute", "dispatchgmm", "combinegmm",
)
_COMM = (
    "hccl", "hcom", "allreduce", "allgather", "reducescatter", "alltoall",
    "broadcast", "sendrecv",
)
_NORM = ("rmsnorm", "layernorm", "l2norm", "normalization", "norm")
_COPY = (
    "transdata", "transpose", "cast", "copy", "concat", "reshape", "slice",
    "d2d", "memcpy", "gather", "scatter", "pad", "permute", "split", "view",
    "layout", "contiguous", "strided",
)

_FOLD_RE = re.compile(r"[^a-z0-9]+")


def fold_text(value: str | None) -> str:
    return _FOLD_RE.sub("", (value or "").lower())


@dataclass(frozen=True, slots=True)
class Classification:
    block: str
    mapping_rule: str
    execution_engine: str
    bound_family: str
    is_comm: bool
    is_aicpu: bool


def _contains(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _execution_engine(
    *,
    folded_core: str,
    folded_task_type: str,
    table: str,
    is_comm: bool,
    is_aicpu: bool,
) -> str:
    """Return the measured execution engine, independent of bottleneck evidence."""

    if table in {"PYTORCH_API", "CANN_API", "MSTX_EVENTS"}:
        return "host"
    if is_comm or "communication" in folded_core or "hccl" in folded_core:
        return "hccl"
    if is_aicpu or "aicpu" in folded_core or "ai_cpu" in folded_core or "aicpu" in folded_task_type:
        return "aicpu"
    if "mix" in folded_core or "mix" in folded_task_type:
        return "mix"
    if "vector" in folded_core or "aiv" in folded_core or "vector" in folded_task_type:
        return "aiv"
    if "core" in folded_core or "cube" in folded_core or "aic" in folded_core:
        return "aic"
    if "aic" in folded_task_type or "cube" in folded_task_type:
        return "aic"
    if "aiv" in folded_task_type or "vector" in folded_task_type:
        return "aiv"
    if "mix" in folded_task_type:
        return "mix"
    return "unknown"


def classify_event(
    *,
    name: str,
    op_type: str = "",
    task_type: str = "",
    core: str = "",
    table: str = "",
) -> Classification:
    """Classify one event using only its exported string fields."""

    text = fold_text(" ".join((name, op_type, task_type, core)))
    folded_core = fold_text(core)
    folded_task_type = fold_text(task_type)
    is_aicpu = "aicpu" in text or "ai_cpu" in folded_core
    is_comm_source = (
        table == "COMMUNICATION_OP"
        or table == "COMMUNICATION_SCHEDULE_TASK_INFO"
        or "communication" in folded_core
        or "hccl" in folded_core
    )
    is_fused_moe = _contains(text, _FUSED_MOE_COMM)
    is_comm = bool(
        is_comm_source
        or _contains(text, _COMM)
        or (op_type.lower().startswith("hcom") if op_type else False)
    )
    if is_comm and is_fused_moe:
        is_comm = False

    if not text:
        block, rule = "Unclassified", MAPPING_RULES["Unclassified"]
    elif is_fused_moe or _contains(text, _MOE):
        block, rule = "MoE", MAPPING_RULES["MoE"]
    elif is_comm:
        block, rule = "Comm", MAPPING_RULES["Comm"]
    elif _contains(text, _ATTENTION):
        block, rule = "Attention", MAPPING_RULES["Attention"]
    elif _contains(text, _NORM):
        block, rule = "Norm", MAPPING_RULES["Norm"]
    elif _contains(text, _COPY):
        block, rule = "Copy_Layout", MAPPING_RULES["Copy_Layout"]
    else:
        block, rule = "Unclassified", MAPPING_RULES["Unclassified"]

    return Classification(
        block=block,
        mapping_rule=rule,
        execution_engine=_execution_engine(
            folded_core=folded_core,
            folded_task_type=folded_task_type,
            table=table,
            is_comm=is_comm,
            is_aicpu=is_aicpu,
        ),
        # No PMU/wait/AICPU evidence is available at classification time.
        bound_family="unknown",
        is_comm=is_comm,
        is_aicpu=is_aicpu,
    )


def apply(events: list) -> None:
    """Classify events in place (events are caller-owned mutable records)."""

    for event in events:
        result = classify_event(
            name=event.name,
            op_type=event.op_type,
            task_type=event.task_type,
            core=event.core,
            table=event.table,
        )
        event.block = result.block
        event.mapping_rule = result.mapping_rule
        event.execution_engine = result.execution_engine
        event.bound_family = result.bound_family
        event.is_comm = result.is_comm
        event.is_aicpu = result.is_aicpu
