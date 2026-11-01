"""Real mindie-bench vllm-ascend precision tasks for the diagnostic primitives.

[Role]
  Reproduce and resolve five verified mindie-bench tasks spanning distributed
  GDN state, SFA draft sequence metadata, KDA native workspace tiling,
  spatial GroupNorm/SparseMLA correctness, and sampler reduce-topk/softmax.

[Invariants]
  * Every CPU task exercises both the historical buggy logic and the fixed
    logic; fixed paths must report zero violations and pass checks.
  * Assertions name the exact failing step/module/storage when a primitive
    localizes a fault.
  * CPU tests are deterministic.  Live NPU tests skip cleanly without
    ``torch_npu`` and never fabricate an NPU execution claim.
  * All artifacts are in-memory or ``tmp_path`` isolated; no host files leak.

[Timeouts & Locks]
  CPU tasks complete in < 10s total; the live A3 task targets < 60s.
  No test acquires locks; NPUGraph capture uses a private stream.

[Side-Effect Policy]
  Reads/writes only caller-owned tensors.  Live tests restore NPU device
  state.  No network, subprocess, environment or persistent file writes.
"""
from __future__ import annotations

import pytest

from mindie_diagnostics.precision import (
    LockstepDiffer,
    MemoryWatchpoint,
    StorageGraph,
    WatchRingBuffer,
    capture_call,
    replay_call,
)
torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Shared task models (CPU and live-NPU)
# ---------------------------------------------------------------------------

def _gdn_states(device):
    torch = pytest.importorskip("torch")
    conv = torch.arange(32, dtype=torch.float32, device=device).reshape(8, 4)
    ssm = (torch.arange(32, dtype=torch.float32, device=device) + 100.0).reshape(8, 4)
    return conv, ssm

class _GDNRecurrent(torch.nn.Module):
    def __init__(self, conv_state, ssm_state, block_ids):
        super().__init__()
        self.conv_state = conv_state
        self.ssm_state = ssm_state
        self.register_buffer("block_ids", block_ids)

    def forward(self, mixed_qkv, num_actual_tokens):
        if num_actual_tokens > 0:
            update = mixed_qkv[:num_actual_tokens].sum(dim=0)
            rows = update.unsqueeze(0).expand(self.block_ids.numel(), -1)
            self.conv_state.index_add_(0, self.block_ids, rows)
            self.ssm_state.index_add_(0, self.block_ids, rows * 2.0)
        else:
            mixed_qkv[:0].sum()
        return mixed_qkv

class _GDNBlock(torch.nn.Module):
    def __init__(self, conv_state, ssm_state, device):
        super().__init__()
        block_ids = torch.tensor([1, 2], dtype=torch.int64, device=device)
        self.self_attn = _GDNRecurrent(conv_state, ssm_state, block_ids)

    def forward(self, mixed_qkv, num_actual_tokens):
        return self.self_attn(mixed_qkv, num_actual_tokens)

class _GDNModel(torch.nn.Module):
    def __init__(self, conv_state, ssm_state, device):
        super().__init__()
        self.layers = torch.nn.ModuleList([_GDNBlock(conv_state, ssm_state, device)])

    def forward(self, mixed_qkv, num_actual_tokens):
        for layer in self.layers:
            mixed_qkv = layer(mixed_qkv, num_actual_tokens)
        return mixed_qkv

class _GDNRunner:
    def __init__(self, model, device):
        torch = pytest.importorskip("torch")
        self.model = model
        self.device = torch.device(device)
        self.last_meta = {}

    def _dummy_run(self, num_tokens, skip_gdn_state_update=False):
        torch = pytest.importorskip("torch")
        actual = 0 if skip_gdn_state_update else num_tokens
        self.last_meta = {
            "num_actual_tokens": actual, "max_query_len": actual,
            "query_start_loc": [0, actual],
        }
        mixed_qkv = torch.arange(
            1, num_tokens * 4 + 1, dtype=torch.float32, device=self.device
        ).reshape(num_tokens, 4)
        return self.model(mixed_qkv, actual)

class _DraftMeta:
    def __init__(self, cum_query_lens, seq_lens):
        self.cum_query_lens, self.seq_lens = cum_query_lens, seq_lens

class _CommonMeta:
    def __init__(self, query_start_loc, seq_lens):
        self.query_start_loc, self.seq_lens = query_start_loc, seq_lens
        self.num_reqs = int(seq_lens.numel())

class _SFABuilder:
    """Historical SFA builder: shared buffers (buggy) vs per-draft buffers."""

    def __init__(self, fixed, device, max_reqs=8, spec_tokens=3):
        torch = pytest.importorskip("torch")
        self.fixed = bool(fixed)
        self.actual_seq_lengths_query = torch.zeros(max_reqs + 1, dtype=torch.int32, device=device)
        self.actual_seq_lengths_key = torch.zeros(max_reqs + 1, dtype=torch.int32, device=device)
        if self.fixed:
            self.spec_actual_seq_lengths_query = [
                torch.zeros(max_reqs + 1, dtype=torch.int32, device=device)
                for _ in range(spec_tokens)]
            self.spec_actual_seq_lengths_key = [
                torch.zeros(max_reqs + 1, dtype=torch.int32, device=device)
                for _ in range(spec_tokens)]
        else:
            self.spec_actual_seq_lengths_query = self.spec_actual_seq_lengths_key = None

    def build_for_drafting(self, common_attn_metadata, draft_index):
        query = (self.spec_actual_seq_lengths_query[draft_index - 1]
                 if self.fixed else self.actual_seq_lengths_query)
        key = (self.spec_actual_seq_lengths_key[draft_index - 1]
               if self.fixed else self.actual_seq_lengths_key)
        num_reqs = common_attn_metadata.num_reqs
        query.zero_(); query[:num_reqs].copy_(common_attn_metadata.query_start_loc[1:num_reqs + 1])
        key.zero_(); key[:num_reqs].copy_(common_attn_metadata.seq_lens[:num_reqs])
        return _DraftMeta(query[:num_reqs], key[:num_reqs])

def _kda_chunk_rows(lengths, chunk_size=64):
    return [min(chunk_size, length - start)
            for length in lengths for start in range(0, length, chunk_size)]

def _run_kda_workspace(torch, lengths, padded_workspace, device):
    """Run the mixed-length KDA chunk writes against one workspace layout."""
    total_tokens = sum(lengths)
    rows = _kda_chunk_rows(lengths)
    num_chunks = len(rows)
    padded_tokens = num_chunks * 64
    offset = padded_tokens if padded_workspace else total_tokens
    workspace = torch.zeros(offset + total_tokens, dtype=torch.float32, device=device)
    u_out_view = workspace[offset:offset + total_tokens]
    u_out_view.copy_(torch.arange(
        1, total_tokens + 1, dtype=torch.float32, device=device))
    watchpoint = MemoryWatchpoint(must_preserve={"u_out_scratch": u_out_view})
    ring = WatchRingBuffer(capacity=num_chunks + 1, max_snapshot_numel=total_tokens,
                           device=device)
    first_stomp = None
    output = None
    try:
        for chunk_idx, valid_rows in enumerate(rows):
            start = chunk_idx * 64
            workspace[start:start + valid_rows] = -1.0
            output = torch.sqrt(u_out_view)
            ring.advance_step()
            ring.watch(output)
            check = watchpoint.check(step=chunk_idx)
            if first_stomp is None and check["changed"]:
                first_stomp = chunk_idx
    finally:
        watchpoint.close()
    return {
        "total_tokens": total_tokens,
        "num_chunks": num_chunks,
        "padded_tokens": padded_tokens,
        "first_stomp": first_stomp,
        "watchpoint": watchpoint,
        "ring": ring,
        "output": output,
    }

def _spatial_group_norm_prefix(x, weight, bias, num_groups=8, eps=1e-5):
    torch = pytest.importorskip("torch")
    n_batch, channels, height, width = x.shape
    per_group = channels // num_groups
    x_reshaped = x.view(n_batch, num_groups, per_group, height, width)
    partial_sum = torch.zeros((n_batch, num_groups), dtype=torch.bfloat16,
                              device=x.device)
    partial_sq = torch.zeros((n_batch, num_groups), dtype=torch.bfloat16,
                             device=x.device)
    for channel in range(per_group):
        for row in range(height):
            for start in range(0, width, 512):
                tile = x_reshaped[:, :, channel, row, start:start + 512].to(
                    torch.bfloat16)
                partial_sum = (partial_sum + tile.sum(dim=-1)).to(torch.bfloat16)
                partial_sq = (partial_sq + (tile * tile).sum(dim=-1)).to(
                    torch.bfloat16)
    elements = float(per_group * height * width)
    mean = partial_sum.to(torch.float32) / elements
    var = torch.clamp(partial_sq.to(torch.float32) / elements - mean * mean, min=0.0)
    grouped = x.view(n_batch, num_groups, -1).to(torch.float32)
    normed = (grouped - mean.unsqueeze(-1)) * torch.rsqrt(
        var.unsqueeze(-1) + eps)
    staging = normed.view(n_batch, channels, height, width).clone()
    return (staging * weight.to(torch.float32).view(1, -1, 1, 1)
            + bias.to(torch.float32).view(1, -1, 1, 1)).to(x.dtype)

def _spatial_group_norm_fixed(x, weight, bias, num_groups=8, eps=1e-5):
    torch = pytest.importorskip("torch")
    n_batch, channels, height, width = x.shape
    grouped = x.float().view(n_batch, num_groups, -1)
    mean = grouped.mean(dim=-1, keepdim=True)
    var = grouped.var(dim=-1, unbiased=False, keepdim=True)
    normed = (grouped - mean) * torch.rsqrt(var + eps)
    out = normed.view(n_batch, channels, height, width)
    out = out * weight.float().view(1, -1, 1, 1) + bias.float().view(1, -1, 1, 1)
    return out.to(x.dtype)

class _SpatialBlock(torch.nn.Module):
    def __init__(self, fixed, weight, bias, num_groups=4):
        super().__init__()
        self.fixed = bool(fixed)
        self.num_groups = num_groups
        self.register_buffer("weight", weight.clone())
        self.register_buffer("bias", bias.clone())

    def forward(self, x):
        fn = _spatial_group_norm_fixed if self.fixed else _spatial_group_norm_prefix
        return fn(x, self.weight, self.bias, num_groups=self.num_groups, eps=1e-5)

def _sparse_mla_decode_step(block_table, topk_indices, seqused_k,
                            enable_device_guard=True):
    torch = pytest.importorskip("torch")
    work_block_table = block_table.clone()
    slots = torch.arange(topk_indices.shape[1], device=topk_indices.device).expand_as(
        topk_indices)
    if enable_device_guard:
        safe_idx = torch.where(
            topk_indices < 0, torch.zeros_like(topk_indices), topk_indices)
        valid = (topk_indices >= 0).sum(dim=-1).to(torch.int32)
        max_seqlen = 128
    else:
        safe_idx = topk_indices
        valid = seqused_k.to(torch.int32)
        max_seqlen = int(seqused_k.max().item())
    work_block_table.scatter_(1, slots, safe_idx)
    return {
        "work_block_table": work_block_table,
        "seqused_k": valid,
        "max_seqlen_k": max_seqlen,
    }

class _SparseMLABlock(torch.nn.Module):
    def __init__(self, enable_device_guard):
        super().__init__()
        self.enable_device_guard = bool(enable_device_guard)

    def forward(self, block_table, topk_indices, seqused_k):
        return _sparse_mla_decode_step(
            block_table, topk_indices, seqused_k,
            enable_device_guard=self.enable_device_guard)

def _sampler_topk_prefix(logits, k, p, top_k):
    torch = pytest.importorskip("torch")
    return torch.topk(logits, k=top_k, dim=-1)

def _sampler_topk_fixed(logits, k, p, top_k):
    torch = pytest.importorskip("torch")
    k_for_topk = min(int(top_k), int(logits.shape[-1]))
    values, indices = torch.topk(logits, k=k_for_topk, dim=-1)
    gathered_values = values
    gathered_indices = indices
    probs = gathered_values.softmax(dim=-1)
    if k is not None:
        kk = k.to(torch.long).clamp(min=1, max=gathered_values.shape[-1])
        sorted_probs, _ = probs.sort(dim=-1, descending=False)
        top_k_count = (sorted_probs.size(1) - kk).unsqueeze(1)
        cutoff = sorted_probs.gather(-1, top_k_count)
        cutoff.masked_fill_((kk == gathered_values.shape[-1]).unsqueeze(1), -float("inf"))
        gathered_values.masked_fill_(probs < cutoff, -float("inf"))
    if p is not None:
        sorted_probs, _ = probs.sort(dim=-1, descending=False)
        cumprob = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumprob <= (1 - p.unsqueeze(1))
        top_p_mask[:, -1] = False
        top_p_count = top_p_mask.sum(dim=-1, keepdim=True)
        cutoff = sorted_probs.gather(-1, top_p_count)
        gathered_values.masked_fill_(probs < cutoff, -float("inf"))
    return gathered_values, gathered_indices

def _sampler_softmax_prefix(cand_logits):
    torch = pytest.importorskip("torch")
    return torch.softmax(cand_logits, dim=-1).to(torch.float32)

def _sampler_softmax_fixed(cand_logits):
    return cand_logits.softmax(dim=-1, dtype=__import__("torch").float32)

class _SoftmaxClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.buggy = False

    def forward(self, cand_logits):
        fn = _sampler_softmax_prefix if self.buggy else _sampler_softmax_fixed
        return fn(cand_logits)

def _sampler_inputs(torch, device="cpu"):
    logits = torch.tensor(
        [[0.1, 2.0, -0.5, 1.0, 0.7, -1.0, 1.8, 0.3]], device=device)
    k = torch.tensor([4], dtype=torch.int32, device=device)
    p = torch.tensor([0.8], dtype=torch.float32, device=device)
    return logits, k, p

def _softmax_logits(torch, device="cpu"):
    return torch.tensor(
        [[-5.0625, 1.2109375, 0.68359375, -1.703125]],
        dtype=torch.bfloat16, device=device)

def _sparse_task_inputs(torch, device="cpu"):
    seqused_k = torch.tensor([32, 12, 32, 16], dtype=torch.int64, device=device)
    slots = torch.arange(8, device=device).expand(4, 8).clone()
    valid_blocks = (seqused_k + 3) // 4
    topk_indices = torch.where(
        slots < valid_blocks.unsqueeze(1), slots, torch.full_like(slots, -1))
    block_table = torch.arange(32, dtype=torch.int64, device=device).reshape(4, 8)
    return block_table, topk_indices, seqused_k

# ---------------------------------------------------------------------------
# Task 1: k3-gdn-idle-dp-state
# ---------------------------------------------------------------------------

def test_task1_k3_gdn_idle_dp_state_watchpoint():
    torch = pytest.importorskip("torch")
    conv_state, ssm_state = _gdn_states("cpu")
    model = _GDNModel(conv_state, ssm_state, "cpu")
    runner = _GDNRunner(model, "cpu")
    stale_block_id, unrelated_block_id = 1, 4
    before_conv = conv_state.clone()
    before_ssm = ssm_state.clone()

    with MemoryWatchpoint(
        must_preserve={"conv_state": conv_state, "ssm_state": ssm_state},
        module=model,
    ) as watchpoint:
        runner._dummy_run(1, skip_gdn_state_update=False)

    assert runner.last_meta == {
        "num_actual_tokens": 1, "max_query_len": 1, "query_start_loc": [0, 1]}
    assert watchpoint.violation_count >= 2
    assert watchpoint.offending_module in {
        "layers.0.self_attn", "model.layers.0.self_attn"}
    assert not torch.equal(conv_state[stale_block_id], before_conv[stale_block_id])
    assert not torch.equal(ssm_state[stale_block_id], before_ssm[stale_block_id])
    assert torch.equal(conv_state[unrelated_block_id], before_conv[unrelated_block_id])
    assert watchpoint.check()["violations"] == watchpoint.violation_count
    assert watchpoint.check()["offending_module"] == watchpoint.offending_module

    fixed_conv, fixed_ssm = _gdn_states("cpu")
    fixed_model = _GDNModel(fixed_conv, fixed_ssm, "cpu")
    fixed_runner = _GDNRunner(fixed_model, "cpu")
    fixed_before = fixed_conv.clone()
    fixed_ssm_before = fixed_ssm.clone()
    with MemoryWatchpoint(
        must_preserve={"conv_state": fixed_conv, "ssm_state": fixed_ssm},
        module=fixed_model,
    ) as fixed_watchpoint:
        fixed_runner._dummy_run(1, skip_gdn_state_update=True)

    assert fixed_runner.last_meta == {
        "num_actual_tokens": 0, "max_query_len": 0, "query_start_loc": [0, 0]}
    assert fixed_watchpoint.violation_count == 0
    assert fixed_watchpoint.offending_module is None
    assert torch.equal(fixed_conv, fixed_before)
    assert torch.equal(fixed_ssm, fixed_ssm_before)

# ---------------------------------------------------------------------------
# Task 2: sfa-mtp-seq-lens-eval-zh-vague
# ---------------------------------------------------------------------------

def test_task2_sfa_mtp_draft_seq_lens_alias_stomp():
    torch = pytest.importorskip("torch")
    meta0 = _CommonMeta(
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([11, 22], dtype=torch.int32))
    meta1 = _CommonMeta(
        torch.tensor([0, 2, 4], dtype=torch.int32),
        torch.tensor([33, 44], dtype=torch.int32))
    meta2 = _CommonMeta(
        torch.tensor([0, 3, 6], dtype=torch.int32),
        torch.tensor([55, 66], dtype=torch.int32))

    builder = _SFABuilder(fixed=False, device="cpu")
    draft_0 = builder.build_for_drafting(meta0, draft_index=1)
    draft_1 = builder.build_for_drafting(meta1, draft_index=2)
    graph = StorageGraph.from_state(meta0=draft_0, meta1=draft_1)

    assert graph.shares_storage(
        draft_0.cum_query_lens, draft_1.cum_query_lens) is True
    assert graph.shares_storage(
        "meta0.cum_query_lens", "meta1.cum_query_lens") is True
    aliases = graph.storage_aliases()
    assert any({"meta0.cum_query_lens", "meta1.cum_query_lens"} <= set(paths)
               for paths in aliases.values())
    assert all(len(key) == 3 and key[0] for key in aliases)

    watched_before = draft_0.cum_query_lens.clone()
    watchpoint = MemoryWatchpoint(must_preserve={
        "cum_query_lens": draft_0.cum_query_lens,
        "seq_lens": draft_0.seq_lens,
    })
    builder.build_for_drafting(meta2, draft_index=2)
    check = watchpoint.check(step=1)
    assert check["changed"] is True
    assert check["violations"] == watchpoint.violation_count >= 2
    assert watchpoint.offending_module is None
    assert not torch.equal(draft_0.cum_query_lens, watched_before)

    fixed_builder = _SFABuilder(fixed=True, device="cpu")
    fixed_0 = fixed_builder.build_for_drafting(meta0, draft_index=1)
    fixed_1 = fixed_builder.build_for_drafting(meta1, draft_index=2)
    fixed_graph = StorageGraph.from_state(meta0=fixed_0, meta1=fixed_1)
    assert fixed_graph.shares_storage(
        fixed_0.cum_query_lens, fixed_1.cum_query_lens) is False
    assert fixed_graph.shares_storage(
        "meta0.cum_query_lens", "meta1.cum_query_lens") is False
    assert fixed_graph.storage_aliases() == {}

    fixed_before = fixed_0.cum_query_lens.clone()
    fixed_watchpoint = MemoryWatchpoint(must_preserve={
        "cum_query_lens": fixed_0.cum_query_lens,
        "seq_lens": fixed_0.seq_lens,
    })
    fixed_builder.build_for_drafting(meta2, draft_index=2)
    fixed_check = fixed_watchpoint.check(step=1)
    assert fixed_check["changed"] is False
    assert fixed_check["violations"] == 0
    assert torch.equal(fixed_0.cum_query_lens, fixed_before)

# ---------------------------------------------------------------------------
# Task 3: kda-chunk-varlen-padded-workspace
# ---------------------------------------------------------------------------

def test_task3_kda_chunk_varlen_padded_workspace_overflow():
    torch = pytest.importorskip("torch")
    lengths = ([16, 32, 64] * 43)[:127]
    assert len(lengths) == 127 and sum(lengths) == 4720
    assert len(_kda_chunk_rows(lengths)) == 127
    assert len(_kda_chunk_rows(lengths)) * 64 == 8128

    prefix = _run_kda_workspace(torch, lengths, padded_workspace=False, device="cpu")
    assert prefix["total_tokens"] == 4720
    assert prefix["num_chunks"] == 127
    assert prefix["padded_tokens"] == 8128
    assert prefix["first_stomp"] == 74
    assert prefix["watchpoint"].violation_count >= 1
    assert prefix["watchpoint"].violations[0]["step"] == 74
    assert prefix["watchpoint"].offending_module is None
    assert prefix["ring"].read_latch()["latch_triggered"] is True
    assert prefix["ring"].read_latch()["epoch"] == 74
    records = {item["epoch"]: item for item in prefix["ring"].read_records()["records"]}
    assert records[74]["nan_count"] > 0
    assert not torch.isfinite(prefix["output"]).all()

    uniform = _run_kda_workspace(torch, [64] * 74, padded_workspace=False,
                                 device="cpu")
    assert uniform["first_stomp"] is None
    assert uniform["watchpoint"].violation_count == 0
    assert uniform["ring"].read_latch()["latch_triggered"] is False
    assert torch.isfinite(uniform["output"]).all()
    assert all(item["nan_count"] == 0 for item in
               uniform["ring"].read_records()["records"])

    postfix = _run_kda_workspace(torch, lengths, padded_workspace=True,
                                 device="cpu")
    assert postfix["first_stomp"] is None
    assert postfix["watchpoint"].violation_count == 0
    assert postfix["ring"].read_latch()["latch_triggered"] is False
    assert torch.isfinite(postfix["output"]).all()

# ---------------------------------------------------------------------------
# Task 4: spatial GroupNorm + SparseMLA decode
# ---------------------------------------------------------------------------

def test_task4_spatial_group_norm_and_sparse_mla_decode():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(91099)
    x = (torch.randn((2, 16, 16, 16), generator=generator) * 1.8 + 0.7).to(
        torch.bfloat16)
    weight = (torch.randn(16, generator=generator) * 0.5 + 1.0).to(torch.bfloat16)
    bias = (torch.randn(16, generator=generator) * 0.2).to(torch.bfloat16)

    prefix_capsule = capture_call(
        _spatial_group_norm_prefix, x, weight, bias, num_groups=4, eps=1e-5)
    prefix_replay = replay_call(
        prefix_capsule, _spatial_group_norm_prefix,
        mode="reference_check", reference_atol=5e-3)
    assert prefix_replay["reference_check"]["status"] == "fail"
    assert prefix_replay["reference_check"]["max_abs_err"] > 5e-3
    assert prefix_replay["verdict"] == "real_numerical_bug"

    fixed_capsule = capture_call(
        _spatial_group_norm_fixed, x, weight, bias, num_groups=4, eps=1e-5)
    fixed_replay = replay_call(
        fixed_capsule, _spatial_group_norm_fixed,
        mode="reference_check", reference_atol=5e-3)
    assert fixed_replay["reference_check"]["status"] == "pass"
    assert fixed_replay["reference_check"]["max_abs_err"] <= 5e-3
    assert fixed_replay["verdict"] == "pass"
    gn_ref = torch.nn.functional.group_norm(
        x.float(), 4, weight.float(), bias.float(), eps=1e-5).to(x.dtype)
    assert torch.allclose(fixed_replay["outputs"], gn_ref, atol=1e-2, rtol=1e-2)

    block = _SpatialBlock(fixed=True, weight=weight, bias=bias)
    with torch.no_grad():
        differ = LockstepDiffer(block)
        differ.run_baseline(lambda: block(x.clone()))
        block.fixed = False
        report = differ.run_target(lambda: block(x.clone()))
        differ.close()
    assert report["status"] == "ok"
    output_metric = report["invocations"][0]["metrics_by_path"]["outputs"]
    assert output_metric["max_abs_err"] > 5e-3

    block_table, topk_indices, seqused_k = _sparse_task_inputs(torch)
    blocked = _sparse_mla_decode_step(
        block_table, topk_indices, seqused_k, enable_device_guard=False)
    guarded = _sparse_mla_decode_step(
        block_table, topk_indices, seqused_k, enable_device_guard=True)
    assert int((blocked["work_block_table"] < 0).sum()) > 0
    assert blocked["work_block_table"].min().item() == -1
    assert int((guarded["work_block_table"] < 0).sum()) == 0
    assert guarded["work_block_table"].min().item() == 0
    assert guarded["seqused_k"].tolist() == [8, 3, 8, 4]
    assert blocked["max_seqlen_k"] == int(seqused_k.max())
    assert guarded["max_seqlen_k"] == 128

    module = _SparseMLABlock(enable_device_guard=True)
    with torch.no_grad():
        sparse_differ = LockstepDiffer(module)
        sparse_differ.run_baseline(
            lambda: module(block_table.clone(), topk_indices.clone(),
                           seqused_k.clone()))
        module.enable_device_guard = False
        sparse_report = sparse_differ.run_target(
            lambda: module(block_table.clone(), topk_indices.clone(),
                           seqused_k.clone()))
        sparse_differ.close()
    assert sparse_report["status"] == "ok"
    sparse_metric = sparse_report["invocations"][0]["metrics_by_path"][
        "outputs.work_block_table"]
    assert sparse_metric["max_abs_err"] >= 1.0

    ring = WatchRingBuffer(capacity=2, max_snapshot_numel=32, device="cpu")
    negative_mask = (blocked["work_block_table"] < 0).to(torch.float32)
    ring.advance_step()
    ring.watch(negative_mask, latch_threshold=0.5)
    latch = ring.read_latch()
    record = ring.read_records()["records"][0]
    assert latch["latch_triggered"] is True
    assert record["max"] == 1.0 and record["argmax"] == 11

    clean_ring = WatchRingBuffer(capacity=2, max_snapshot_numel=32, device="cpu")
    clean_mask = (guarded["work_block_table"] < 0).to(torch.float32)
    clean_ring.advance_step()
    clean_ring.watch(clean_mask, latch_threshold=0.5)
    assert clean_ring.read_latch()["latch_triggered"] is False
    assert clean_ring.read_records()["records"][0]["max"] == 0.0

# ---------------------------------------------------------------------------
# Task 5: sampler oversized top-k + BF16 softmax drift
# ---------------------------------------------------------------------------

def test_task5_sampler_oversized_topk_and_bf16_softmax():
    torch = pytest.importorskip("torch")
    logits, k, p = _sampler_inputs(torch)

    with pytest.raises(RuntimeError, match="topk|selected index"):
        _sampler_topk_prefix(logits, k, p, top_k=4096)
    values, indices = _sampler_topk_fixed(logits, k, p, top_k=4096)
    assert values.shape == logits.shape and indices.shape == logits.shape
    assert values[0, 0].item() == pytest.approx(2.0)
    assert indices[0, 0].item() == 1
    assert torch.isfinite(values[0, 0])
    full_vals, _ = _sampler_topk_fixed(logits, None, torch.tensor([1.0]), top_k=4096)
    assert torch.isfinite(full_vals).all()
    one_vals, _ = _sampler_topk_fixed(logits[:, :1], None, torch.tensor([0.8]), top_k=4)
    assert torch.isfinite(one_vals).all()

    cand_logits = _softmax_logits(torch)
    prefix_capsule = capture_call(_sampler_softmax_prefix, cand_logits)
    prefix_replay = replay_call(
        prefix_capsule, _sampler_softmax_prefix, mode="reference_check")
    assert prefix_replay["reference_check"]["status"] == "fail"
    assert prefix_replay["reference_check"]["max_abs_err"] > 1e-3
    assert prefix_replay["verdict"] == "real_numerical_bug"

    fixed_capsule = capture_call(_sampler_softmax_fixed, cand_logits)
    fixed_replay = replay_call(
        fixed_capsule, _sampler_softmax_fixed, mode="reference_check")
    assert fixed_replay["reference_check"]["status"] == "pass"
    assert fixed_replay["verdict"] == "pass"

    classifier = _SoftmaxClassifier()
    with torch.no_grad():
        differ = LockstepDiffer(classifier)
        differ.run_baseline(lambda: classifier(cand_logits.clone()))
        classifier.buggy = True
        report = differ.run_target(lambda: classifier(cand_logits.clone()))
        differ.close()
    assert report["status"] == "ok"
    softmax_metric = report["invocations"][0]["metrics_by_path"]["outputs"]
    assert softmax_metric["max_abs_err"] > 1e-3

    projection = torch.nn.Linear(4, 4, bias=False)
    capsule = capture_call(projection, torch.randn(2, 4))
    quantized = replay_call(capsule, projection, quant_bits=8)
    assert quantized["fakequant_check"]["status"] == "ok"
    assert quantized["fakequant_check"]["verdict"] in {
        "pass", "inherent_quantization_error", "real_numerical_bug"}
    assert quantized["verdict"] in {
        "pass", "inherent_quantization_error", "real_numerical_bug"}

# ---------------------------------------------------------------------------
# Live Ascend A3 NPU execution
# ---------------------------------------------------------------------------

@pytest.mark.live_npu
def test_live_npu_mindie_bench_tasks_on_a3():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU device is not available")
    prev_device = torch.npu.current_device()
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    try:
        _run_live_npu_tasks(torch, device)
    finally:
        torch.npu.set_device(prev_device)


def _run_live_npu_tasks(torch, device):
    # NPUGraph + WatchRingBuffer latch on a real device-side NaN.
    ring = WatchRingBuffer(capacity=4, max_snapshot_numel=8, device=device)
    static = torch.zeros(4, dtype=torch.float32, device=device)
    graph = torch.npu.NPUGraph()
    capture_stream = torch.npu.Stream(device=device)
    with torch.npu.graph(graph, stream=capture_stream):
        ring.advance_step()
        static.fill_(float("nan"))
        ring.watch(static)
    graph.replay()
    torch.npu.synchronize()
    assert ring.read_latch()["latch_triggered"] is True

    # Task 1: GDN recurrent state stomp.
    conv_state, ssm_state = _gdn_states(device)
    model = _GDNModel(conv_state, ssm_state, device)
    runner = _GDNRunner(model, device)
    with MemoryWatchpoint(
        must_preserve={"conv_state": conv_state, "ssm_state": ssm_state},
        module=model,
    ) as watchpoint:
        runner._dummy_run(1, skip_gdn_state_update=False)
    torch.npu.synchronize()
    assert watchpoint.violation_count >= 2
    assert watchpoint.offending_module in {
        "layers.0.self_attn", "model.layers.0.self_attn"}

    fixed_conv, fixed_ssm = _gdn_states(device)
    fixed_model = _GDNModel(fixed_conv, fixed_ssm, device)
    with MemoryWatchpoint(
        must_preserve={"conv_state": fixed_conv, "ssm_state": fixed_ssm},
        module=fixed_model,
    ) as fixed_watchpoint:
        _GDNRunner(fixed_model, device)._dummy_run(
            1, skip_gdn_state_update=True)
    torch.npu.synchronize()
    assert fixed_watchpoint.violation_count == 0

    # Task 2: SFA draft buffers alias on the pre-fix path.
    meta0 = _CommonMeta(torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
                        torch.tensor([11, 22], dtype=torch.int32, device=device))
    meta1 = _CommonMeta(torch.tensor([0, 2, 4], dtype=torch.int32, device=device),
                        torch.tensor([33, 44], dtype=torch.int32, device=device))
    meta2 = _CommonMeta(torch.tensor([0, 3, 6], dtype=torch.int32, device=device),
                        torch.tensor([55, 66], dtype=torch.int32, device=device))
    builder = _SFABuilder(fixed=False, device=device)
    draft_0 = builder.build_for_drafting(meta0, draft_index=1)
    draft_1 = builder.build_for_drafting(meta1, draft_index=2)
    graph_pre = StorageGraph.from_state(meta0=draft_0, meta1=draft_1)
    assert graph_pre.shares_storage(
        draft_0.cum_query_lens, draft_1.cum_query_lens) is True
    alias_watch = MemoryWatchpoint(must_preserve={
        "cum_query_lens": draft_0.cum_query_lens,
        "seq_lens": draft_0.seq_lens,
    })
    builder.build_for_drafting(meta2, draft_index=2)
    torch.npu.synchronize()
    assert alias_watch.check(step=1)["changed"] is True

    fixed_builder = _SFABuilder(fixed=True, device=device)
    fixed_draft_0 = fixed_builder.build_for_drafting(meta0, draft_index=1)
    fixed_draft_1 = fixed_builder.build_for_drafting(meta1, draft_index=2)
    graph_fixed = StorageGraph.from_state(meta0=fixed_draft_0, meta1=fixed_draft_1)
    assert graph_fixed.shares_storage(
        fixed_draft_0.cum_query_lens, fixed_draft_1.cum_query_lens) is False
    clean_alias_watch = MemoryWatchpoint(must_preserve={
        "cum_query_lens": fixed_draft_0.cum_query_lens,
        "seq_lens": fixed_draft_0.seq_lens,
    })
    fixed_builder.build_for_drafting(meta2, draft_index=2)
    torch.npu.synchronize()
    assert clean_alias_watch.check(step=1)["changed"] is False

    # Task 3: KDA chunk-overflow first stomp and NaN latch on device.
    lengths = ([16, 32, 64] * 43)[:127]
    live_prefix = _run_kda_workspace(torch, lengths, padded_workspace=False,
                                     device=device)
    torch.npu.synchronize()
    assert live_prefix["first_stomp"] == 74
    assert live_prefix["ring"].read_latch()["epoch"] == 74
    live_fixed = _run_kda_workspace(torch, lengths, padded_workspace=True,
                                    device=device)
    torch.npu.synchronize()
    assert live_fixed["first_stomp"] is None
    assert live_fixed["ring"].read_latch()["latch_triggered"] is False

    # Task 4: spatial GroupNorm reference check and guarded SparseMLA decode.
    generator = torch.Generator().manual_seed(91099)
    x = (torch.randn((2, 16, 16, 16), generator=generator) * 1.8 + 0.7
         ).to(torch.bfloat16).to(device)
    weight = (torch.randn(16, generator=generator) * 0.5 + 1.0
              ).to(torch.bfloat16).to(device)
    bias = (torch.randn(16, generator=generator) * 0.2).to(torch.bfloat16).to(device)
    spatial_capsule = capture_call(
        _spatial_group_norm_prefix, x, weight, bias, num_groups=4, eps=1e-5)
    spatial_report = replay_call(
        spatial_capsule, _spatial_group_norm_prefix,
        mode="reference_check", reference_atol=5e-3)
    assert spatial_report["verdict"] == "real_numerical_bug"

    block_table, topk_indices, seqused_k = _sparse_task_inputs(torch, device)
    blocked = _sparse_mla_decode_step(
        block_table, topk_indices, seqused_k, enable_device_guard=False)
    guarded = _sparse_mla_decode_step(
        block_table, topk_indices, seqused_k, enable_device_guard=True)
    assert int((blocked["work_block_table"] < 0).sum()) > 0
    assert int((guarded["work_block_table"] < 0).sum()) == 0
    neg_ring = WatchRingBuffer(capacity=2, max_snapshot_numel=32, device=device)
    neg_ring.advance_step()
    neg_ring.watch((blocked["work_block_table"] < 0).float(), latch_threshold=0.5)
    torch.npu.synchronize()
    assert neg_ring.read_latch()["latch_triggered"] is True

    sparse_module = _SparseMLABlock(enable_device_guard=True)
    with torch.no_grad():
        sparse_differ = LockstepDiffer(sparse_module)
        sparse_differ.run_baseline(
            lambda: sparse_module(block_table.clone(), topk_indices.clone(),
                                  seqused_k.clone()))
        sparse_module.enable_device_guard = False
        sparse_report = sparse_differ.run_target(
            lambda: sparse_module(block_table.clone(), topk_indices.clone(),
                                  seqused_k.clone()))
        sparse_differ.close()
    assert sparse_report["invocations"][0]["metrics_by_path"][
        "outputs.work_block_table"]["max_abs_err"] >= 1.0

    # Task 5: oversized top-k and BF16 softmax drift on device.
    logits, k, p = _sampler_inputs(torch, device)
    with pytest.raises(RuntimeError, match="topk|selected index"):
        _sampler_topk_prefix(logits, k, p, top_k=4096)
    values, indices = _sampler_topk_fixed(logits, k, p, top_k=4096)
    assert values.shape == logits.shape and indices[0, 0].item() == 1

    cand_logits = _softmax_logits(torch, device)
    softmax_capsule = capture_call(_sampler_softmax_prefix, cand_logits)
    softmax_report = replay_call(
        softmax_capsule, _sampler_softmax_prefix, mode="reference_check")
    assert softmax_report["verdict"] == "real_numerical_bug"

    classifier = _SoftmaxClassifier()
    with torch.no_grad():
        classifier_differ = LockstepDiffer(classifier)
        classifier_differ.run_baseline(lambda: classifier(cand_logits.clone()))
        classifier.buggy = True
        classifier_report = classifier_differ.run_target(
            lambda: classifier(cand_logits.clone()))
        classifier_differ.close()
    assert classifier_report["invocations"][0]["metrics_by_path"][
        "outputs"]["max_abs_err"] > 1e-3
    torch.npu.synchronize()
