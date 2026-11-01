"""End-to-end real-task and live-NPU effectiveness validation for precision primitives.

[Role]
  Empirically validate that the 3 Universal Precision Primitives (`npu-watch`,
  `lockstep-diff`, `call-repro` + `storage_graph`) solve 4 realistic debugging
  tasks where naive tensor diffs/dumps fail, including live Ascend NPU
  (`torch_npu` + `NPUGraph`) execution when available.

[Invariants]
  * Every test quantifies both the baseline/naive failure mode and the
    precision-toolkit resolution with explicit numerical assertions.
  * Live NPU tests (`@pytest.mark.live_npu`) execute on real Ascend NPU
    hardware when `torch_npu` is available and skip cleanly on CPU-only hosts.

[Timeouts & Locks]
  Tier 0/1 CPU tasks complete in < 5s; Tier 2 live NPU graph test completes
  in < 15s.

[Side-Effect Policy]
  All file artifacts use `tmp_path` isolation; zero host pollution.
"""
from __future__ import annotations

import pytest

from mindie_diagnostics.precision import (
    FakeQuantContract,
    LockstepDiffer,
    MemoryWatchpoint,
    WatchRingBuffer,
    capture_call,
    fake_quantize,
    replay_call,
    views_overlap,
)


# ---------------------------------------------------------------------------
# Real Task 1: Dynamic Bucket Replay with NaN-Poisoned Padding & Crime-Scene Latch
# ---------------------------------------------------------------------------


def test_real_task1_dynamic_bucket_nan_padding_and_latch():
    """Prove WatchRingBuffer ignores NaN/Inf in padded bucket rows and latches the exact fault step."""
    torch = pytest.importorskip("torch")

    max_batch = 32
    hidden_dim = 16
    ring = WatchRingBuffer(capacity=4, max_snapshot_numel=max_batch * hidden_dim, device="cpu")

    # Bucket schedule: B32 -> B16 -> B24 -> B16 -> B20 (fault on active row 3!) -> B16 -> B32
    valid_lengths = [32, 16, 24, 16, 20, 16, 32]
    naive_false_alarms = 0

    for step, valid_len in enumerate(valid_lengths):
        ring.advance_step()
        # Allocate fixed [32, 16] graph buffer where inactive rows [valid_len:32] contain stale NaNs/Infs
        buf = torch.full((max_batch, hidden_dim), float("nan"), dtype=torch.float32)
        if valid_len < max_batch:
            buf[valid_len:, -1] = float("inf")
        # Populate active rows [0:valid_len] with clean values
        active = torch.arange(valid_len * hidden_dim, dtype=torch.float32).reshape(valid_len, hidden_dim) * 0.01
        buf[:valid_len] = active

        # Stage pre-op values before the operator executes
        ring.capture_pre(buf)

        # Simulate operator execution: on step 4, inject a NaN corruption into active row 3!
        if step == 4:
            buf[3, 5] = float("nan")
        else:
            buf[:valid_len].mul_(1.01)

        # Naive unmasked check falsely triggers on every step with valid_len < 32
        if step != 4 and bool(torch.isnan(buf).any().item()):
            naive_false_alarms += 1

        len_tensor = torch.tensor(valid_len, dtype=torch.int64)
        ring.watch(buf, valid_length=len_tensor)

    # Naive check had 4 false alarms on steps 1, 2, 3, 5 where valid_len < 32
    assert naive_false_alarms == 4

    # WatchRingBuffer latched strictly on step 4 (the true active-row fault) and preserved it
    # even after steps 5 and 6 wrapped the capacity=4 ring buffer!
    latch = ring.read_latch()
    assert latch["latch_triggered"] is True
    assert latch["snapshot_complete"] is True
    assert latch["epoch"] == 4
    assert latch["current_epoch"] == 6
    # Pre-snapshot at [3, 5] (flat index 3 * 16 + 5 = 53) was finite before step 4 corrupted it
    pre_val = latch["pre_snapshot"][3 * hidden_dim + 5]
    post_val = latch["post_snapshot"][3 * hidden_dim + 5]
    assert pre_val == pytest.approx((3 * hidden_dim + 5) * 0.01)
    assert post_val != post_val  # NaN in post_snapshot

    # Recent ring records for steps 5 and 6 have nan_count == 0 despite NaN padding
    records = {r["epoch"]: r for r in ring.read_records()["records"]}
    assert records[3]["nan_count"] == 0 and records[3]["valid_count"] == 16 * hidden_dim
    assert records[4]["nan_count"] == 1 and records[4]["valid_count"] == 20 * hidden_dim
    assert records[5]["nan_count"] == 0 and records[5]["valid_count"] == 16 * hidden_dim
    assert records[6]["nan_count"] == 0 and records[6]["valid_count"] == 32 * hidden_dim


# ---------------------------------------------------------------------------
# Real Task 2: 8-Layer MoE Model with Shuffle-K Jitter & Router Tie-Break Flip
# ---------------------------------------------------------------------------


def test_real_task2_moe_router_tie_flip_vs_real_bug_localization():
    """Prove lockstep-diff + healing suppresses Layer 2 MoE router tie-flip cascade and pinpoints Layer 6."""
    torch = pytest.importorskip("torch")

    class MoELayer(torch.nn.Module):
        def __init__(self, layer_idx: int, hidden: int = 8):
            super().__init__()
            self.layer_idx = layer_idx
            # Expert 0 and Expert 1 have distinct projections so a Top-1 tie flip changes output dramatically
            gen = torch.Generator().manual_seed(100 + layer_idx)
            self.expert0 = torch.nn.Parameter(torch.randn(hidden, hidden, generator=gen) * 0.4)
            self.expert1 = torch.nn.Parameter(torch.randn(hidden, hidden, generator=gen) * -0.4)
            self.register_buffer("step_buf", torch.zeros(1), persistent=False)
            self.jitter_eps = 0.0
            self.inject_bug = False

        def forward(self, x):
            with torch.no_grad():
                self.step_buf.add_(1.0)
                if self.layer_idx == 2:
                    # Near-tie router score:jitter_eps flips top-1 expert between expert0 and expert1!
                    score_diff = 5e-5 + self.jitter_eps
                else:
                    score_diff = 1.0
                routed = (x @ self.expert0) if score_diff >= 0 else (x @ self.expert1)
                out = x + 0.5 * routed + self.jitter_eps * 0.1
                if self.inject_bug:
                    # Real bug in Layer 6: corrupts activation in-place
                    out.add_(0.25)
                return out

    class MoEStack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([MoELayer(i) for i in range(8)])

        def set_mode(self, jitter_eps: float, bug_layer: int | None = None):
            for idx, layer in enumerate(self.layers):
                layer.jitter_eps = jitter_eps if idx == 2 else jitter_eps * 0.05
                layer.inject_bug = idx == bug_layer

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    torch.manual_seed(7)
    x0 = torch.randn(4, 8)
    model = MoEStack().eval()

    # 1. Naive unhealed layer-by-layer comparison between A1 (jitter=+1e-4) and B (jitter=-1e-4, bug at layer 6)
    naive_layer_errors = {}
    with torch.no_grad():
        model.set_mode(jitter_eps=+1e-4, bug_layer=None)
        cur_a = x0.clone()
        a_outs = []
        for idx, layer in enumerate(model.layers):
            cur_a = layer(cur_a)
            a_outs.append(cur_a.clone())

        model.set_mode(jitter_eps=-1e-4, bug_layer=6)
        cur_b = x0.clone()
        for idx, layer in enumerate(model.layers):
            cur_b = layer(cur_b)
            naive_layer_errors[f"layers.{idx}"] = float((cur_b - a_outs[idx]).abs().max().item())

    # Without healing, Layer 2 flips experts and cascades across Layers 2..7 (all > 0.5 error!),
    # so naive first-divergence blames layers.2 and naive max-error blames layers.7 — missing layers.6!
    assert all(naive_layer_errors[f"layers.{i}"] > 0.5 for i in range(2, 8))
    naive_first_suspect = min((i for i in range(8) if naive_layer_errors[f"layers.{i}"] > 0.1))
    assert naive_first_suspect == 2  # False positive on Layer 2!

    # 2. Now run LockstepDiffer with heal=True and 3 baselines (A1, A2, A3) sampling natural Shuffle-K jitter
    with torch.no_grad(), LockstepDiffer(
        model,
        heal=True,
        frontier=[f"layers.{i}" for i in range(8)],
        frontier_mode="leaf",
    ) as differ:
        for b_idx, eps in enumerate([+1e-4, -1e-4, +0.5e-4], start=1):
            model.set_mode(jitter_eps=eps, bug_layer=None)
            differ.run_baseline(lambda: model(x0.clone()), name=f"A{b_idx}")

        # Target run B: jitter flips Layer 2 AND Layer 6 has the real +0.25 bug
        model.set_mode(jitter_eps=-1e-4, bug_layer=6)
        report = differ.run_target(lambda: model(x0.clone()))

    assert report["status"] == "ok"
    assert report["jitter_available"] is True

    by_layer = {
        inv["module_path"]: inv["metrics_by_path"]["outputs"]
        for inv in report["invocations"]
        if inv["module_path"].startswith("layers.")
    }

    # Layer 2's router tie-flip is calibrated by A1/A2/A3 jitter, so its SJR is <= 1.0!
    assert by_layer["layers.2"]["sjr"] <= 1.01
    # Because Layer 2 was healed in-place to A1, Layers 3, 4, 5 see clean baseline inputs and have ~0 error!
    for clean_idx in (0, 1, 3, 4, 5, 7):
        assert by_layer[f"layers.{clean_idx}"]["max_abs_err"] < 1e-4
        assert by_layer[f"layers.{clean_idx}"]["sjr"] <= 2.0

    # Layer 6 (the true bug) stands out with max_abs_err ~= 0.25 and SJR > 1000!
    assert by_layer["layers.6"]["max_abs_err"] == pytest.approx(0.25, abs=1e-4)
    assert by_layer["layers.6"]["sjr"] > 1000.0
    ranked = sorted(by_layer.items(), key=lambda kv: kv[1]["sjr"], reverse=True)
    assert ranked[0][0] == "layers.6"


# ---------------------------------------------------------------------------
# Real Task 3: Packed 656-Byte Strided KV Views & Cross-Owner Memory Stomping
# ---------------------------------------------------------------------------


def test_real_task3_packed_strided_kv_views_and_stomp_attribution():
    """Prove MemoryWatchpoint distinguishes disjoint interleaved [noPE|PE|scale] views and catches a 1-row stomp."""
    torch = pytest.importorskip("torch")

    # Simulate a 32-token packed 656-byte per-token buffer: [noPE (512B) | PE (128B) | scale (16B)]
    num_tokens = 32
    row_bytes = 656
    kv_arena = torch.zeros((num_tokens, row_bytes), dtype=torch.uint8)

    nope_view = kv_arena[:, :512]
    pe_view = kv_arena[:, 512:640]
    scale_view = kv_arena[:, 640:656]
    must_preserve_nope = kv_arena[16:32, :512]

    # Bounding address ranges [min_byte, max_byte] overlap 100%, but physical strided runs are disjoint!
    assert views_overlap(nope_view, pe_view) is False
    assert views_overlap(pe_view, scale_view) is False
    assert views_overlap(nope_view, scale_view) is False
    assert views_overlap(nope_view, must_preserve_nope) is True

    wp = MemoryWatchpoint()
    wp.watch("nope_all", nope_view)
    wp.watch("scale_all", scale_view)
    wp.watch("must_preserve_16_32", must_preserve_nope)

    # Step 1: RoPE operator updates pe_view in-place across all 32 tokens -> 0 false alarms on noPE or scale!
    pe_view.fill_(42)
    check1 = wp.check(step=1, module_name="mla_attn", op_name="rope_inplace_update", active_streams=1)
    assert check1["changed"] is False

    # Step 2: Accepted draft write updates only rows [0:16] of nope_view -> must_preserve_16_32 is untouched!
    nope_view[:16].fill_(7)
    check2 = wp.check(step=2, module_name="mla_attn", op_name="write_accepted_tokens", active_streams=1)
    assert check2["changed"] is True
    changed_names_step2 = {c["name"] for c in check2["changes"]}
    assert changed_names_step2 == {"nope_all"}
    assert "must_preserve_16_32" not in changed_names_step2

    # Step 3: Innocent op runs without writing -> digest refresh ensures step 2 is NOT re-reported!
    check3 = wp.check(step=3, module_name="mla_attn", op_name="innocent_read_op", active_streams=1)
    assert check3["changed"] is False

    # Step 4: Off-by-one custom op writes 17 rows [0:17] instead of 16, stomping row 16 of must_preserve_16_32!
    nope_view[16, 0] = 99
    check4 = wp.check(step=4, module_name="mla_attn", op_name="off_by_one_scatter_op", active_streams=1)
    assert check4["changed"] is True
    changed_names_step4 = {c["name"] for c in check4["changes"]}
    assert "must_preserve_16_32" in changed_names_step4
    assert check4["attribution_confidence"] == "exact_single_stream_interval"
    stomp_entry = next(c for c in check4["changes"] if c["name"] == "must_preserve_16_32")
    assert stomp_entry["op_name"] == "off_by_one_scatter_op"


# ---------------------------------------------------------------------------
# Real Task 4: Standalone call-repro on In-Place Mutating W4A8 Quantized Block
# ---------------------------------------------------------------------------


def test_real_task4_call_repro_inplace_reset_and_w4a8_fakequant_triage(tmp_path):
    """Prove call-repro resets in-place mutations across 20 repeats and separates W4 loss from scale bugs."""
    torch = pytest.importorskip("torch")

    contract = FakeQuantContract(
        bits=4,
        symmetric=True,
        scale_axis=0,
        group_size=8,
        round_mode="nearest",
        saturate=True,
        accum_dtype="float32",
    )
    gen = torch.Generator().manual_seed(2026)
    orig_bf16_weight = (torch.randn(8, 16, generator=gen) * 0.5).to(torch.bfloat16)
    w4_dequant_weight = fake_quantize(orig_bf16_weight, contract)

    class QuantizedBlock(torch.nn.Module):
        def __init__(self, weight_tensor, broken_scale: bool = False):
            super().__init__()
            self.weight = torch.nn.Parameter(weight_tensor.clone())
            self.broken_scale = broken_scale

        def forward(self, residual, x_view):
            with torch.no_grad():
                # In-place mutation on residual (which shares storage with x_view!)
                w = self.weight.flip(0) if self.broken_scale else self.weight
                proj = x_view.to(torch.float32) @ w.t()
                residual.add_(proj)
                return residual.clone()

    # Shared backing storage between residual and x_view
    arena = torch.ones(2, 24, dtype=torch.float32)
    residual = arena[:, :8]
    x_view = arena[:, 8:24]

    healthy_mod = QuantizedBlock(w4_dequant_weight, broken_scale=False)
    capsule_path = tmp_path / "w4a8_call.pt"
    capture_call(healthy_mod, residual, x_view, save_path=capsule_path)

    # Replay 20 times on the healthy W4A8 kernel:
    # 1) Pre-call storage reset keeps jitter_repeat["max_range"] == 0.0 across all 20 trials
    # 2) float_vs_fakequant measures inherent W4 quantization loss (> 0.01)
    # 3) fakequant_vs_kernel is 0.0 (proving the kernel obeys the W4 contract!)
    healthy_rep = replay_call(
        capsule_path,
        healthy_mod,
        repeat=20,
        fakequant_contract=contract,
        original_float_weight=orig_bf16_weight,
    )
    assert healthy_rep["status"] == "ok"
    assert healthy_rep["jitter_repeat"]["repeats"] == 20
    assert healthy_rep["jitter_repeat"]["max_range"] == 0.0
    fq_healthy = healthy_rep["fakequant_check"]
    assert fq_healthy["status"] == "ok"
    assert fq_healthy["inherent_loss_status"] == "computed_from_explicit_original"
    assert fq_healthy["float_vs_fakequant"]["max_abs_err"] > 0.01
    assert fq_healthy["fakequant_vs_kernel"]["max_abs_err"] < 1e-6

    # Now capture & replay a module whose weight has a scale/row-wiring bug:
    # fakequant_vs_kernel immediately flags a massive discrepancy (> 0.5)!
    broken_mod = QuantizedBlock(w4_dequant_weight.flip(0), broken_scale=False)
    broken_capsule = capture_call(broken_mod, arena[:, :8].clone(), arena[:, 8:24].clone())
    broken_rep = replay_call(
        broken_capsule,
        broken_mod,
        repeat=3,
        fakequant_contract=contract,
        original_float_weight=orig_bf16_weight,
    )
    fq_broken = broken_rep["fakequant_check"]
    assert fq_broken["status"] == "ok"
    assert fq_broken["fakequant_vs_kernel"]["max_abs_err"] > 0.5


# ---------------------------------------------------------------------------
# Tier 2 Live Ascend NPU Validation (@pytest.mark.live_npu)
# ---------------------------------------------------------------------------


@pytest.mark.live_npu
def test_live_npu_graph_watch_ring_and_lockstep():
    """Execute WatchRingBuffer inside real torch.npu.NPUGraph replay + NPU LockstepDiffer & call-repro."""
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU device is not available")

    device = torch.device("npu:0")
    torch.npu.set_device(device)

    ring = WatchRingBuffer(capacity=4, max_snapshot_numel=64, device=device, probe_id="npu_graph_probe")
    static_x = torch.zeros((8, 4), dtype=torch.float32, device=device)
    static_valid_len = torch.tensor(4, dtype=torch.int64, device=device)

    # Warmup before NPUGraph capture + verify single-stream ownership & producer_event on live NPU
    s = torch.npu.Stream(device=device)
    s_other = torch.npu.Stream(device=device)
    prod_event = torch.npu.Event()
    with torch.npu.stream(s_other):
        static_x.fill_(0.0)
        prod_event.record(s_other)
    with torch.npu.stream(s):
        ring.advance_step()
        ring.capture_pre(static_x)
        static_x.add_(1.0)
        ring.watch(static_x, valid_length=static_valid_len, producer_event=prod_event)
    with torch.npu.stream(s_other):
        with pytest.raises(RuntimeError, match="single-writer"):
            ring.advance_step()
        with pytest.raises(RuntimeError, match="single-writer"):
            ring.capture_pre(static_x)
        with pytest.raises(RuntimeError, match="single-writer"):
            ring.watch(static_x)
    torch.npu.synchronize()

    # Reset ring state and capture into a real torch.npu.NPUGraph
    ring = WatchRingBuffer(capacity=4, max_snapshot_numel=64, device=device, probe_id="npu_graph_probe")
    static_x.zero_()
    static_x[4:, :] = float("nan")  # Poison padded tail rows [4:8] on NPU HBM
    static_valid_len.fill_(4)

    graph = torch.npu.NPUGraph()
    capture_stream = torch.npu.Stream(device=device)
    with torch.npu.graph(graph, stream=capture_stream):
        ring.advance_step()
        ring.capture_pre(static_x[:4].clone())
        static_x[:4].add_(2.0)
        ring.watch(static_x, valid_length=static_valid_len, latch_threshold=7.0)

    # Replay graph 5 times on NPU without any host sync inside watch(), mutating
    # device tensor static_valid_len across replays (2 rows = 8 valid elems on even steps,
    # 4 rows = 16 valid elems on odd steps):
    # Step 0: active=2.0 (valid=8), Step 1: active=4.0 (valid=16), Step 2: active=6.0 (valid=8),
    # Step 3: active=8.0 > 7.0 -> LATCH! (valid=16), Step 4: active=10.0 (valid=8)
    for step_idx in range(5):
        static_valid_len.fill_(2 if step_idx % 2 == 0 else 4)
        graph.replay()
    torch.npu.synchronize()

    records = ring.read_records()
    latch = ring.read_latch()
    assert records["epoch"] == 4
    assert all(r["nan_count"] == 0 for r in records["records"])
    by_epoch = {r["epoch"]: r for r in records["records"]}
    assert by_epoch[3]["valid_count"] == 16 and by_epoch[4]["valid_count"] == 8
    assert latch["latch_triggered"] is True
    assert latch["snapshot_complete"] is True
    assert latch["epoch"] == 3  # Latched on step 3 when active values hit 8.0 > 7.0!

    # Also verify LockstepDiffer & call-repro on live NPU tensors (npu:0)
    npu_mod = torch.nn.Linear(4, 4, bias=False).to(device)
    npu_in = torch.ones((2, 4), dtype=torch.float32, device=device)
    with torch.no_grad(), LockstepDiffer(npu_mod, heal=True) as differ:
        for idx in range(3):
            differ.run_baseline(lambda: npu_mod(npu_in.clone()), name=f"A{idx + 1}")
        rep = differ.run_target(lambda: npu_mod(npu_in.clone()))
    assert rep["status"] == "ok" and rep["invocations"][0]["max_abs_err"] == 0.0

    cap = capture_call(npu_mod, npu_in.clone())
    rep_call = replay_call(cap, npu_mod, repeat=3)
    assert rep_call["status"] == "ok" and rep_call["jitter_repeat"]["max_range"] == 0.0

