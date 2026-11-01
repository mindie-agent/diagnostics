import dataclasses
import json
import struct
import sys
from collections import namedtuple

import pytest

from mindie_diagnostics import cli
from mindie_diagnostics.precision import (
    FakeQuantContract,
    LockstepDiffer,
    MemoryWatchpoint,
    WatchRingBuffer,
    _require_torch,
    build_deterministic_stimulus,
    capture_call,
    capture_state_snapshot,
    reconstruct_from_snapshot,
    replay_call,
    resolve_healing_frontier,
    restore_state_snapshot,
    slim_moe_checkpoint,
    snapshot_tensor_map,
    views_overlap,
)


PointPair = namedtuple("PointPair", ["left", "right"])


@dataclasses.dataclass
class Nested:
    value: object
    name: str
    extra: object = None


def _write_safetensors_PurePy(path, tensor_specs, metadata=None):
    """Write a valid .safetensors file in pure Python without torch/safetensors."""
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    blobs = []
    cursor = 0
    for name, (dtype, shape, raw_bytes) in tensor_specs.items():
        end = cursor + len(raw_bytes)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, end],
        }
        blobs.append(raw_bytes)
        cursor = end
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(header_bytes)) % 8
    header_bytes += b" " * pad
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for blob in blobs:
            handle.write(blob)


def _read_safetensors_PurePy(path):
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len).decode("utf-8").rstrip(" "))
        body = handle.read()
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        tensors[name] = {
            "dtype": info["dtype"],
            "shape": tuple(info["shape"]),
            "raw_bytes": body[start:end],
        }
    return header, tensors


def test_zero_dependency_lazy_torch_guard(monkeypatch):
    import mindie_diagnostics.precision.storage_graph as sg

    monkeypatch.setattr(sg, "_TORCH", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="PyTorch is required for mindie_diagnostics.precision"):
        _require_torch()


def test_healing_frontier():
    paths = ["model", "model.layers.0", "model.layers.0.mlp", "model.layers.1", "other"]
    assert resolve_healing_frontier(paths, mode="leaf") == {
        "model.layers.0.mlp",
        "model.layers.1",
        "other",
    }
    assert resolve_healing_frontier(paths, mode="top") == {"model", "other"}
    assert resolve_healing_frontier(["model", "model.layers.0"], "top") == {"model"}
    with pytest.raises(ValueError):
        resolve_healing_frontier(paths, "bad")


def test_stimulus_deterministic_and_slim_safetensors_pure_python(tmp_path, capsys):
    first = build_deterministic_stimulus(vocab_size=11, batch_size=2, seq_len=3, seed=9, num_steps=2)
    second = build_deterministic_stimulus(vocab_size=11, batch_size=2, seq_len=3, seed=9, num_steps=2)
    assert first == second
    assert len(first["steps"]) == 2
    assert all(0 <= token < 11 for row in first["prompt_token_ids"] for token in row)

    source, output = tmp_path / "in", tmp_path / "out"
    source.mkdir()
    f32_pack = lambda vals: struct.pack(f"<{len(vals)}f", *vals)
    specs = {
        "embed.weight": ("F32", (5, 2), f32_pack(range(10))),
        "layers.0.mlp.experts.gate.weight": ("F32", (4, 2, 2), f32_pack(range(16))),
        "layers.0.mlp.experts.0.down.weight": ("F32", (2, 2), f32_pack([1, 2, 3, 4])),
        "layers.0.mlp.experts.3.down.weight": ("F32", (2, 2), f32_pack([5, 6, 7, 8])),
        "layers.0.mlp.shared_experts.gate.weight": ("F32", (2,), f32_pack([9, 10])),
    }
    _write_safetensors_PurePy(source / "model.safetensors", specs)
    (source / "config.json").write_text(
        json.dumps(
            {
                "n_routed_experts": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "num_hidden_layers": 7,
            }
        ),
        encoding="utf-8",
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 100}, "weight_map": {k: "model.safetensors" for k in specs}}),
        encoding="utf-8",
    )
    report = slim_moe_checkpoint(source, output, 2)
    assert report["status"] == "ok"
    assert report["num_hidden_layers_before"] == report["num_hidden_layers_after"] == 7
    _, saved = _read_safetensors_PurePy(output / "model.safetensors")
    assert saved["layers.0.mlp.experts.gate.weight"]["shape"] == (2, 2, 2)
    assert saved["layers.0.mlp.experts.gate.weight"]["raw_bytes"] == f32_pack(range(8))
    assert "layers.0.mlp.experts.3.down.weight" not in saved
    assert "layers.0.mlp.experts.0.down.weight" in saved
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["n_routed_experts"] == 2 and config["num_hidden_layers"] == 7

    # Also test CLI `precision slim-moe` and `precision compare-traces` and `precision watch`
    cli_out = tmp_path / "cli_out"
    assert cli.main(["precision", "slim-moe", str(source), str(cli_out), "2"]) == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["status"] == "ok" and cli_payload["target_routed_experts"] == 2

    trace_a = tmp_path / "trace_a.json"
    trace_b = tmp_path / "trace_b.json"
    trace_a.write_text(
        json.dumps({"metrics": [{"module_path": "m", "invocation_ordinal": 1, "tree_path": "outputs", "max_abs_err": 0.1, "sjr": 1.0}]}),
        encoding="utf-8",
    )
    trace_b.write_text(
        json.dumps({"metrics": [{"module_path": "m", "invocation_ordinal": 1, "tree_path": "outputs", "max_abs_err": 0.5, "sjr": 5.0}]}),
        encoding="utf-8",
    )
    assert cli.main(["precision", "compare-traces", str(trace_a), str(trace_b)]) == 0
    cmp_payload = json.loads(capsys.readouterr().out)
    assert cmp_payload["status"] == "ok" and cmp_payload["compared"] == 1

    watch_file = tmp_path / "watch.json"
    watch_file.write_text(json.dumps({"records": [{"epoch": 0}], "latch": {"latch_triggered": True}}), encoding="utf-8")
    assert cli.main(["precision", "watch", str(watch_file), "--latch"]) == 0
    assert json.loads(capsys.readouterr().out)["latch_triggered"] is True

    # Non-leading expert dimension must be rejected explicitly
    source2, output2 = tmp_path / "in2", tmp_path / "out2"
    source2.mkdir()
    _write_safetensors_PurePy(
        source2 / "model.safetensors",
        {"layers.0.mlp.experts.weight": ("F32", (2, 4), f32_pack(range(8)))},
    )
    (source2 / "config.json").write_text(json.dumps({"n_routed_experts": 4}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-leading dimension"):
        slim_moe_checkpoint(source2, output2, 2)


def test_storage_graph_nested_alias_restore_and_reconstruct():
    torch = pytest.importorskip("torch")
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    even, odd = base[:, 0::2], base[:, 1::2]
    nested = Nested(value=base, name="n", extra=PointPair(even, odd))
    args = (base, even, odd, nested)
    kwargs = {"alias": base, "items": [even, {"odd": odd}]}
    snapshot = capture_state_snapshot(args=args, kwargs=kwargs, outputs=base[:, 2:5])
    assert len(snapshot.storages) == 1
    before_ptr = base.data_ptr()
    base.add_(1000)
    nested.value.add_(1)
    assert (
        restore_state_snapshot(snapshot, args=args, kwargs=kwargs, outputs=base[:, 2:5])[
            "storages_restored"
        ]
        == 1
    )
    assert base.data_ptr() == before_ptr
    assert torch.equal(base, torch.arange(24, dtype=torch.float32).reshape(4, 6))
    rebuilt = reconstruct_from_snapshot(snapshot)
    rbase, reven, rodd, rnested = rebuilt["args"]
    assert rbase.untyped_storage()._cdata == reven.untyped_storage()._cdata
    assert reven.untyped_storage()._cdata == rodd.untyped_storage()._cdata
    assert rnested.value.untyped_storage()._cdata == rbase.untyped_storage()._cdata
    assert rnested.extra.left.untyped_storage()._cdata == rbase.untyped_storage()._cdata
    assert torch.equal(reven, even)
    assert rodd.stride() == odd.stride() and rodd.storage_offset() == odd.storage_offset()


def test_storage_graph_rejects_structure_change():
    torch = pytest.importorskip("torch")
    x = torch.ones(2, 3)
    snapshot = capture_state_snapshot(args=(x,))
    with pytest.raises(ValueError):
        restore_state_snapshot(snapshot, args=(torch.ones(3, 2),))
    with pytest.raises(ValueError):
        restore_state_snapshot(snapshot, args=(x, x))
    with pytest.raises(ValueError):
        restore_state_snapshot(snapshot, args=None)


def test_storage_graph_nonpersistent_buffer_and_params():
    torch = pytest.importorskip("torch")

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.register_buffer("persistent", torch.ones(1))
            self.register_buffer("ephemeral", torch.ones(1), persistent=False)

        def forward(self, x):
            with torch.no_grad():
                x.add_(1)
                self.weight.add_(1)
                self.persistent.add_(1)
                self.ephemeral.add_(1)
            return x

    module = Module()
    x = torch.zeros(1, 2)
    pre = capture_state_snapshot(args=(x,), module=module)
    module(x)
    assert module.ephemeral.item() == 2
    assert restore_state_snapshot(pre, args=(x,), module=module)["tensors_checked"] == 4
    assert module.ephemeral.item() == 1
    assert module.weight.data.equal(torch.ones(2, 2))
    assert x.equal(torch.zeros(1, 2))


def test_watch_ring_masking_wrap_and_latch():
    torch = pytest.importorskip("torch")
    ring = WatchRingBuffer(capacity=2, max_snapshot_numel=8, device="cpu")
    for step in range(4):
        ring.advance_step()
        values = torch.tensor(
            [[float(step), 2.0], [float(step) + 1.0, 3.0], [float("nan"), float("inf")]]
        )
        ring.capture_pre(values)
        valid_len = 1 if step % 2 == 0 else 2
        ring.watch(values, valid_length=valid_len)
    records = ring.read_records()["records"]
    assert [record["epoch"] for record in records] == [2, 3]
    assert all(record["nan_count"] == 0 and record["inf_count"] == 0 for record in records)
    assert [record["valid_count"] for record in records] == [2, 4]
    latch = ring.read_latch()
    assert latch["latch_triggered"] is False

    ring2 = WatchRingBuffer(capacity=2, max_snapshot_numel=8, device="cpu")
    ring2.advance_step()
    ring2.capture_pre(torch.tensor([1.0, 2.0]))
    ring2.watch(torch.tensor([1.0, float("nan")]), latch_threshold=None)
    first = ring2.read_latch()
    assert first["latch_triggered"] and first["snapshot_complete"]
    assert first["pre_snapshot"] == [1.0, 2.0]
    assert first["latched_slot"] == 0
    ring2.advance_step()
    ring2.capture_pre(torch.tensor([9.0, 9.0]))
    ring2.watch(torch.tensor([9.0, 9.0]), latch_threshold=0.1)
    second = ring2.read_latch()
    assert second["pre_snapshot"] == first["pre_snapshot"]
    assert torch.allclose(
        torch.tensor(second["post_snapshot"]),
        torch.tensor(first["post_snapshot"]),
        equal_nan=True,
    )
    assert second["latched_slot"] == first["latched_slot"]


def test_watch_empty_mask_and_zero_sync(monkeypatch):
    torch = pytest.importorskip("torch")
    ring = WatchRingBuffer(capacity=2, max_snapshot_numel=4, device="cpu")
    ring.advance_step()
    ring.watch(
        torch.tensor([[1.0, 2.0], [float("nan"), 4.0]]),
        valid_mask=torch.tensor([[False, False], [False, False]]),
    )
    record = ring.read_records()["records"][0]
    assert record["valid_count"] == 0
    assert record["min"] == 0.0 and record["max"] == 0.0 and record["nan_count"] == 0

    def forbidden(*args, **kwargs):
        raise AssertionError("watch must not synchronize to host")

    ring.advance_step()
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    ring.watch(torch.tensor([[1.0, 2.0], [float("nan"), 4.0]]), valid_length=1)
    monkeypatch.undo()
    assert ring.read_records()["records"][-1]["valid_count"] == 2
    assert ring.read_records()["records"][-1]["nan_count"] == 0


def test_memory_watchpoint_is_stride_exact():
    torch = pytest.importorskip("torch")
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    even, odd = base[:, 0::2], base[:, 1::2]
    assert not views_overlap(even, odd)
    assert not MemoryWatchpoint.views_overlap(even, odd)
    assert views_overlap(base[:, 0:3], base[:, 2:5])
    transposed = base.t()
    assert views_overlap(transposed, base)
    assert not views_overlap(base[0:2], base[2:4])
    pointer = MemoryWatchpoint()
    pointer.watch("even", even)
    assert pointer.check(step=1, module_name="m")["changed"] is False
    base[0, 1] = 999.0  # odd column, must not be reported for even
    assert pointer.check(step=2, module_name="m")["changed"] is False
    base[0, 0] = 999.0
    report = pointer.check(step=3, op_name="op", active_streams=2)
    assert report["changed"] is True
    assert report["attribution_confidence"] == "coarse_interval_only"
    report_single = pointer.check(step=4, op_name="op", active_streams=1)
    assert report_single["attribution_confidence"] == "exact_single_stream_interval"


def test_lockstep_healing_inplace_none_output_and_nested_containers():
    torch = pytest.importorskip("torch")

    class InPlaceNoneMutator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.delta = 1.0
            self.register_buffer("ephemeral", torch.tensor([1.0]), persistent=False)
            self.register_buffer("persistent", torch.tensor([10.0]))

        def forward(self, x, *, scale_kw=1.0):
            with torch.no_grad():
                self.ephemeral.add_(self.delta)
                self.persistent.add_(self.delta)
                x.add_(self.delta * scale_kw)
            return None

    class DownstreamReader(torch.nn.Module):
        def forward(self, x):
            return Nested(value={"out": (x * 2.0,)}, name="reader")

    class Pipeline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mutator = InPlaceNoneMutator()
            self.reader = DownstreamReader()

        def forward(self, x):
            self.mutator(x, scale_kw=1.0)
            return self.reader(x)

    model = Pipeline()
    differ = LockstepDiffer(model, heal=True, frontier=["mutator", "reader"])
    # Three baselines give two pairwise distances, enough for jitter quantiles.
    differ.run_baseline(lambda: model(torch.tensor([1.0, 2.0])), name="A1")
    model.mutator.ephemeral.fill_(1.0)
    model.mutator.persistent.fill_(10.0)
    differ.run_baseline(lambda: model(torch.tensor([1.0, 2.0])), name="A2")
    model.mutator.ephemeral.fill_(1.0)
    model.mutator.persistent.fill_(10.0)
    differ.run_baseline(lambda: model(torch.tensor([1.0, 2.0])), name="A3")

    # Corrupt mutator in target run B (adds 100.0 instead of 1.0)
    model.mutator.delta = 100.0
    model.mutator.ephemeral.fill_(1.0)
    model.mutator.persistent.fill_(10.0)
    target_x = torch.tensor([1.0, 2.0])
    report = differ.run_target(lambda: model(target_x))
    assert report["status"] == "ok"
    assert report["jitter_available"] is True
    # Because mutator was healed in-place (even though its return value was None!),
    # target_x was restored to [2.0, 3.0], so reader saw clean baseline inputs and had 0 error!
    assert torch.equal(target_x, torch.tensor([2.0, 3.0]))
    assert model.mutator.ephemeral.item() == 2.0
    assert model.mutator.persistent.item() == 11.0
    mut_inv = next(i for i in report["invocations"] if i["module_path"] == "mutator")
    read_inv = next(i for i in report["invocations"] if i["module_path"] == "reader")
    assert mut_inv["healed"] is True
    assert mut_inv["max_abs_err"] == pytest.approx(99.0)
    assert read_inv["max_abs_err"] == 0.0
    differ.close()


def test_lockstep_repeated_calls_and_divergence():
    torch = pytest.importorskip("torch")

    class Repeat(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)

        def forward(self, x):
            return self.linear(self.linear(x))

    model = Repeat()
    differ = LockstepDiffer(model)
    differ.run_baseline(lambda: model(torch.ones(1, 2)))
    report = differ.run_target(lambda: model(torch.ones(1, 2)))
    assert report["status"] == "ok"
    ordinals = [
        item["invocation_ordinal"]
        for item in report["invocations"]
        if item["module_path"] == "linear"
    ]
    assert ordinals == [1, 2]

    class Branch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Identity()
            self.b = torch.nn.Identity()
            self.flag = True

        def forward(self, x):
            x = self.a(x)
            if self.flag:
                x = self.b(x)
            return x

    branch = Branch()
    differ2 = LockstepDiffer(branch)
    differ2.run_baseline(lambda: branch(torch.ones(1)))
    branch.flag = False
    report2 = differ2.run_target(lambda: branch(torch.ones(1)))
    assert report2["status"] == "control_flow_diverged"
    assert report2["divergence"]["reason"] == "invocation_sequence_mismatch"


def test_call_repro_resets_every_trial_and_rng(tmp_path, capsys):
    torch = pytest.importorskip("torch")

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.register_buffer("ephemeral", torch.ones(1), persistent=False)

        def forward(self, x):
            with torch.no_grad():
                x.add_(1.0)
                self.weight.add_(1.0)
                self.ephemeral.add_(1.0)
            return x * self.weight + torch.rand(2)

    module = Module()
    capsule_path = tmp_path / "call_capsule.pt"
    capsule = capture_call(module, torch.zeros(2), save_path=capsule_path)
    result = replay_call(capsule, module, repeat=5)
    assert result["status"] == "ok"
    assert result["repeat"] == 5
    # Proves trial 5 produced the exact same output as trial 1 instead of 5x accumulation!
    assert result["jitter_repeat"]["max_range"] == 0.0
    assert result["cold_start_ms"] >= 0
    assert result["warm_replay_p50_ms"] is not None
    assert result["warm_replay_p95_ms"] is not None
    assert result["reference_check"] == {
        "status": "unsupported",
        "reason": "explicit_reference_fn_required",
    }
    assert result["fakequant_check"] == {
        "status": "unsupported",
        "reason": "explicit_fakequant_contract_required",
    }

    assert cli.main(["precision", "inspect-capsule", str(capsule_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["pre"]["storages"] >= 3


def test_call_repro_reference_and_fakequant(tmp_path):
    torch = pytest.importorskip("torch")

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))

        def forward(self, x):
            x.add_(1.0)
            return x @ self.weight

    module = Module()
    path = tmp_path / "capsule.pt"
    capture_call(module, torch.ones(1, 2), save_path=path)
    reference = lambda x: x.add_(1.0) @ module.weight
    result = replay_call(path, module, repeat=2, reference_fn=reference)
    assert result["reference_check"]["status"] == "pass"
    result2 = replay_call(
        path,
        module,
        fakequant_contract=FakeQuantContract(bits=8, symmetric=True, scale_axis=0),
    )
    assert result2["fakequant_check"]["status"] == "ok"
    assert result2["fakequant_check"]["inherent_loss_status"] == "original_weights_unavailable"
    result3 = replay_call(
        path,
        module,
        fakequant_contract=FakeQuantContract(bits=8, symmetric=True, scale_axis=0),
        original_float_weight=module.weight.detach().clone(),
    )
    assert "float_vs_fakequant" in result3["fakequant_check"]


def test_det_lock_reports_initialized_backend(monkeypatch):
    torch = pytest.importorskip("torch")
    module = torch.nn.Identity()
    capsule = capture_call(module, torch.ones(2))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    result = replay_call(capsule, module, repeat=1, det_lock=True)
    assert result["det_lock"]["warning"] == "backend_already_initialized"


def test_codex_guardrail_regressions():
    torch = pytest.importorskip("torch")
    from mindie_diagnostics.precision import fake_quantize

    # 1. Tensor identity and conjugate view preservation
    x = torch.ones(2)
    rebuilt_pair = reconstruct_from_snapshot(capture_state_snapshot(args=(x, x)))["args"]
    assert rebuilt_pair[0] is rebuilt_pair[1]
    c = torch.tensor([1 + 2j]).conj()
    rebuilt_c = reconstruct_from_snapshot(capture_state_snapshot(args=(c,)))["args"][0]
    assert torch.equal(c, rebuilt_c) and rebuilt_c.is_conj()

    # 2. Masked argmax with valid -Inf must not pick invalid NaN at index 0
    ring = WatchRingBuffer(capacity=2, max_snapshot_numel=2, device="cpu")
    ring.advance_step().watch(
        torch.tensor([float("nan"), -float("inf")]),
        valid_mask=torch.tensor([False, True]),
    )
    assert ring.read_records()["records"][0]["argmax"] == 1

    # 3. Latch epoch provenance preserved across ring wraparound
    ring_wrap = WatchRingBuffer(capacity=2, max_snapshot_numel=1, device="cpu")
    for idx in range(5):
        ring_wrap.advance_step().watch(
            torch.tensor([10.0 if idx == 0 else 0.0]), latch_threshold=5.0
        )
    assert ring_wrap.read_latch()["epoch"] == 0

    # 4. External buffer physical overlap (torch.frombuffer)
    buf = bytearray(16)
    v1 = torch.frombuffer(buf, dtype=torch.float32)
    v2 = torch.frombuffer(buf, dtype=torch.float32)
    assert views_overlap(v1, v2) is True

    # 5. Autograd active must be rejected when heal=True
    lin = torch.nn.Linear(1, 1)
    req_x = torch.ones(1, 1, requires_grad=True)
    with pytest.raises(RuntimeError, match="autograd"):
        with LockstepDiffer(lin, heal=True) as d:
            d.run_baseline(lin, req_x)

    # 6. Explicit torch.Generator reset across repeat trials
    gen = torch.Generator().manual_seed(123)
    fn_gen = lambda g: torch.rand(2, generator=g)
    cap_gen = capture_call(fn_gen, gen)
    rep_gen = replay_call(cap_gen, fn_gen, repeat=4)
    assert rep_gen["jitter_repeat"]["max_range"] == 0.0

    # 7. Grouped & constant asymmetric fake_quantize
    grouped = fake_quantize(
        torch.tensor([[1.0, 2.0, 10.0, 20.0]]),
        FakeQuantContract(bits=2, symmetric=True, scale_axis=0, group_size=2),
    )
    assert grouped.tolist() == [[0.0, 2.0, 0.0, 20.0]]
    const_asym = fake_quantize(
        torch.ones(2), FakeQuantContract(bits=8, symmetric=False, scale_axis=None)
    )
    assert const_asym.tolist() == pytest.approx([1.0, 1.0])

