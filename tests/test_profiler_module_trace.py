"""Focused tests for actual module hooks and DB event containment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindie_profiler.classify import apply as classify_events
from mindie_profiler.events import Window
from mindie_profiler.module_trace import (
    ModuleTraceCollector,
    build_module_table,
    build_structure_evidence,
    map_module_trace,
)
from mindie_profiler.sources_db import DbSource, probe_db
from tests.test_profiler_core import RANK0_ID, build_run


class _Handle:
    def __init__(self, owner, hook):
        self.owner = owner
        self.hook = hook

    def remove(self):
        if self.hook in self.owner.pre_hooks:
            self.owner.pre_hooks.remove(self.hook)
        if self.hook in self.owner.post_hooks:
            self.owner.post_hooks.remove(self.hook)


class _FakeTensor:
    def __init__(self, shape, dtype="torch.float32"):
        self.shape = tuple(shape)
        self.dtype = dtype


class _FakeModule:
    def __init__(self, name, *, children=(), type_name="Linear"):
        self._name = name
        self._children = list(children)
        self._type_name = type_name
        self.pre_hooks = []
        self.post_hooks = []

    def children(self):
        return self._children

    def named_modules(self, prefix=""):
        full = f"{prefix}.{self._name}" if prefix else self._name
        yield full, self
        for child in self._children:
            yield from child.named_modules(full)

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        self.pre_hooks.append(hook)
        return _Handle(self, hook)

    def register_forward_hook(self, hook, with_kwargs=False):
        self.post_hooks.append(hook)
        return _Handle(self, hook)

    def trigger(self, inputs):
        for hook in list(self.pre_hooks):
            hook(self, (inputs,), {})
        output = _FakeTensor((8, 256))
        for hook in list(self.post_hooks):
            hook(self, (inputs,), {}, output)
        return output

    def __repr__(self):
        return f"_FakeModule({self._type_name})"


class _FakeModel(_FakeModule):
    def __init__(self, modules):
        super().__init__("", children=modules, type_name="Sequential")


class Linear(_FakeModule):
    pass


class ReLU(_FakeModule):
    pass


def _model():
    return _FakeModel([
        Linear("0"),
        ReLU("1"),
        Linear("2"),
    ])


def test_collector_records_actual_leaf_order_step_and_removes_hooks():
    model = _model()
    sync_calls = []
    collector = ModuleTraceCollector(model, sync=lambda: sync_calls.append(1))
    collector.attach()
    collector.set_step(7)
    for module in model.children():
        module.trigger(_FakeTensor((8, 256)))

    trace = collector.trace()
    assert [row["module_name"] for row in trace["records"]] == ["0", "1", "2"]
    assert [row["module_type"] for row in trace["records"]] == ["Linear", "ReLU", "Linear"]
    assert all(row["step_id"] == 7 for row in trace["records"])
    assert len(sync_calls) == 6
    assert trace["instrumentation"]["npu_synchronize"] is True
    assert trace["instrumentation"]["throughput_evidence"] is False
    assert trace["records"][0]["input_tensors"] == [{"shape": [8, 256], "dtype": "torch.float32"}]
    assert trace["records"][0]["output_tensors"] == [{"shape": [8, 256], "dtype": "torch.float32"}]

    collector.remove()
    for module in model.children():
        module.trigger(_FakeTensor((8, 256)))
    assert len(collector.records) == 3


def test_maps_schema_fixture_events_inside_synchronized_invocation(tmp_path):
    root = build_run(tmp_path)
    db_path = next((root / RANK0_ID / "ASCEND_PROFILER_OUTPUT").glob("ascend_pytorch_profiler*.db"))
    window = Window(0, 60_000, step=4, label="step_4")
    events, _stats = DbSource(probe_db(db_path)).iter_window(window)
    classify_events(events)
    assert events
    start = min(event.start_ns for event in events)
    end = max(event.end_ns for event in events)
    trace = {
        "records": [{
            "module_id": "0", "module_name": "0", "module_type": "Linear", "ordinal": 0,
            "invocation": 0, "step_id": 4,
            "wall_start_ns": start, "wall_end_ns": end,
            "monotonic_start_ns": 0, "monotonic_end_ns": 1,
            "input_tensors": [], "output_tensors": [],
        }],
        "steps": [4],
    }

    mapping = map_module_trace(trace, db_path, windows=[window])

    assert mapping["mapping"]["events_total"] == len(events)
    assert mapping["mapping"]["events_mapped"] == len(events)
    assert mapping["mapping"]["events_unmapped"] == 0
    module = mapping["modules"][0]
    assert module["module_id"] == "0"
    assert module["mapping_coverage"] == 1.0
    assert module["busy_union_ms_p50"] == pytest.approx(0.040, abs=1e-9)
    assert module["host_gap_ms_p50"] == pytest.approx(0.005, abs=1e-9)
    structure = build_structure_evidence(mapping)
    assert structure["layers"] == []
    assert structure["module_scope"] == "actual_pytorch_leaf_modules"
    assert len(structure["operator_sequence"]) == 1
    assert structure["operator_sequence"] == ["0:Linear"]
    table = build_module_table({"structure": structure}, {"structure": structure})
    assert table["status"] == "comparable"
    assert table["modules"][0]["module_type"] == "Linear"

    # Two invocation claims for the same event cannot establish either owner.
    second = {**trace["records"][0], "module_id": "1", "module_name": "1",
              "ordinal": 1, "invocation": 1}
    overlapping = {**trace, "records": [trace["records"][0], second]}
    ambiguous = map_module_trace(overlapping, db_path, windows=[window])
    assert ambiguous["mapping"]["events_mapped"] == 0
    assert ambiguous["mapping"]["events_unmapped"] == len(events)
    assert ambiguous["mapping"]["events_ambiguous"] == len(events)
    assert all(row["busy_union_ms_p50"] is None for row in ambiguous["modules"])


def test_live_workload_support_writes_module_trace_file(tmp_path):
    from tests.live_framework_capture_support import write_module_trace_evidence

    root = build_run(tmp_path / "run")
    db_path = next((root / RANK0_ID / "ASCEND_PROFILER_OUTPUT").glob("ascend_pytorch_profiler*.db"))
    window = Window(0, 60_000, step=4, label="step_4")
    events, _stats = DbSource(probe_db(db_path)).iter_window(window)
    classify_events(events)
    start = min(event.start_ns for event in events)
    end = max(event.end_ns for event in events)
    raw_trace = {
        "schema_version": "mindie-module-trace/1",
        "scope": "tiny_mlp_leaf_modules",
        "instrumentation": {"npu_synchronize": True, "throughput_evidence": False},
        "records": [{
            "module_id": "0", "module_name": "0", "module_type": "Linear", "ordinal": 0,
            "invocation": 0, "step_id": 4,
            "wall_start_ns": start, "wall_end_ns": end,
            "monotonic_start_ns": 0, "monotonic_end_ns": 1,
            "input_tensors": [], "output_tensors": [],
        }],
    }
    steps_document = {"steps": [{"step": 4, "start_ns": 0, "end_ns": 60_000}]}

    result = write_module_trace_evidence(
        raw_trace, db_path, steps_document, tmp_path / "out", None
    )

    assert result["status"] == "mapped"
    assert Path(result["path"]).is_file()
    document = json.loads(Path(result["path"]).read_text())
    assert document["structure"]["modules"][0]["module_type"] == "Linear"
    assert document["mapping_result"]["mapping"]["events_unmapped"] == 0
