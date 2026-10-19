#!/usr/bin/env python3
"""Standalone real torch_npu workload for T8.4 capture, CPU reference, and HBM.

This script is intended for the supervisor's shared HostQueue on an Ascend
container, never for the local Mac unit-test suite.  It builds a small real
MLP, checks CPU-reference numerical correctness on the same state/input,
measures an actual unprofiled benchmark, then profiles the same workload with
an explicit caller step window.

It does not start a server, mutate a container image, or write outside
``--output-dir``.  It writes ``result.json``, ``capture.json``, ``steps.json``,
and a manifest-ready run JSON under that directory.

Example HostQueue command (run from the repo root on the NPU host):

    python tests/live_framework_capture_workload.py \
        --run-id t8.4-baseline-001 --label baseline \
        --start-step 0 --end-step 3 --iterations-per-step 25 \
        --output-dir .mindie-003-wave4/t8.4-framework-baseline

No result from this script is a substitute for the supervisor's real-device
acceptance; it only produces evidence for that acceptance.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindie_profiler.collector_evidence import (
    CollectorEvidenceError,
    align_capture_to_db,
    find_capture_db,
    inspect_kernel_db_clock_range,
    write_steps_file,
)
from mindie_profiler.framework_capture import build_capture_plan, run_framework_capture, write_capture_result
from mindie_profiler.module_trace import ModuleTraceCollector
from live_framework_capture_support import (
    execute_step,
    run_unprofiled,
    write_module_trace_evidence,
)
from mindie_profiler.memory_evidence import (
    BoundedMemorySampler,
    MemoryEvidenceError,
    collect_runtime_memory_snapshot,
    load_npu_module_mem_sqlite,
    manifest_from_snapshots,
    msprof_components_to_manifest,
    parse_msmemscope_snapshot_csv,
    parse_npu_module_mem_csv,
    sampled_peak_to_snapshot,
    take_msmemscope_snapshot,
)
from mindie_profiler.memory_profile import attribute_memory
from mindie_profiler.weight_inspector import inspect_weights


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _snapshot(torch_npu, phase: str, device: int, errors: list[dict]) -> dict:
    try:
        return collect_runtime_memory_snapshot(torch_npu, device=device, phase=phase)
    except Exception as exc:
        errors.append({"phase": phase, "code": getattr(exc, "code", "snapshot_failed"),
                       "message": f"{type(exc).__name__}: {exc}"})
        return {"schema_version": "mindie-runtime-memory-snapshot/1", "phase": phase,
                "device": device, "error": f"{type(exc).__name__}: {exc}"}


def _run_memory_peak_sampling(
    torch,
    torch_npu,
    model,
    inputs,
    args,
    errors: list[dict],
    model_state_sha256: str,
    input_sha256: str,
) -> dict:
    """Run a separate, bounded memory workload with a background sampler.

    This path is never entered by the timed unprofiled/profiled benchmarks.  It
    uses the identical model state/input and records actual sample timestamps,
    cadence, coverage, and a sampled device maximum (not an absolute peak).
    """
    try:
        torch_npu.npu.synchronize()
        sampler = BoundedMemorySampler(
            torch_npu,
            device=args.device,
            interval_ms=args.memory_sampling_interval_ms,
            max_samples=args.memory_sampling_max_samples,
        )
        sampler.start()
        try:
            for _ in range(args.memory_sampling_iterations):
                execute_step(torch, torch_npu, model, inputs, 1)
            torch_npu.npu.synchronize()
        finally:
            summary = sampler.stop()
        summary["workload"] = {
            "kind": "separate_memory_workload",
            "phase": "sampled_peak",
            "iterations": args.memory_sampling_iterations,
            "timed_profile": False,
            "profiled": False,
            "model_state_sha256": model_state_sha256,
            "input_sha256": input_sha256,
            "same_model_and_input": True,
            "source_note": "background sampler reads mem_get_info/memory_*; reset_peak_memory_stats is allocator-only",
        }
        return summary
    except Exception as exc:
        errors.append({
            "code": getattr(exc, "code", "memory_peak_sampling_failed"),
            "message": f"{type(exc).__name__}: {exc}",
        })
        return {
            "schema_version": "mindie-sampled-device-memory-peak/1",
            "status": "failed",
            "phase": "sampled_peak",
            "device": args.device,
            "error": f"{type(exc).__name__}: {exc}",
            "measurement_kind": "sampled_device_peak",
            "is_peak": False,
        }


def _module_mem_evidence(capture, csv_path: str | None, errors: list[dict]) -> tuple[dict | None, dict | None]:
    """Prefer the actual capture DB NPU_MODULE_MEM; fall back to an explicit CSV."""
    db_path = find_capture_db(capture) if capture is not None else None
    if db_path is not None:
        try:
            parsed = load_npu_module_mem_sqlite(db_path)
            source = {
                "kind": "npu_module_mem_sqlite",
                "path": str(db_path),
                "sha256": parsed["source_sha256"],
                "schema_version": parsed["schema_version"],
                "rows": parsed["rows"],
                "complete": parsed["complete"],
                "unmapped_module_ids": parsed["unmapped_module_ids"],
                "timestamp_range_ns": parsed["timestamp_range_ns"],
                "nonzero_components": {
                    device: {name: values["max_bytes"] for name, values in components.items() if values["max_bytes"] > 0}
                    for device, components in parsed["devices"].items()
                },
            }
            return msprof_components_to_manifest(parsed), source
        except Exception as exc:
            errors.append({"code": getattr(exc, "code", "module_mem_db_failed"),
                           "message": f"{type(exc).__name__}: {exc}"})
    if csv_path:
        try:
            parsed = parse_npu_module_mem_csv(csv_path)
            source = {
                "kind": "npu_module_mem_csv",
                "path": csv_path,
                "rows": parsed["rows"],
                "complete": parsed["complete"],
                "unit_basis": parsed["unit_basis"],
                "invalid_rows": parsed["invalid_rows"],
            }
            return msprof_components_to_manifest(parsed), source
        except Exception as exc:
            errors.append({"code": "module_mem_parse_failed", "message": f"{type(exc).__name__}: {exc}"})
    return None, None


def _tensor_digest_update(digest: "hashlib._Hash", tensor) -> None:
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.detach().cpu().contiguous().float().numpy().tobytes())


def _state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        _tensor_digest_update(digest, tensor)
    return digest.hexdigest()


def _input_sha256(inputs) -> str:
    digest = hashlib.sha256()
    _tensor_digest_update(digest, inputs)
    return digest.hexdigest()


def _correctness(torch, cpu_model, npu_model, cpu_inputs, npu_inputs, *, atol: float, rtol: float) -> dict:
    with torch.no_grad():
        cpu_output = cpu_model(cpu_inputs)
        npu_output = npu_model(npu_inputs).detach().cpu()
    finite = bool(torch.isfinite(npu_output).all().item())
    shape_match = tuple(cpu_output.shape) == tuple(npu_output.shape)
    max_abs_error = float((cpu_output - npu_output).abs().max().item()) if shape_match else None
    passed = bool(finite and shape_match and torch.allclose(cpu_output, npu_output, atol=atol, rtol=rtol))
    return {
        "status": "passed" if passed else "failed",
        "reference_device": "cpu",
        "candidate_device": "npu",
        "same_model_state": True,
        "same_inputs": True,
        "shape_match": shape_match,
        "finite": finite,
        "max_abs_error": max_abs_error,
        "atol": atol,
        "rtol": rtol,
        "output_shape": list(npu_output.shape),
        "output_dtype": str(npu_output.dtype).replace("torch.", ""),
    }


def _save_safetensors(model, output_dir: Path, errors: list[dict]) -> dict | None:
    try:
        from safetensors.torch import save_file  # type: ignore[import-not-found]
    except ImportError:
        errors.append({"code": "safetensors_missing",
                       "message": "safetensors is not installed; P3 header evidence skipped"})
        return None
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    target = weights_dir / "model.safetensors"
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    try:
        save_file(state, str(target))
    except Exception as exc:
        errors.append({"code": "safetensors_save_failed",
                       "message": f"{type(exc).__name__}: {exc}"})
        return None
    inspected = inspect_weights(weights_dir, tp=1, dp=1)
    return inspected if inspected.get("status") in ("ok", "partial") else None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--label", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, default=3)
    parser.add_argument("--iterations-per-step", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--physical-card", type=int, default=None)
    parser.add_argument("--host", default=socket.gethostname())
    parser.add_argument("--clock-domain", default="unix_wall")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--activation", choices=("relu", "gelu"), default="relu")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--memory-only", action="store_true")
    parser.add_argument("--memory-peak-sampling", action="store_true",
                        help="run a separate bounded memory workload with a background device-memory sampler")
    parser.add_argument("--memory-sampling-interval-ms", type=float, default=2.0,
                        help="requested sampler cadence in ms (default 2.0)")
    parser.add_argument("--memory-sampling-iterations", type=int, default=10,
                        help="forward iterations in the separate memory workload (default 10)")
    parser.add_argument("--memory-sampling-max-samples", type=int, default=512,
                        help="hard sampler sample cap (default 512)")
    parser.add_argument("--kv-cache-applicable", choices=("yes", "no", "unknown"), default="no",
                        help="explicit architecture KV applicability for this MLP workload (default no)")
    parser.add_argument("--msmemscope", action="store_true",
                        help="call msmemscope.take_snapshot for each phase (opt-in)")
    parser.add_argument("--module-mem-csv", default=None,
                        help="existing msprof npu_module_mem_*.csv to merge as P0 evidence")
    parser.add_argument("--msmemscope-dump-csv", default=None,
                        help="existing msmemscope memscope_dump_*.csv to parse as evidence")
    parser.add_argument("--export-budget-s", type=float, default=600.0)
    parser.add_argument("--trace-leaf-modules", action="store_true",
                        help="opt-in actual PyTorch leaf-module trace with NPU sync at module boundaries")
    parser.add_argument("--module-trace-output", default=None,
                        help="module trace JSON path (default: <output-dir>/module_trace.json)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.iterations_per_step <= 0:
        raise SystemExit("--iterations-per-step must be positive")
    if args.memory_sampling_iterations <= 0:
        raise SystemExit("--memory-sampling-iterations must be positive")
    if args.memory_sampling_interval_ms <= 0:
        raise SystemExit("--memory-sampling-interval-ms must be positive")
    if args.memory_sampling_max_samples <= 1:
        raise SystemExit("--memory-sampling-max-samples must be > 1")
    build_capture_plan(args.start_step, args.end_step)
    active_step_ids = list(range(args.start_step, args.end_step))
    output_dir = (Path(args.output_dir) if args.output_dir else ROOT / ".mindie-003-wave4" / f"t8.4-{args.label}-{_utc()}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    physical_card = args.device if args.physical_card is None else args.physical_card
    result: dict = {
        "schema_version": "mindie-live-framework-workload/1",
        "status": "failed",
        "run_id": args.run_id,
        "label": args.label,
        "host": args.host,
        "physical_card": physical_card,
        "started_at": time.time(),
        "output_dir": str(output_dir),
        "caller_window": {"start_step": args.start_step, "end_step": args.end_step},
        "iterations_per_step": args.iterations_per_step,
        "clock_domain": args.clock_domain,
        "memory_snapshots": {},
        "memory_peak_observation": None,
        "module_mem_source": None,
        "memory_acceptance": None,
        "benchmark": None,
        "capture": None,
        "steps_file": None,
        "capture_wall_window": None,
        "module_trace": None,
        "module_trace_status": "not_requested",
        "clock_alignment": None,
        "manifest": None,
        "attribution": None,
        "limitations": [],
    }
    try:
        import torch  # type: ignore[import-not-found]
        import torch_npu  # type: ignore[import-not-found]
    except ImportError as exc:
        result["error"] = {"code": "torch_npu_missing", "message": str(exc)}
        _write_json(output_dir / "result.json", result)
        return 2
    if not getattr(torch_npu.npu, "is_available", lambda: False)():
        result["error"] = {"code": "npu_unavailable", "message": "torch_npu.npu.is_available() is false"}
        _write_json(output_dir / "result.json", result)
        return 2
    torch_npu.npu.set_device(args.device)
    torch_npu.npu.matmul.allow_hf32 = False
    torch.manual_seed(7)
    errors: list[dict] = []
    result["environment"] = {
        "torch_version": torch.__version__,
        "torch_npu_version": getattr(torch_npu, "__version__", None),
        "device": args.device,
        "device_name": torch_npu.npu.get_device_name(args.device),
        "python": sys.version.split()[0],
        "matmul_allow_hf32": torch_npu.npu.matmul.allow_hf32,
        "activation": args.activation,
    }

    snapshots: dict[str, dict] = {}
    snapshots["baseline"] = _snapshot(torch_npu, "baseline", args.device, errors)
    cpu_model = torch.nn.Sequential(
        torch.nn.Linear(args.hidden, args.hidden * 2),
        torch.nn.ReLU() if args.activation == "relu" else torch.nn.GELU(),
        torch.nn.Linear(args.hidden * 2, args.hidden),
    ).eval()
    result["model_modules"] = [
        {"module_id": name, "module_type": type(module).__name__, "ordinal": index}
        for index, (name, module) in enumerate(cpu_model.named_modules())
        if name and not list(module.children())
    ]
    cpu_inputs = torch.randn(args.batch, args.hidden)
    npu_model = copy.deepcopy(cpu_model).to(torch.device(f"npu:{args.device}")).eval()
    npu_inputs = cpu_inputs.to(f"npu:{args.device}")
    model_state_sha256 = _state_sha256(cpu_model)
    input_sha256 = _input_sha256(cpu_inputs)
    result["model_state_sha256"] = model_state_sha256
    result["input_sha256"] = input_sha256
    result["input_shape"] = list(cpu_inputs.shape)
    result["input_dtype"] = str(cpu_inputs.dtype).replace("torch.", "")
    result["correctness"] = _correctness(
        torch, cpu_model, npu_model, cpu_inputs, npu_inputs, atol=args.atol, rtol=args.rtol
    )
    if result["correctness"]["status"] != "passed":
        errors.append({"code": "cpu_reference_mismatch",
                       "message": "NPU output did not match the CPU reference on the same state/input"})
    for _ in range(2):
        execute_step(torch, torch_npu, npu_model, npu_inputs, 1)
    snapshots["ready"] = _snapshot(torch_npu, "ready", args.device, errors)
    if args.msmemscope:
        try:
            result["msmemscope_snapshot"] = take_msmemscope_snapshot(args.device, "mindie-ready")
        except MemoryEvidenceError as exc:
            errors.append({"code": exc.code, "message": str(exc)})

    unprofiled = run_unprofiled(
        torch, torch_npu, npu_model, npu_inputs, active_step_ids, args.iterations_per_step
    )
    result["unprofiled_benchmark"] = unprofiled

    capture = None
    module_collector = None
    raw_module_trace = None
    if not args.memory_only:
        if args.trace_leaf_modules:
            module_collector = ModuleTraceCollector(
                npu_model, sync=lambda: torch_npu.npu.synchronize()
            ).attach()
            result["module_trace_status"] = "collecting"

        def step_fn(_step: int) -> None:
            if module_collector is not None:
                module_collector.set_step(_step)
            execute_step(torch, torch_npu, npu_model, npu_inputs, args.iterations_per_step)

        try:
            capture = run_framework_capture(
                step_fn,
                start_step=args.start_step,
                end_step=args.end_step,
                output_dir=output_dir / "profile",
                profile_memory=args.profile_memory,
                export_budget_s=args.export_budget_s,
                device=args.device,
            )
        finally:
            if module_collector is not None:
                raw_module_trace = module_collector.trace()
                module_collector.remove()
        write_capture_result(output_dir / "capture.json", capture)
        result["capture"] = capture
    execute_step(torch, torch_npu, npu_model, npu_inputs, 1)
    snapshots["inference_peak"] = _snapshot(torch_npu, "inference_peak", args.device, errors)
    memory_peak = None
    if args.memory_peak_sampling:
        memory_peak = _run_memory_peak_sampling(
            torch, torch_npu, npu_model, npu_inputs, args, errors,
            model_state_sha256, input_sha256,
        )
        result["memory_peak_observation"] = memory_peak
        if memory_peak.get("status") == "ok" and memory_peak.get("is_peak") is True:
            snapshots["sampled_peak"] = sampled_peak_to_snapshot(memory_peak)
    result["memory_snapshots"] = snapshots

    weight_manifest = _save_safetensors(cpu_model, output_dir, errors)
    if args.msmemscope_dump_csv:
        try:
            parsed_msmemscope = parse_msmemscope_snapshot_csv(args.msmemscope_dump_csv)
            result["msmemscope_dump"] = {
                "path": args.msmemscope_dump_csv,
                "snapshot_count": len(parsed_msmemscope["snapshots"]),
                "complete": parsed_msmemscope["complete"],
                "invalid_rows": parsed_msmemscope["invalid_rows"],
            }
        except Exception as exc:
            errors.append({"code": "msmemscope_parse_failed", "message": f"{type(exc).__name__}: {exc}"})
    module_manifest, module_source = _module_mem_evidence(capture, args.module_mem_csv, errors)
    result["module_mem_source"] = module_source
    kv_cache_applicable = {"yes": True, "no": False, "unknown": None}[args.kv_cache_applicable]
    target_phase = (
        "sampled_peak"
        if isinstance(memory_peak, dict)
        and memory_peak.get("status") == "ok"
        and memory_peak.get("is_peak") is True
        else "inference_peak"
    )
    result["memory_target_phase_requested"] = target_phase
    memory_manifest = manifest_from_snapshots(
        snapshots, tp=1, dp=1, msprof_components_by_device=module_manifest,
        weight_manifest=weight_manifest, kv_cache_applicable=kv_cache_applicable,
    )
    result["manifest"] = memory_manifest
    result["attribution"] = attribute_memory(
        memory_manifest, residual_target_bytes=200 * 1024 * 1024,
        target_phase=target_phase, kv_cache_applicable=kv_cache_applicable,
    )

    steps_path = output_dir / "steps.json"
    steps_document = None
    if capture is not None and capture.get("status") == "ok":
        try:
            steps_document = write_steps_file(steps_path, capture, clock_domain="unix_wall")
            result["steps_file"] = str(steps_path)
            result["steps_clock_domain"] = "unix_wall"
        except CollectorEvidenceError as exc:
            errors.append({"code": exc.code, "message": str(exc)})

    clock_alignment = {
        "schema_version": "mindie-capture-db-alignment/1",
        "status": "unavailable",
        "capture_clock_domain": "unix_wall",
        "offset_applied_ns": 0,
        "offset_guessed": False,
        "reason": "no successful capture DB",
    }
    db_path = find_capture_db(capture) if capture is not None else None
    if db_path is not None:
        try:
            db_clock = inspect_kernel_db_clock_range(
                db_path, active_steps=[row for row in capture["step_records"] if row["segment"] == "active"]
            )
            clock_alignment = align_capture_to_db(capture, db_clock)
            clock_alignment["db_path"] = str(db_path)
        except CollectorEvidenceError as exc:
            clock_alignment["error"] = {"code": exc.code, "message": str(exc)}
            errors.append(clock_alignment["error"])
    result["clock_alignment"] = clock_alignment

    module_result = write_module_trace_evidence(
        raw_module_trace, db_path, steps_document, output_dir, args.module_trace_output
    )
    module_structure = module_result.get("structure")
    module_trace_path = Path(module_result["path"]) if module_result.get("path") else None
    result["module_trace_status"] = module_result["status"]
    if module_result.get("data") is not None:
        result["module_trace"] = module_result["data"]
    if module_result.get("error") is not None:
        result["module_trace_error"] = module_result["error"]
        errors.append(module_result["error"])

    active = capture["steps"]["active"] if capture and capture.get("status") == "ok" else None
    profiled = None
    if active is not None:
        profile_window = {
            "start_ts": capture["capture_wall_start_ts"],
            "end_ts": capture["capture_wall_end_ts"],
            "start_ns": capture["capture_wall_start_ns"],
            "end_ns": capture["capture_wall_end_ns"],
        }
        active_window = {
            "start_ts": capture["active_wall_start_ts"],
            "end_ts": capture["active_wall_end_ts"],
            "start_ns": capture["active_wall_start_ns"],
            "end_ns": capture["active_wall_end_ns"],
        }
        profiled = {
            "measurement_window": profile_window,
            "duration_ms": capture["phases"]["total"],
            "sample_count": active["count"],
            "timing": dict(active),
            "iterations_per_step": args.iterations_per_step,
            "active_step_ids": list(capture["plan"]["capture"]["captured_step_ids"]),
        }
    else:
        profile_window = None
        active_window = None
    result["capture_wall_window"] = profile_window
    result["profile_measurement_window"] = active_window
    benchmark = {
        "unprofiled": {
            "measurement_window": unprofiled["measurement_window"],
            "duration_ms": unprofiled["duration_ms"],
            "sample_count": unprofiled["sample_count"],
            "timing": unprofiled["timing"],
            "iterations_per_step": args.iterations_per_step,
        },
        "profiled": profiled,
        "iterations_per_step": args.iterations_per_step,
        "windows_non_overlap": (
            profiled is not None
            and profiled["measurement_window"]["start_ts"] >= unprofiled["measurement_window"]["end_ts"]
        ),
    }
    result["benchmark"] = benchmark

    runtime = {
        "soc": result["environment"]["device_name"],
        "framework": {
            "torch": result["environment"]["torch_version"],
            "torch_npu": result["environment"]["torch_npu_version"],
        },
        "cann": None,
    }
    scene = {
        "runtime": runtime,
        "device": {"card": physical_card, "host": args.host},
        "model": f"tiny-mlp-{args.hidden}-{args.activation}",
        "shape": result["input_shape"],
        "dtype": result["input_dtype"],
        "framework": "torch_npu",
        "phase": "eager_mlp",
        "tp": 1,
        "dp": 1,
        "graph_mode": "eager",
        "clock_domain": args.clock_domain,
        "input_signature": input_sha256,
    }
    missing_inputs = [
        "telemetry (C2 collector-owned)",
        "telemetry_raw (C2 collector-owned)",
        "scene.runtime.cann (not reported by this workload)",
    ]
    if module_structure is None:
        missing_inputs.extend([
            "structure.operator_sequence (module trace-owned)",
            "structure.modules (module trace-owned)",
        ])
    run_manifest = {
        "schema_version": "mindie-profile-matched-run-inputs/1",
        "run_id": args.run_id,
        "label": args.label,
        "root": str((output_dir / "profile").resolve()),
        "steps_file": result["steps_file"],
        "scene": scene,
        "structure": module_structure or {
            "source": "not_collected_by_workload",
            "event_sequence_shape": [],
            "operator_sequence": [],
            "modules": [],
            "layers": [],
        },
        "structure_file": str(module_trace_path) if module_trace_path is not None else None,
        "module_trace": str(module_trace_path) if module_trace_path is not None else None,
        "telemetry": None,
        "telemetry_raw": None,
        "physical_card": physical_card,
        "host": args.host,
        "correctness": result["correctness"],
        "benchmark": benchmark,
        "profile_measurement_window": active_window,
        "capture_wall_window": profile_window,
        "telemetry_clock_domain": args.clock_domain,
        "capture": capture,
        "clock_alignment": clock_alignment,
        "model_state_sha256": model_state_sha256,
        "input_sha256": input_sha256,
        "iterations_per_step": args.iterations_per_step,
        "missing_inputs": missing_inputs,
        "acceptance_ready": False,
    }
    result["run_manifest"] = run_manifest
    result["manifest_ready"] = bool(capture is not None and steps_document is not None and profiled)
    result["acceptance_ready"] = False

    required_snapshots = ("baseline", "ready", "inference_peak")
    memory_ok = all(
        isinstance(snapshots.get(phase), dict)
        and isinstance(snapshots[phase].get("device_memory"), dict)
        and isinstance(snapshots[phase]["device_memory"].get("used_bytes"), (int, float))
        for phase in required_snapshots
    )
    correctness_ok = result["correctness"]["status"] == "passed"
    capture_ok = bool(args.memory_only or (capture and capture.get("status") == "ok"))
    clock_ok = bool(args.memory_only or clock_alignment.get("status") == "aligned")
    benchmark_ok = bool(
        unprofiled["sample_count"] == len(active_step_ids)
        and unprofiled["timing"]["count"] == unprofiled["sample_count"]
        and (
            args.memory_only
            or (
                profiled is not None
                and profiled["sample_count"] == len(active_step_ids)
                and profiled["timing"] == active
                and benchmark["windows_non_overlap"]
            )
        )
    )
    if not memory_ok:
        errors.append({"code": "memory_phases_incomplete",
                       "message": "baseline/ready/inference_peak device memory is not all present"})
    memory_sampling_ok = bool(
        not args.memory_peak_sampling
        or (
            isinstance(memory_peak, dict)
            and memory_peak.get("status") == "ok"
            and memory_peak.get("is_peak") is True
            and int(memory_peak.get("sample_count") or 0) >= 2
            and bool(memory_peak.get("sample_window_ns"))
        )
    )
    if not memory_sampling_ok:
        errors.append({"code": "memory_peak_sampling_failed",
                       "message": "bounded memory sampler did not produce timestamped samples"})
    if not benchmark_ok:
        errors.append({"code": "benchmark_contract_incomplete",
                       "message": "unprofiled/profiled sample counts or non-overlap windows are invalid"})
    if not clock_ok:
        errors.append({"code": "kernel_db_clock_not_aligned",
                       "message": f"DB alignment status is {clock_alignment.get('status')!r}"})
    step_frequency = None
    if active is not None:
        mean_step_s = active["mean_ms"] / 1000.0
        step_frequency = {
            "target_interval_s": 0.2,
            "mean_step_s": mean_step_s,
            "meets_5hz": mean_step_s >= 0.2,
            "iterations_per_step": args.iterations_per_step,
        }
        if not step_frequency["meets_5hz"]:
            errors.append({"code": "step_interval_below_5hz",
                           "message": "increase --iterations-per-step for 0.2 s step coverage; no sleeps were added"})
    result["step_frequency"] = step_frequency
    result["correctness_ok"] = correctness_ok
    result["capture_ok"] = capture_ok
    result["memory_ok"] = memory_ok
    result["benchmark_ok"] = benchmark_ok
    result["clock_ok"] = clock_ok
    capture_contract_ok = bool(capture_ok and correctness_ok and benchmark_ok and clock_ok)
    memory_target_met = bool(result["attribution"].get("target_met"))
    attribution_device = result["attribution"]["devices"][0] if result["attribution"].get("devices") else {}
    result["memory_acceptance"] = {
        "scope": "evidence_collection_and_scope_not_residual_target_claim",
        "sampled_peak_requested": bool(args.memory_peak_sampling),
        "sampled_peak_status": memory_peak.get("status") if isinstance(memory_peak, dict) else None,
        "sampled_peak_is_peak": memory_peak.get("is_peak") if isinstance(memory_peak, dict) else None,
        "sample_count": memory_peak.get("sample_count") if isinstance(memory_peak, dict) else None,
        "sample_window_ns": memory_peak.get("sample_window_ns") if isinstance(memory_peak, dict) else None,
        "coverage_ratio": memory_peak.get("coverage_ratio") if isinstance(memory_peak, dict) else None,
        "peak_scope": attribution_device.get("target_peak_scope"),
        "continuous_peak": attribution_device.get("target_continuous_peak"),
        "absolute_peak_limited": attribution_device.get("target_absolute_peak_limited"),
        "target_phase": attribution_device.get("target_phase"),
        "target_is_peak": attribution_device.get("target_is_peak"),
        "target_met": memory_target_met,
        "residual_bytes": attribution_device.get("residual", {}).get("bytes"),
        "residual_status": attribution_device.get("residual", {}).get("status"),
        "residual_coverage": attribution_device.get("residual", {}).get("coverage"),
        "source_applicability": result["attribution"].get("source_applicability"),
        "kv_cache_applicable": kv_cache_applicable,
        "module_mem_source": module_source,
    }
    result["memory_acceptance_ok"] = memory_sampling_ok
    card_binding_ok = physical_card == args.device
    if not card_binding_ok:
        errors.append({"code": "physical_card_device_mismatch",
                       "message": f"physical_card={physical_card} but capture device={args.device}"})
    result["card_binding_ok"] = card_binding_ok
    result["capture_contract_ok"] = capture_contract_ok and card_binding_ok
    result["memory_target_met"] = memory_target_met
    result["limitations"] = errors
    result["status_scope"] = "capture_and_measurement_contract"
    if args.memory_only:
        result["status"] = "ok" if (correctness_ok and memory_ok and memory_sampling_ok) else "failed"
    elif result["capture_contract_ok"] and memory_ok:
        result["status"] = "ok"
    elif result["capture_contract_ok"] or memory_ok:
        result["status"] = "partial"
    else:
        result["status"] = "failed"
    result["completed_at"] = time.time()
    _write_json(output_dir / "run_manifest.json", run_manifest)
    _write_json(output_dir / "result.json", result)
    print(json.dumps({
        "status": result["status"],
        "run_id": args.run_id,
        "label": args.label,
        "output_dir": str(output_dir),
        "capture_status": capture.get("status") if capture else None,
        "steps_file": result["steps_file"],
        "clock_alignment": clock_alignment.get("status"),
        "attribution_status": result["attribution"].get("status"),
        "target_met": result["attribution"].get("target_met"),
        "memory_acceptance_ok": result.get("memory_acceptance_ok"),
        "memory_target_phase": result["attribution"]["devices"][0].get("target_phase") if result["attribution"].get("devices") else None,
        "memory_peak_sample_count": (memory_peak or {}).get("sample_count") if isinstance(memory_peak, dict) else None,
        "step_frequency": step_frequency,
        "limitations": errors,
    }, ensure_ascii=False))
    return 0 if result["status"] == "ok" else (2 if result["status"] == "partial" else 1)


if __name__ == "__main__":
    sys.exit(main())
