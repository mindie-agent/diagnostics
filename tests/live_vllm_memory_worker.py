"""Opt-in vLLM worker fixture for real online memory-phase observations.

This module is resolved by vLLM only when the online harness passes
``--worker-cls tests.live_vllm_memory_worker.MemoryObservingNPUWorker``.
It subclasses the exact installed ``vllm_ascend.worker.worker.NPUWorker`` and
wraps real super calls.  It does not provide or monkeypatch any vLLM API.

The fixture writes named JSON observations under
``MINDIE_ONLINE_MEMORY_PHASE_DIR``.  Each observation calls the production
``collect_runtime_memory_snapshot`` API and preserves its allocator/device
semantics.  A bounded sampler is started only when the opt-in harness sets the
sampler environment, is stopped in ``profile`` finally paths, and writes a
separate summary.  Missing APIs or query failures are recorded; zero is never
substituted.

The worker module imports vLLM/torch only when vLLM resolves it.  Local contract
tests must test the stdlib-only helper functions instead of importing this file.
"""

from __future__ import annotations

import atexit
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from mindie_profiler.memory_evidence import collect_runtime_memory_snapshot
from mindie_profiler.memory_sampler import BoundedMemorySampler
from vllm_ascend.worker.worker import NPUWorker

PHASE_SCHEMA = "mindie-online-memory-phase/1"
WORKER_SCHEMA = "mindie-online-memory-worker/1"
SAMPLER_SCHEMA = "mindie-online-memory-sampler/1"
MODEL_STATE_SCHEMA = "mindie-online-model-state/1"
REQUIRED_PHASES = (
    "after_device_init_before_model_load",
    "after_model_load",
    "after_determine_available_memory",
    "after_cache_initialization",
    "after_warmup",
    "immediately_before_profile_start",
    "immediately_after_profile_stop",
)


def _sha256_file(path: str | Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _safe_write(path: Path, value: Any) -> None:
    """Best-effort probe write; instrumentation must not abort serving."""
    try:
        _atomic_json(path, value)
    except Exception as exc:  # noqa: BLE001 - report to worker stderr, do not break vLLM
        print(f"mindie memory probe write failed: {path}: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)


def _method_origin(cls: type, name: str) -> dict[str, Any]:
    method = getattr(cls, name, None)
    code = getattr(method, "__code__", None)
    if code is None:
        return {"name": name, "available": False}
    return {
        "name": name,
        "available": True,
        "source_file": code.co_filename,
        "first_line": int(code.co_firstlineno),
        "source_sha256": _sha256_file(code.co_filename),
    }


class MemoryObservingNPUWorker(NPUWorker):
    """Real NPUWorker with read-only phase snapshots and a bounded sampler."""

    _mindie_phase_dir: Path | None = None
    _mindie_phase_seq: int = 0
    _mindie_errors: list[dict[str, Any]] = []
    _mindie_sampler: BoundedMemorySampler | None = None
    _mindie_worker_written: bool = False
    _mindie_sampler_started_at: tuple[int, int] | None = None
    _mindie_pre_sampler_snapshot: dict[str, Any] | None = None
    _mindie_sampler_reset_attempted: bool = False
    _mindie_sampler_reset_ok: bool = False
    _mindie_native_profile_start_bounds: tuple[int, int] | None = None
    _mindie_native_profile_stop_bounds: tuple[int, int] | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mindie_phase_dir = None
        self._mindie_phase_seq = 0
        self._mindie_errors = []
        self._mindie_sampler = None
        self._mindie_worker_written = False
        self._mindie_sampler_started_at = None
        self._mindie_pre_sampler_snapshot = None
        self._mindie_sampler_reset_attempted = False
        self._mindie_sampler_reset_ok = False
        self._mindie_native_profile_start_bounds = None
        self._mindie_native_profile_stop_bounds = None

    def _phase_dir(self) -> Path | None:
        if self._mindie_phase_dir is not None:
            return self._mindie_phase_dir
        raw = os.environ.get("MINDIE_ONLINE_MEMORY_PHASE_DIR")
        if not raw or os.environ.get("MINDIE_ONLINE_MEMORY_PHASE_PROBES") != "1":
            return None
        try:
            self._mindie_phase_dir = Path(raw).resolve()
            self._mindie_phase_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"mindie memory probe directory unavailable: {raw}: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            return None
        return self._mindie_phase_dir

    def _device_index(self) -> int:
        value = getattr(self, "device", 0)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        index = getattr(value, "index", None)
        return int(index) if isinstance(index, int) else 0

    def _record_error(self, phase: str, code: str, message: str) -> None:
        if len(self._mindie_errors) >= 32:
            return
        self._mindie_errors.append({
            "phase": phase,
            "code": code,
            "message": message,
            "pid": os.getpid(),
            "observed_at": time.time(),
        })
        directory = self._phase_dir()
        if directory is not None:
            _safe_write(directory / "errors.json", self._mindie_errors)

    def _write_worker_metadata(self, directory: Path) -> None:
        if self._mindie_worker_written:
            return
        self._mindie_worker_written = True
        base_source = inspect.getsourcefile(NPUWorker)
        worker_source = inspect.getsourcefile(type(self))
        payload = {
            "schema_version": WORKER_SCHEMA,
            "worker_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "base_worker_class": f"{NPUWorker.__module__}.{NPUWorker.__qualname__}",
            "base_worker_source": base_source,
            "base_worker_source_sha256": _sha256_file(base_source) if base_source else None,
            "worker_source": worker_source,
            "worker_source_sha256": _sha256_file(worker_source) if worker_source else None,
            "collector_source": inspect.getsourcefile(collect_runtime_memory_snapshot),
            "collector_source_sha256": _sha256_file(inspect.getsourcefile(collect_runtime_memory_snapshot) or ""),
            "sampler_source": inspect.getsourcefile(BoundedMemorySampler),
            "sampler_source_sha256": _sha256_file(inspect.getsourcefile(BoundedMemorySampler) or ""),
            "overridden_methods": [
                "init_device", "load_model", "determine_available_memory",
                "initialize_cache", "initialize_from_config", "compile_or_warm_up_model", "profile",
            ],
            "base_method_origins": {
                name: _method_origin(NPUWorker, name)
                for name in ("init_device", "load_model", "determine_available_memory",
                             "initialize_cache", "initialize_from_config", "compile_or_warm_up_model", "profile")
            },
            "pid": os.getpid(),
            "python": sys.version,
            "executable": sys.executable,
        }
        try:
            status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
            stat = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
            marker = os.environ.get("REMOTE_DEV_JOB_TOKEN")
            payload["process_identity"] = {
                "pid": os.getpid(), "ppid": os.getppid(), "start_ticks": stat[19],
                "namespace_pids": [int(value) for value in status.get("NSpid", "").split()],
                "cgroup": Path("/proc/self/cgroup").read_text().splitlines(),
                "job_marker_present": marker is not None,
                "job_marker_sha256": hashlib.sha256(marker.encode()).hexdigest() if marker else None,
            }
        except (OSError, ValueError, IndexError) as exc:
            payload["process_identity_error"] = f"{type(exc).__name__}: {exc}"
        _safe_write(directory / "worker.json", payload)

    def _observe(self, phase: str, *, extra: dict[str, Any] | None = None) -> dict[str, Any] | None:
        directory = self._phase_dir()
        if directory is None:
            return
        self._write_worker_metadata(directory)
        self._mindie_phase_seq += 1
        sequence = self._mindie_phase_seq
        monotonic_start_ns = time.monotonic_ns()
        wall_start_ns = time.time_ns()
        status = "ok"
        snapshot: dict[str, Any] | None = None
        error: dict[str, Any] | None = None
        try:
            import torch_npu

            snapshot = collect_runtime_memory_snapshot(
                torch_npu,
                device=self._device_index(),
                phase=phase,
                synchronize=True,
            )
        except Exception as exc:  # noqa: BLE001 - preserve every observation failure
            status = "error"
            error = {"code": "phase_snapshot_failed", "message": f"{type(exc).__name__}: {exc}"}
            self._record_error(phase, error["code"], error["message"])
        if snapshot is not None and self._mindie_sampler_reset_ok:
            snapshot["allocator_max_kind_before_scope_correction"] = snapshot.get("allocator_max_kind")
            snapshot["allocator_max_kind"] = "allocator_max_since_sampler_reset"
            snapshot["allocator_max_scope"] = "since_sampler_reset"
        monotonic_end_ns = time.monotonic_ns()
        wall_end_ns = time.time_ns()
        payload = {
            "schema_version": PHASE_SCHEMA,
            "phase": phase,
            "sequence": sequence,
            "pid": os.getpid(),
            "device": self._device_index(),
            "rank": getattr(self, "rank", None),
            "local_rank": getattr(self, "local_rank", None),
            "status": status,
            "query_bounds": {
                "monotonic_start_ns": monotonic_start_ns,
                "monotonic_end_ns": monotonic_end_ns,
                "wall_start_ns": wall_start_ns,
                "wall_end_ns": wall_end_ns,
            },
            "snapshot": snapshot,
            "error": error,
        }
        if extra:
            payload["extra"] = extra
        _safe_write(directory / f"{sequence:03d}_{phase}.json", payload)
        return payload

    def _start_sampler(self) -> None:
        directory = self._phase_dir()
        if directory is None or self._mindie_sampler is not None:
            if self._mindie_sampler is not None:
                self._record_error("profile_start", "sampler_already_started",
                                   "sampler start requested twice")
            return
        try:
            interval_ms = float(os.environ.get("MINDIE_ONLINE_MEMORY_SAMPLER_INTERVAL_MS", "100"))
            max_samples = int(os.environ.get("MINDIE_ONLINE_MEMORY_SAMPLER_MAX_SAMPLES", "2048"))
            stop_timeout_s = float(os.environ.get("MINDIE_ONLINE_MEMORY_SAMPLER_STOP_TIMEOUT_S", "2"))
            import torch_npu

            sampler = BoundedMemorySampler(
                torch_npu,
                device=self._device_index(),
                interval_ms=interval_ms,
                max_samples=max_samples,
                stop_timeout_s=stop_timeout_s,
            )
            sampler.start()
            self._mindie_sampler = sampler
            self._mindie_sampler_started_at = (time.monotonic_ns(), time.time_ns())
            atexit.register(self._stop_sampler_at_exit)
        except Exception as exc:  # noqa: BLE001 - instrumentation failure must remain visible
            self._record_error("profile_start", "sampler_start_failed",
                               f"{type(exc).__name__}: {exc}")

    def _stop_sampler_safely(self, reason: str) -> None:
        try:
            self._stop_sampler(reason)
        except Exception as exc:  # noqa: BLE001 - never skip native profile stop
            self._record_error("profile_stop", "sampler_stop_wrapper_failed",
                               f"{type(exc).__name__}: {exc}")

    def _stop_sampler(self, reason: str) -> None:
        sampler = self._mindie_sampler
        if sampler is None:
            return
        self._mindie_sampler = None
        atexit.unregister(self._stop_sampler_at_exit)
        directory = self._phase_dir()
        summary: dict[str, Any] | None = None
        error: dict[str, Any] | None = None
        try:
            summary = sampler.stop()
            reset = ((summary or {}).get("allocator_peak") or {}).get("allocator_peak_reset") or {}
            self._mindie_sampler_reset_attempted = bool(reset.get("attempted"))
            self._mindie_sampler_reset_ok = bool(reset.get("ok"))
        except Exception as exc:  # noqa: BLE001 - preserve sampler stop failure
            error = {"code": "sampler_stop_failed", "message": f"{type(exc).__name__}: {exc}"}
            self._record_error("profile_stop", error["code"], error["message"])
        if directory is not None:
            started = self._mindie_sampler_started_at
            sampler_errors = (summary or {}).get("errors") or []
            coverage_complete = bool((summary or {}).get("target_coverage_complete"))
            status = "ok" if summary is not None and not error and not sampler_errors and coverage_complete else "partial"
            if summary is None or error:
                status = "error"
            payload = {
                "schema_version": SAMPLER_SCHEMA,
                "phase": "profiled_requests",
                "reason": reason,
                "pid": os.getpid(),
                "device": self._device_index(),
                "started_monotonic_ns": started[0] if started else None,
                "started_wall_ns": started[1] if started else None,
                "stopped_monotonic_ns": time.monotonic_ns(),
                "stopped_wall_ns": time.time_ns(),
                "status": status,
                "error": error,
                "summary": summary,
                "coverage_complete": coverage_complete,
                "sampler_errors": sampler_errors,
                "pre_sampler_native_allocator": (
                    ((self._mindie_pre_sampler_snapshot or {}).get("snapshot") or {}).get("allocator")
                ),
            }
            _safe_write(directory / "sampler.json", payload)

    def _update_sampler_scope(self) -> None:
        directory = self._phase_dir()
        if directory is None:
            return
        path = directory / "sampler.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._record_error("profile_stop", "sampler_scope_read_failed",
                               f"{type(exc).__name__}: {exc}")
            return
        scope = payload.setdefault("scope", {})
        scope["native_profile_start_bounds"] = (
            list(self._mindie_native_profile_start_bounds)
            if self._mindie_native_profile_start_bounds else None
        )
        scope["native_profile_stop_bounds"] = (
            list(self._mindie_native_profile_stop_bounds)
            if self._mindie_native_profile_stop_bounds else None
        )
        scope["export_excluded"] = True
        _safe_write(path, payload)

    def _stop_sampler_at_exit(self) -> None:
        self._stop_sampler_safely("process_exit")

    def _export_model_state(self) -> None:
        if os.environ.get("MINDIE_ONLINE_EXPORT_STATE_DICT") != "1":
            return
        directory = self._phase_dir()
        if directory is None:
            return
        try:
            model = self.model_runner.get_model()
            state = model.state_dict()
            tensor_values = [
                value for value in state.values()
                if hasattr(value, "detach") and hasattr(value, "to") and hasattr(value, "numel")
            ]
            if not tensor_values:
                raise ValueError("state_dict has no tensor values")
            parameter_count = sum(int(value.numel()) for value in tensor_values)
            limit = int(os.environ.get("MINDIE_ONLINE_EXPORT_STATE_DICT_MAX_PARAMS", "1000000"))
            if parameter_count > limit:
                raise ValueError(
                    f"state_dict has {parameter_count} parameters, above tiny fixture limit {limit}"
                )
            tensors = {
                str(name): value.detach().to("cpu").contiguous()
                for name, value in state.items()
                if hasattr(value, "detach") and hasattr(value, "to")
            }
            from safetensors.torch import save_file

            state_path = directory / "model-state.safetensors"
            save_file(tensors, str(state_path))
            metadata = {
                "schema_version": MODEL_STATE_SCHEMA,
                "path": str(state_path),
                "sha256": _sha256_file(state_path),
                "size_bytes": state_path.stat().st_size,
                "tensor_count": len(tensors),
                "parameter_count": parameter_count,
                "keys": sorted(tensors),
                "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
                "weight_source": "actual_loaded_model.state_dict",
                "p3_weight_basis": "actual_loaded_weights",
                "exported_after_phase": "after_model_load",
                "cpu_copy": True,
                "device_to_cpu": True,
                "timed_request_contamination": False,
                "note": "device-to-CPU copy performed after the after_model_load snapshot, outside the timed request window",
            }
            _safe_write(directory / "model-state.json", metadata)
        except Exception as exc:  # noqa: BLE001 - report unsupported export, never invent a file
            self._record_error("after_model_load", "state_dict_export_failed",
                               f"{type(exc).__name__}: {exc}")
            _safe_write(directory / "model-state-error.json", {
                "schema_version": MODEL_STATE_SCHEMA,
                "status": "error",
                "exported_after_phase": "after_model_load",
                "error": {"code": "state_dict_export_failed", "message": f"{type(exc).__name__}: {exc}"},
                "file_created": False,
            })

    def init_device(self):
        result = super().init_device()
        self._observe("after_device_init_before_model_load")
        return result

    def load_model(self) -> None:
        super().load_model()
        self._observe("after_model_load")
        self._export_model_state()

    def determine_available_memory(self) -> int:
        result = super().determine_available_memory()
        self._observe("after_determine_available_memory")
        return result

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        super().initialize_cache(num_gpu_blocks, num_cpu_blocks)
        self._observe("after_cache_block_configuration")

    def initialize_from_config(self, kv_cache_config) -> None:
        # The installed v1 worker allocates KV tensors here; initialize_cache
        # only records block counts and does not establish allocation evidence.
        super().initialize_from_config(kv_cache_config)
        self._observe("after_cache_initialization")

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        self._observe("after_warmup")
        return result

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        if is_start:
            self._mindie_pre_sampler_snapshot = self._observe("immediately_before_profile_start")
            self._start_sampler()
            native_started = time.monotonic_ns()
            try:
                return super().profile(is_start=is_start, profile_prefix=profile_prefix)
            except Exception as exc:
                self._record_error("profile_start", "profile_start_failed",
                                   f"{type(exc).__name__}: {exc}")
                self._stop_sampler_safely("profile_start_error")
                raise
            finally:
                self._mindie_native_profile_start_bounds = (native_started, time.monotonic_ns())
        # Scope ends BEFORE native stop, because native stop performs synchronous
        # DB export/parsing. The sampler must not include that post-workload tail.
        self._stop_sampler_safely("profile_stop_before_export")
        native_stop_started = time.monotonic_ns()
        error: dict[str, Any] | None = None
        try:
            return super().profile(is_start=is_start, profile_prefix=profile_prefix)
        except Exception as exc:
            error = {"code": "profile_stop_failed", "message": f"{type(exc).__name__}: {exc}"}
            self._record_error("profile_stop", error["code"], error["message"])
            raise
        finally:
            self._mindie_native_profile_stop_bounds = (native_stop_started, time.monotonic_ns())
            self._update_sampler_scope()
            self._observe("immediately_after_profile_stop", extra={
                "profile_stop_error": error,
                "sampler_reset_peak_stats": self._mindie_sampler_reset_ok,
                "sampler_reset_attempted": self._mindie_sampler_reset_attempted,
                "export_excluded_from_sampler": True,
                "pre_sampler_native_allocator": (
                    ((self._mindie_pre_sampler_snapshot or {}).get("snapshot") or {}).get("allocator")
                ),
            })
