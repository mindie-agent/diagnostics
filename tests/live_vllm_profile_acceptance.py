#!/usr/bin/env python3
"""Real tiny vLLM online profile acceptance harness.

This script is intended to run INSIDE one already admitted HostQueue job in the
existing A2 container.  It creates a tiny local ``LlamaForCausalLM`` config,
starts the real installed vLLM HTTP server with deterministic dummy weights and
no tokenizer, sends token-id completion requests, exercises the actual
``/start_profile`` and ``/stop_profile`` endpoints once, exercises inference
before/during/after the window, terminates only its owned process group, and
collects the actual profile artifacts.

It never creates a container, installs dependencies, initializes NPU itself,
or uses a fake HTTP server.  It is not run on the Mac unit-test suite.

Exact supervisor command (inside the pinned container):

    python tests/live_vllm_profile_acceptance.py \
        --run-id t8.4-online-001 \
        --output-dir /workspace/.mindie-003-wave4/online-001 \
        --container-name mindie-bench-a2-core-20261017 \
        --image-digest sha256:d85730f56df057d890f9f1ded8f9d42b55e746e1cd1e9e3bb6a8adf36eeb4a16 \
        --total-timeout 1800

No model-correctness or performance claim is made for this tiny dummy model.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindie_profiler.collect import ProfileController, profile_window_from_serve_args
from mindie_profiler.vllm_online_evidence import (
    OnlineEvidenceError,
    build_vllm_argv,
    completion_payload,
    export_db_for_roots,
    find_profile_roots,
    load_source_bundle,
    load_source_pair,
    runtime_identity,
    scan_profile_artifacts,
    terminate_process_group,
    validate_budget,
    write_json_atomic,
    write_tiny_llama_model,
)
from mindie_profiler.vllm_online_memory import (
    MEMORY_REQUIRED_PHASES,
    classify_online_status,
    partial_result_limitations,
    scan_memory_phase_probes,
    validate_memory_probe_options,
)

MAX_BODY = 512 * 1024


class HarnessFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _free_port(host: str = "127.0.0.1") -> int:
    probe = socket.socket()
    try:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()


def _read_tail(path: Path, limit: int = 8000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > limit:
                handle.seek(size - limit)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _http_json(method: str, url: str, payload: dict | None, timeout: float) -> dict:
    started = time.monotonic()
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_BODY + 1)
            body = raw[:MAX_BODY].decode("utf-8", errors="replace")
            parsed = None
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                pass
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "body": body,
                "json": parsed,
                "truncated": len(raw) > MAX_BODY,
                "error": None,
                "elapsed_s": time.monotonic() - started,
            }
    except urllib.error.HTTPError as exc:
        body = exc.read(MAX_BODY).decode("utf-8", errors="replace")
        parsed = None
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            pass
        return {
            "ok": False,
            "status": exc.code,
            "body": body,
            "json": parsed,
            "truncated": False,
            "error": f"HTTP {exc.code}",
            "elapsed_s": time.monotonic() - started,
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "ok": False,
            "status": None,
            "body": "",
            "json": None,
            "truncated": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.monotonic() - started,
        }


def _wait_ready(
    process: subprocess.Popen,
    base_url: str,
    served_model: str,
    *,
    timeout_s: float,
    log_paths: dict[str, str],
) -> dict:
    deadline = time.monotonic() + timeout_s
    attempts: list[dict] = []
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return {
                "ready": False,
                "attempts": attempts,
                "error": f"server exited before readiness: rc={process.returncode}",
                "stdout_tail": _read_tail(Path(log_paths["stdout"])),
                "stderr_tail": _read_tail(Path(log_paths["stderr"])),
            }
        remaining = max(0.5, min(5.0, deadline - time.monotonic()))
        health = _http_json("GET", base_url + "/health", None, remaining)
        if health["status"] == 200:
            models = _http_json("GET", base_url + "/v1/models", None, remaining)
            model_ids = []
            if isinstance(models.get("json"), dict):
                model_ids = [item.get("id") for item in models["json"].get("data", []) if isinstance(item, dict)]
            if models["status"] == 200 and served_model in model_ids:
                attempts.append({"at_s": time.monotonic(), "health": 200, "models": model_ids})
                return {"ready": True, "attempts": attempts, "error": None,
                        "stdout_tail": "", "stderr_tail": ""}
            last_error = f"models={models['status']} ids={model_ids}"
        else:
            last_error = health.get("error") or f"health={health['status']}"
        if len(attempts) < 64:
            attempts.append({"at_s": time.monotonic(), "health": health["status"],
                             "error": last_error})
        time.sleep(0.5)
    return {
        "ready": False,
        "attempts": attempts,
        "error": f"readiness timeout after {timeout_s}s: {last_error}",
        "stdout_tail": _read_tail(Path(log_paths["stdout"])),
        "stderr_tail": _read_tail(Path(log_paths["stderr"])),
    }


def _completion(base_url: str, model: str, token_ids: list[int], *, max_tokens: int, timeout: float) -> dict:
    payload = completion_payload(model, token_ids, max_tokens=max_tokens)
    response = _http_json("POST", base_url + "/v1/completions", payload, timeout)
    parsed = response.get("json")
    choice = None
    usage = None
    if isinstance(parsed, dict) and parsed.get("choices"):
        choice = parsed["choices"][0]
        usage = parsed.get("usage")
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    token_result = choice.get("token_ids") if isinstance(choice, dict) else None
    response.update({
        "request": payload,
        "choice": choice,
        "usage": usage,
        "completion_tokens": completion_tokens,
        "returned_token_ids": token_result,
        "inference_ok": bool(
            response["status"] == 200
            and isinstance(choice, dict)
            and isinstance(completion_tokens, int)
            and completion_tokens > 0
        ),
    })
    return response


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--container-name", default="mindie-bench-a2-core-20261017")
    parser.add_argument("--image-digest", default="sha256:d85730f56df057d890f9f1ded8f9d42b55e746e1cd1e9e3bb6a8adf36eeb4a16")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--served-model-name", default="tiny-llama")
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--token-ids", default="1,5,9,12")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--profile-requests", type=int, default=2)
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--readiness-timeout", type=float, default=300.0)
    parser.add_argument("--control-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--pre-request-timeout", type=float, default=60.0,
                        help="first request budget includes cold runtime compilation")
    parser.add_argument("--db-export-timeout", type=float, default=300.0)
    parser.add_argument("--max-db-export-roots", type=int, default=1)
    parser.add_argument("--cleanup-grace", type=float, default=10.0)
    parser.add_argument("--cleanup-kill-timeout", type=float, default=5.0)
    parser.add_argument("--total-timeout", type=float, default=1800.0)
    parser.add_argument("--overhead", type=float, default=30.0)
    parser.add_argument("--skip-db-export", action="store_true")
    parser.add_argument("--source-dir", default=None,
                        help="isolated vLLM source root containing BUNDLE_METADATA.json")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--source-tarball-sha256", default=None)
    parser.add_argument("--ascend-source-dir", default=None,
                        help="clean vllm-ascend source root containing BUNDLE_METADATA.json")
    parser.add_argument("--ascend-source-commit", default=None)
    parser.add_argument("--ascend-source-tarball-sha256", default=None)
    parser.add_argument("--memory-phase-probes", action="store_true",
                        help="enable the opt-in real NPUWorker memory phase observations")
    parser.add_argument("--worker-cls", default=None,
                        help="vLLM worker class; defaults to the memory fixture when probes are enabled")
    parser.add_argument("--memory-phase-interval-ms", type=float, default=100.0)
    parser.add_argument("--memory-phase-max-samples", type=int, default=8192)
    parser.add_argument("--memory-phase-stop-timeout", type=float, default=2.0)
    parser.add_argument("--export-state-dict", action="store_true",
                        help="export the tiny fixture model state after load_model (opt-in)")
    parser.add_argument("--receipt", default=None)
    return parser.parse_args(argv)


def _token_ids(text: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise HarnessFailure("invalid_token_ids", f"token ids must be integers: {exc}") from exc
    if not values or any(item < 0 for item in values):
        raise HarnessFailure("invalid_token_ids", "token ids must be a non-empty non-negative list")
    return values


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = (output_dir / "profile").resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    memory_enabled = bool(args.memory_phase_probes)
    if args.export_state_dict and not memory_enabled:
        raise HarnessFailure("invalid_memory_options", "--export-state-dict requires --memory-phase-probes")
    memory_options = None
    if memory_enabled:
        memory_options = validate_memory_probe_options(
            interval_ms=args.memory_phase_interval_ms,
            max_samples=args.memory_phase_max_samples,
            stop_timeout_s=args.memory_phase_stop_timeout,
            profile_requests=args.profile_requests,
            request_timeout=args.request_timeout,
            control_timeout=args.control_timeout,
        )
        if not args.worker_cls:
            args.worker_cls = "tests.live_vllm_memory_worker.MemoryObservingNPUWorker"
    budget = validate_budget(
        total_timeout=args.total_timeout,
        readiness_timeout=args.readiness_timeout,
        control_timeout=args.control_timeout,
        request_timeout=args.request_timeout,
        profile_requests=args.profile_requests,
        db_export_timeout=args.db_export_timeout,
        max_db_export_roots=args.max_db_export_roots,
        cleanup_grace=args.cleanup_grace,
        cleanup_kill_timeout=args.cleanup_kill_timeout,
        overhead=args.overhead,
        memory_phase_overhead=5.0 if memory_enabled else 0.0,
        pre_request_timeout=args.pre_request_timeout,
    )
    token_ids = _token_ids(args.token_ids)
    if args.max_tokens <= 0:
        raise HarnessFailure("invalid_max_tokens", "--max-tokens must be positive")
    model = write_tiny_llama_model(output_dir / "tiny-llama", max_position_embeddings=args.max_model_len)
    source_pair = None
    source_bundle = None
    if args.source_dir and args.ascend_source_dir:
        source_pair = load_source_pair(
            args.source_dir, args.ascend_source_dir,
            expected_vllm_commit=args.source_commit,
            expected_vllm_tarball_sha256=args.source_tarball_sha256,
            expected_ascend_commit=args.ascend_source_commit,
            expected_ascend_tarball_sha256=args.ascend_source_tarball_sha256,
        )
        source_bundle = source_pair["vllm"]
    elif args.source_dir:
        source_bundle = load_source_bundle(
            args.source_dir,
            expected_commit=args.source_commit,
            expected_tarball_sha256=args.source_tarball_sha256,
        )
    if any(item >= model["config"]["vocab_size"] for item in token_ids):
        raise HarnessFailure("token_id_out_of_range", "token ids must be below the tiny model vocab_size")
    if len(token_ids) + args.max_tokens > args.max_model_len:
        raise HarnessFailure("prompt_too_long", "prompt plus max_tokens exceeds --max-model-len")
    port = args.port or _free_port(args.host)
    base_url = f"http://{args.host}:{port}"
    server_stdout = output_dir / "server.stdout.log"
    server_stderr = output_dir / "server.stderr.log"
    argv = build_vllm_argv(
        python=sys.executable,
        model_dir=model["path"],
        served_model_name=args.served_model_name,
        host=args.host,
        port=port,
        profiler_dir=profile_dir,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        seed=args.seed,
        profile_memory=args.profile_memory,
        worker_cls=args.worker_cls if (memory_enabled or args.worker_cls) else None,
    )
    profiler_config = {
        "profiler": "torch",
        "torch_profiler_dir": str(profile_dir),
        "torch_profiler_with_stack": False,
        "torch_profiler_with_memory": args.profile_memory,
        "torch_profiler_record_shapes": False,
        "torch_profiler_with_flops": False,
        "torch_profiler_use_gzip": False,
    }
    injection = profile_window_from_serve_args(
        ["--profiler-config", json.dumps(profiler_config, separators=(",", ":"))],
        endpoint=base_url,
        execution_id=args.run_id,
    )
    receipt: dict = {
        "schema_version": "mindie-vllm-online-acceptance/1",
        "run_id": args.run_id,
        "status": "failed",
        "started_at": time.time(),
        "output_dir": str(output_dir),
        "container": {
            "name": args.container_name,
            "image_digest": args.image_digest,
            "hostname": os.environ.get("HOSTNAME"),
        },
        "runtime": runtime_identity(),
        "source_bundle": source_bundle,
        "source_pair": source_pair,
        "model": model,
        "claim_scope": (
            "tiny dummy-random Llama online profile-control/artifact and memory-phase collection; "
            "not application model performance"
            if memory_enabled else
            "tiny dummy-random Llama online profile-control/artifact acceptance; not application model performance"
        ),
        "budget": budget,
        "server": {
            "argv": argv,
            "base_url": base_url,
            "pid": None,
            "logs": {"stdout": str(server_stdout), "stderr": str(server_stderr)},
            "cleanup": None,
        },
        "health": None,
        "requests": {"pre": None, "during": [], "post": None},
        "control": {
            "injection": injection.to_dict(),
            "start": None,
            "stop": None,
            "stop_attempted": False,
            "stop_skipped_reason": None,
        },
        "artifacts": None,
        "db_export": {"attempted": False, "reason": None, "receipts": []},
        "checks": {},
        "limitations": [],
        "error": None,
    }
    if memory_enabled:
        receipt["memory_observation"] = {
            "enabled": True,
            "worker_cls": args.worker_cls,
            "phase_dir": str((output_dir / "memory-phases").resolve()),
            "options": memory_options,
            "export_state_dict": bool(args.export_state_dict),
            "required_phases": list(MEMORY_REQUIRED_PHASES),
            "status": "pending",
            "sampler_scope": "profile(True) boundary through before native profile(False); DB export/parse excluded",
            "scan": None,
        }
    process = None
    controller = None
    start_ok = False
    stop_attempted = False
    try:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["VLLM_LOGGING_LEVEL"] = "INFO"
        if source_pair is not None:
            existing_pythonpath = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = os.pathsep.join(source_pair["pythonpath"]) + (
                os.pathsep + existing_pythonpath if existing_pythonpath else ""
            )
            environment["VLLM_VERSION"] = str(source_pair["env"].get("VLLM_VERSION") or "0.29.0")
            receipt["server"]["env_overrides"] = {
                "PYTHONPATH": environment["PYTHONPATH"],
                "VLLM_VERSION": environment["VLLM_VERSION"],
            }
        elif source_bundle is not None:
            existing_pythonpath = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = source_bundle["pythonpath"] + (
                os.pathsep + existing_pythonpath if existing_pythonpath else ""
            )
            environment["VLLM_VERSION"] = str(source_bundle.get("vllm_version") or "0.29.0")
            receipt["server"]["env_overrides"] = {
                "PYTHONPATH": environment["PYTHONPATH"],
                "VLLM_VERSION": environment["VLLM_VERSION"],
            }
        if memory_enabled:
            phase_dir = (output_dir / "memory-phases").resolve()
            phase_dir.mkdir(parents=True, exist_ok=True)
            existing_pythonpath = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = str(ROOT) + (
                os.pathsep + existing_pythonpath if existing_pythonpath else ""
            )
            environment["MINDIE_ONLINE_MEMORY_PHASE_DIR"] = str(phase_dir)
            environment["MINDIE_ONLINE_MEMORY_PHASE_PROBES"] = "1"
            environment["MINDIE_ONLINE_MEMORY_SAMPLER_INTERVAL_MS"] = str(args.memory_phase_interval_ms)
            environment["MINDIE_ONLINE_MEMORY_SAMPLER_MAX_SAMPLES"] = str(args.memory_phase_max_samples)
            environment["MINDIE_ONLINE_MEMORY_SAMPLER_STOP_TIMEOUT_S"] = str(args.memory_phase_stop_timeout)
            environment["MINDIE_ONLINE_EXPORT_STATE_DICT"] = "1" if args.export_state_dict else "0"
            environment["MINDIE_ONLINE_EXPORT_STATE_DICT_MAX_PARAMS"] = "1000000"
            receipt["server"].setdefault("env_overrides", {}).update({
                "PYTHONPATH": environment["PYTHONPATH"],
                "MINDIE_ONLINE_MEMORY_PHASE_DIR": environment["MINDIE_ONLINE_MEMORY_PHASE_DIR"],
                "MINDIE_ONLINE_MEMORY_PHASE_PROBES": "1",
                "MINDIE_ONLINE_MEMORY_SAMPLER_INTERVAL_MS": environment["MINDIE_ONLINE_MEMORY_SAMPLER_INTERVAL_MS"],
                "MINDIE_ONLINE_MEMORY_SAMPLER_MAX_SAMPLES": environment["MINDIE_ONLINE_MEMORY_SAMPLER_MAX_SAMPLES"],
                "MINDIE_ONLINE_MEMORY_SAMPLER_STOP_TIMEOUT_S": environment["MINDIE_ONLINE_MEMORY_SAMPLER_STOP_TIMEOUT_S"],
                "MINDIE_ONLINE_EXPORT_STATE_DICT": environment["MINDIE_ONLINE_EXPORT_STATE_DICT"],
            })
        with server_stdout.open("wb") as stdout_handle, server_stderr.open("wb") as stderr_handle:
            process = subprocess.Popen(
                argv,
                cwd=str(output_dir),
                stdout=stdout_handle,
                stderr=stderr_handle,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=environment,
            )
        receipt["server"]["pid"] = process.pid
        readiness = _wait_ready(
            process, base_url, args.served_model_name,
            timeout_s=args.readiness_timeout,
            log_paths=receipt["server"]["logs"],
        )
        receipt["health"] = readiness
        if not readiness["ready"]:
            raise HarnessFailure("readiness_failed", readiness["error"] or "server not ready")

        pre = _completion(base_url, args.served_model_name, token_ids,
                          max_tokens=args.max_tokens, timeout=args.pre_request_timeout)
        receipt["requests"]["pre"] = pre
        if not pre["inference_ok"]:
            raise HarnessFailure("pre_request_failed", f"pre-profile completion failed: {pre.get('error')}")

        controller = ProfileController(
            injection,
            timeout=args.control_timeout,
            receipt_path=output_dir / "profile-control.json",
        )
        started = controller.start()
        receipt["control"]["start"] = started
        if started["status"] != "ok":
            if started["status"] == "unknown":
                receipt["control"]["stop_skipped_reason"] = "start_profile_unknown_no_replay"
            raise HarnessFailure(f"start_profile_{started['status']}",
                                 started.get("error", {}).get("message") if started.get("error") else "start_profile failed")
        start_ok = True
        try:
            for index in range(args.profile_requests):
                response = _completion(base_url, args.served_model_name, token_ids,
                                       max_tokens=args.max_tokens, timeout=args.request_timeout)
                response["profile_request_index"] = index
                receipt["requests"]["during"].append(response)
        finally:
            stop_attempted = True
            receipt["control"]["stop_attempted"] = True
            try:
                receipt["control"]["stop"] = controller.stop()
            except Exception as exc:  # noqa: BLE001 - no replay after an attempted stop
                receipt["control"]["stop_error"] = f"{type(exc).__name__}: {exc}"
        stop_result = receipt["control"].get("stop") or {}
        if stop_result.get("status") != "ok":
            raise HarnessFailure(f"stop_profile_{stop_result.get('status', 'missing')}",
                                 "stop_profile did not acknowledge")
        receipt["health"]["post_stop"] = _http_json("GET", base_url + "/health", None, args.request_timeout)
        post = _completion(base_url, args.served_model_name, token_ids,
                           max_tokens=args.max_tokens, timeout=args.request_timeout)
        receipt["requests"]["post"] = post
        if not post["inference_ok"]:
            raise HarnessFailure("post_request_failed", f"post-profile completion failed: {post.get('error')}")
    except HarnessFailure as exc:
        receipt["error"] = {"code": exc.code, "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 - preserve unexpected harness failures in the receipt
        receipt["error"] = {"code": "harness_failed", "message": f"{type(exc).__name__}: {exc}"}
    finally:
        if controller is not None and start_ok and not stop_attempted:
            stop_attempted = True
            receipt["control"]["stop_attempted"] = True
            try:
                receipt["control"]["stop"] = controller.stop()
            except Exception as exc:  # noqa: BLE001 - no replay after an attempted stop
                receipt["control"]["stop_error"] = f"{type(exc).__name__}: {exc}"
        if process is not None:
            cleanup = terminate_process_group(
                process,
                grace_s=args.cleanup_grace,
                kill_timeout_s=args.cleanup_kill_timeout,
            )
            receipt["server"]["cleanup"] = cleanup
            receipt["server"]["exit_code"] = cleanup.get("exit_code")
            receipt["server"]["stdout_tail"] = _read_tail(server_stdout)
            receipt["server"]["stderr_tail"] = _read_tail(server_stderr)

        before = scan_profile_artifacts(profile_dir)
        db_before = [item for item in before.get("db", []) if item.get("non_empty")]
        export_receipts: list[dict] = []
        export_reason = None
        if not db_before and not args.skip_db_export:
            roots = find_profile_roots(profile_dir)
            if roots:
                selected_roots = roots[:args.max_db_export_roots]
                receipt["db_export"]["roots_found"] = len(roots)
                receipt["db_export"]["roots_exported"] = len(selected_roots)
                receipt["db_export"]["roots_truncated"] = len(roots) > len(selected_roots)
                if len(roots) > len(selected_roots):
                    receipt["limitations"].append({"code": "db_export_roots_truncated",
                                                   "message": f"only {len(selected_roots)} of {len(roots)} raw roots were exported"})
                export_receipts = export_db_for_roots(
                    selected_roots,
                    timeout_s=args.db_export_timeout,
                    cwd=output_dir,
                )
                receipt["db_export"]["attempted"] = True
                receipt["db_export"]["reason"] = "db_artifact_missing_after_stop; installed wrapper exports text"
            else:
                export_reason = "no_raw_profile_root_found"
        elif db_before:
            export_reason = "db_artifact_already_present"
        else:
            export_reason = "db_export_skipped_by_caller"
        after = scan_profile_artifacts(profile_dir)
        db_after = [item for item in after.get("db", []) if item.get("non_empty")]
        receipt["artifacts"] = {"before_export": before, "after_export": after,
                                "db_nonempty": db_after}
        receipt["db_export"]["receipts"] = export_receipts
        receipt["db_export"]["reason"] = receipt["db_export"]["reason"] or export_reason

        if memory_enabled:
            try:
                memory_scan = scan_memory_phase_probes(
                    output_dir / "memory-phases",
                    require_model_state=bool(args.export_state_dict),
                )
                receipt["memory_observation"]["scan"] = memory_scan
                receipt["memory_observation"]["status"] = memory_scan["status"]
                if args.export_state_dict and memory_scan.get("model_state") is None:
                    receipt["memory_observation"]["status"] = "partial"
            except Exception as exc:  # noqa: BLE001 - keep observation failure separate
                receipt["memory_observation"]["status"] = "failed"
                receipt["memory_observation"]["scan_error"] = f"{type(exc).__name__}: {exc}"
        control_start = receipt["control"].get("start") or {}
        control_stop = receipt["control"].get("stop") or {}
        during_ok = bool(receipt["requests"]["during"]) and all(
            item.get("inference_ok") for item in receipt["requests"]["during"]
        )
        checks = {
            "server_ready": bool(receipt["health"] and receipt["health"].get("ready")),
            "pre_request": bool(receipt["requests"]["pre"] and receipt["requests"]["pre"].get("inference_ok")),
            "start_profile": control_start.get("status") == "ok",
            "during_requests": during_ok,
            "stop_profile": control_stop.get("status") == "ok",
            "post_stop_health": bool(
                (receipt["health"] or {}).get("post_stop")
                and receipt["health"]["post_stop"].get("status") == 200
            ),
            "post_request": bool(receipt["requests"]["post"] and receipt["requests"]["post"].get("inference_ok")),
            "actual_db_artifact": bool(db_after),
            "process_cleanup": bool(
                receipt["server"]["cleanup"]
                and receipt["server"]["cleanup"].get("cleanup_error") is None
                and receipt["server"]["cleanup"].get("exit_code") is not None
            ),
        }
        if memory_enabled:
            checks["memory_phase_probes"] = (
                receipt.get("memory_observation", {}).get("status") == "ok"
            )
        receipt["checks"] = checks
        receipt["status"] = classify_online_status(
            checks,
            control_start_status=control_start.get("status"),
            control_stop_status=control_stop.get("status"),
        )
        if receipt["status"] == "partial":
            receipt["limitations"].extend(partial_result_limitations(checks))
        if not checks["during_requests"]:
            receipt["limitations"].append({"code": "during_request_failed",
                                           "message": "one or more profile-window completions failed"})
        if not checks["process_cleanup"]:
            receipt["limitations"].append({"code": "cleanup_incomplete",
                                           "message": "owned server process tree did not cleanly exit"})
        if not db_after:
            receipt["limitations"].append({"code": "db_export_missing",
                                           "message": "no non-empty ascend_pytorch_profiler_*.db after DB export attempt"})
        receipt["completed_at"] = time.time()
        write_json_atomic(output_dir / "result.json", receipt)
        if args.receipt:
            write_json_atomic(args.receipt, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run(args)
    except OnlineEvidenceError as exc:
        print(json.dumps({"status": "failed", "error": {"code": exc.code, "message": str(exc)}}, ensure_ascii=False))
        return 1
    except Exception as exc:  # noqa: BLE001 - keep an executable failure receipt
        print(json.dumps({"status": "failed", "error": {"code": "harness_failed",
                                                        "message": f"{type(exc).__name__}: {exc}"}},
                         ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": result["status"],
        "run_id": result["run_id"],
        "output_dir": result["output_dir"],
        "server_pid": result["server"].get("pid"),
        "checks": result["checks"],
        "db_export": result["db_export"],
        "memory_observation": result.get("memory_observation"),
        "error": result["error"],
    }, ensure_ascii=False))
    return 0 if result["status"] == "ok" else (2 if result["status"] in ("partial", "unknown") else 1)


if __name__ == "__main__":
    sys.exit(main())
