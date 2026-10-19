"""Evidence helpers for a real tiny vLLM online profile acceptance run.

[Role]
Provide the tiny local Llama config, exact vLLM argv, bounded process-tree
cleanup, profile artifact scanning, offline DB export from actual raw profiler
roots, runtime/source identity, and outer-budget arithmetic used by the live
online acceptance harness.  The harness itself starts the real server.

[Invariants]
- The tiny model is an explicit ``LlamaForCausalLM`` dummy-random workload;
  it is never presented as application-model correctness or performance.
- Only process groups created by the harness are signalled, with bounded
  TERM then KILL and every cleanup error preserved.
- Artifact classification uses actual filenames on disk.  A missing DB stays
  missing; the helper only invokes the installed ``torch_npu.profiler``
  ``analyse(..., export_type=Db)`` path on actual raw roots and records
  failures instead of fabricating a DB.
- The outer budget must cover readiness, start, stop, all requests, DB export,
  TERM/KILL cleanup, and overhead before the server is started.

[Timeouts & Locks]
All waits are caller-bounded.  No network server is created here; no NPU
initialization occurs at import time.  The DB export subprocess is bounded.

[Side-Effect Policy]
Writes only the caller-provided model/output/receipt paths and the actual
profiler export under those paths.  No container/image/global configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

TINY_MODEL_SCHEMA = "mindie-tiny-llama-model/1"
ARTIFACT_SCHEMA = "mindie-vllm-profile-artifacts/1"
DB_EXPORT_SCHEMA = "mindie-vllm-db-export/1"
MAX_ARTIFACT_FILES = 20_000


class OnlineEvidenceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def tiny_llama_config(
    *,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_hidden_layers: int = 1,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    vocab_size: int = 128,
    max_position_embeddings: int = 128,
    dtype: str = "float16",
) -> dict[str, Any]:
    """Return a minimal standard Llama config for local dummy-weight serving."""
    if num_attention_heads <= 0 or num_key_value_heads <= 0:
        raise ValueError("attention head counts must be positive")
    if hidden_size % num_attention_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": hidden_size // num_attention_heads,
        "hidden_act": "silu",
        "max_position_embeddings": max_position_embeddings,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": vocab_size,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "torch_dtype": dtype,
        "dtype": dtype,
    }


def write_tiny_llama_model(model_dir: str | Path, **overrides: Any) -> dict[str, Any]:
    """Write only the local config required by ``--load-format dummy``."""
    root = Path(model_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = tiny_llama_config(**overrides)
    text = json.dumps(config, indent=2, sort_keys=True) + "\n"
    path = root / "config.json"
    path.write_text(text, encoding="utf-8")
    return {
        "schema_version": TINY_MODEL_SCHEMA,
        "path": str(root.resolve()),
        "config_path": str(path.resolve()),
        "config": config,
        "config_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "weights": {
            "kind": "vllm_load_format_dummy",
            "deterministic": True,
            "source": "vllm.model_executor.model_loader.dummy_loader",
            "note": "tiny dummy-random LlamaForCausalLM; not application model performance",
        },
        "tokenizer": {
            "mode": "skip_tokenizer_init",
            "requests": "prompt token-id lists only; no remote tokenizer download",
        },
    }


def _installed_wrapper_export_expectation() -> dict[str, Any]:
    path = Path("/vllm-workspace/vllm-ascend/vllm_ascend/profiler/torch_npu_profiler.py")
    if not path.is_file():
        return {"path": str(path), "exists": False, "export_type": "unknown"}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"path": str(path), "exists": True, "export_type": "unreadable", "error": str(exc)}
    if "ExportType.Db" in text:
        export_type = "db"
    elif "ExportType.Text" in text:
        export_type = "text"
    else:
        export_type = "unknown"
    return {"path": str(path), "exists": True, "export_type": export_type}


def load_source_bundle(
    source_dir: str | Path,
    *,
    expected_commit: str | None = None,
    expected_tarball_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate an isolated vLLM source bundle without importing it.

    The bundle must contain ``BUNDLE_METADATA.json`` and the pinned
    ``_get_packed_kv_cache_groups`` symbol required by the installed
    vllm-ascend patch.  This is source/API evidence, not a version-name guess.
    """
    root = Path(source_dir).resolve()
    metadata_path = root / "BUNDLE_METADATA.json"
    if not metadata_path.is_file():
        raise OnlineEvidenceError("source_bundle_metadata_missing",
                                  f"missing source bundle metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OnlineEvidenceError("source_bundle_metadata_invalid", str(exc)) from exc
    if not isinstance(metadata, dict):
        raise OnlineEvidenceError("source_bundle_metadata_invalid", "BUNDLE_METADATA.json must be an object")
    commit = metadata.get("source_commit")
    tarball_sha = metadata.get("source_tarball_sha256")
    if expected_commit and commit != expected_commit:
        raise OnlineEvidenceError("source_commit_mismatch",
                                  f"bundle commit {commit!r} != expected {expected_commit!r}")
    if expected_tarball_sha256 and tarball_sha != expected_tarball_sha256:
        raise OnlineEvidenceError("source_tarball_mismatch",
                                  f"bundle tarball hash {tarball_sha!r} != expected {expected_tarball_sha256!r}")
    kv_cache_utils = root / "vllm" / "v1" / "core" / "kv_cache_utils.py"
    if not kv_cache_utils.is_file():
        raise OnlineEvidenceError("source_package_missing", f"missing {kv_cache_utils}")
    text = kv_cache_utils.read_text(encoding="utf-8", errors="replace")
    if "def _get_packed_kv_cache_groups(" not in text:
        raise OnlineEvidenceError(
            "source_api_incompatible",
            "vllm.v1.core.kv_cache_utils._get_packed_kv_cache_groups is absent",
        )
    required = metadata.get("required_symbols")
    if isinstance(required, dict) and any("uses_xdrope_dim" in str(key) for key in required):
        runner = root / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
        if not runner.is_file() or "uses_xdrope_dim" not in runner.read_text(encoding="utf-8", errors="replace"):
            raise OnlineEvidenceError(
                "source_api_incompatible",
                "vllm.v1.worker.gpu_model_runner.uses_xdrope_dim is absent",
            )
    native = []
    for entry in metadata.get("native_libraries", []):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
            raise OnlineEvidenceError("source_bundle_metadata_invalid", "vllm native_libraries entries need path/sha256")
        path = root / entry["path"]
        if not path.is_file():
            raise OnlineEvidenceError("native_library_missing", f"missing native library: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise OnlineEvidenceError("native_library_hash_mismatch",
                                      f"{path}: {actual} != {entry['sha256']}")
        native.append({"path": entry["path"], "sha256": actual, "size_bytes": path.stat().st_size})
    version_file = root / "vllm" / "_version.py"
    version = metadata.get("vllm_version")
    if version_file.is_file():
        text = version_file.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
        version = version or (match.group(1) if match else None)
    return {
        "schema_version": "mindie-vllm-source-bundle/1",
        "path": str(root),
        "pythonpath": str(root),
        "metadata_path": str(metadata_path),
        "source_commit": commit,
        "source_tarball_sha256": tarball_sha,
        "vllm_version": version,
        "compat_basis": metadata.get("compat_basis"),
        "required_symbol": "vllm.v1.core.kv_cache_utils._get_packed_kv_cache_groups",
        "required_symbol_present": True,
        "native_libraries": native,
        "server_launched": False,
    }


def _load_ascend_bundle(
    source_dir: str | Path,
    *,
    expected_commit: str | None = None,
    expected_tarball_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(source_dir).resolve()
    metadata_path = root / "BUNDLE_METADATA.json"
    if not metadata_path.is_file():
        raise OnlineEvidenceError("source_bundle_metadata_missing",
                                  f"missing vllm-ascend bundle metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OnlineEvidenceError("source_bundle_metadata_invalid", str(exc)) from exc
    if not isinstance(metadata, dict):
        raise OnlineEvidenceError("source_bundle_metadata_invalid", "vllm-ascend metadata must be an object")
    commit = metadata.get("source_commit")
    tarball_sha = metadata.get("source_tarball_sha256")
    if expected_commit and commit != expected_commit:
        raise OnlineEvidenceError("source_commit_mismatch",
                                  f"vllm-ascend commit {commit!r} != expected {expected_commit!r}")
    if expected_tarball_sha256 and tarball_sha != expected_tarball_sha256:
        raise OnlineEvidenceError("source_tarball_mismatch",
                                  f"vllm-ascend tarball hash {tarball_sha!r} != expected {expected_tarball_sha256!r}")
    layernorm = root / "vllm_ascend" / "ops" / "layernorm.py"
    utils = root / "vllm_ascend" / "utils.py"
    if not layernorm.is_file() or not utils.is_file():
        raise OnlineEvidenceError("source_package_missing", "vllm_ascend files missing")
    layernorm_text = layernorm.read_text(encoding="utf-8", errors="replace")
    utils_text = utils.read_text(encoding="utf-8", errors="replace")
    if "class AscendFusedRMSNormGated(" not in layernorm_text:
        raise OnlineEvidenceError("source_api_incompatible",
                                  "vllm_ascend.ops.layernorm.AscendFusedRMSNormGated is absent")
    if "AscendFusedRMSNormGated" not in utils_text:
        raise OnlineEvidenceError("source_api_incompatible",
                                  "vllm_ascend.utils does not import AscendFusedRMSNormGated")
    native = []
    for entry in metadata.get("native_libraries", []):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
            raise OnlineEvidenceError("source_bundle_metadata_invalid", "native_libraries entries need path/sha256")
        path = root / entry["path"]
        if not path.is_file():
            raise OnlineEvidenceError("native_library_missing", f"missing native library: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise OnlineEvidenceError("native_library_hash_mismatch",
                                      f"{path}: {actual} != {entry['sha256']}")
        native.append({"path": entry["path"], "sha256": actual, "size_bytes": path.stat().st_size})
    if not native:
        raise OnlineEvidenceError("source_pair_native_missing",
                                  "vllm-ascend metadata must record reused native libraries")
    version_file = root / "vllm_ascend" / "_version.py"
    version = metadata.get("vllm_ascend_version")
    if version_file.is_file():
        match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]",
                          version_file.read_text(encoding="utf-8", errors="replace"))
        version = version or (match.group(1) if match else None)
    return {
        "schema_version": "mindie-vllm-ascend-source-bundle/1",
        "path": str(root),
        "metadata_path": str(metadata_path),
        "source_commit": commit,
        "source_tarball_sha256": tarball_sha,
        "vllm_ascend_version": version,
        "compat_vllm_commit": metadata.get("compat_vllm_commit"),
        "required_symbol": "vllm_ascend.ops.layernorm.AscendFusedRMSNormGated",
        "required_symbol_present": True,
        "native_libraries": native,
        "server_launched": False,
    }


def load_source_pair(
    vllm_dir: str | Path,
    ascend_dir: str | Path,
    *,
    expected_vllm_commit: str | None = None,
    expected_vllm_tarball_sha256: str | None = None,
    expected_ascend_commit: str | None = None,
    expected_ascend_tarball_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a clean vLLM/vllm-ascend source pair by source evidence."""
    vllm = load_source_bundle(
        vllm_dir,
        expected_commit=expected_vllm_commit,
        expected_tarball_sha256=expected_vllm_tarball_sha256,
    )
    ascend = _load_ascend_bundle(
        ascend_dir,
        expected_commit=expected_ascend_commit,
        expected_tarball_sha256=expected_ascend_tarball_sha256,
    )
    if ascend.get("compat_vllm_commit") and ascend["compat_vllm_commit"] != vllm["source_commit"]:
        raise OnlineEvidenceError("source_pair_pin_mismatch",
                                  f"vllm-ascend pin {ascend['compat_vllm_commit']!r} != vllm source {vllm['source_commit']!r}")
    return {
        "schema_version": "mindie-vllm-source-pair/1",
        "vllm": vllm,
        "vllm_ascend": ascend,
        "pythonpath": [vllm["pythonpath"], ascend["path"]],
        "expected_child_module_paths": {
            "vllm": str(Path(vllm["path"]) / "vllm" / "__init__.py"),
            "vllm_ascend": str(Path(ascend["path"]) / "vllm_ascend" / "__init__.py"),
        },
        "env": {"PYTHONPATH": os.pathsep.join([vllm["pythonpath"], ascend["path"]]),
                "VLLM_VERSION": vllm.get("vllm_version") or "0.29.0"},
        "source_commits_equal_verified_pin": (
            vllm["source_commit"] == (ascend.get("compat_vllm_commit") or vllm["source_commit"])
        ),
        "server_launched": False,
        "import_verification": "static source/API validation only; full child import must be recorded by supervisor run",
    }


def runtime_identity() -> dict[str, Any]:
    """Read installed package metadata/source URLs without importing vLLM."""
    def version_of(*names: str) -> str | None:
        for name in names:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
        return None

    def source_of(*names: str) -> str | None:
        for name in names:
            try:
                distribution = metadata.distribution(name)
            except metadata.PackageNotFoundError:
                continue
            try:
                direct = json.loads(distribution.read_text("direct_url.json") or "{}")
            except (OSError, json.JSONDecodeError):
                direct = {}
            url = direct.get("url")
            if isinstance(url, str):
                return url
        return None

    return {
        "python": sys.version.split()[0],
        "hostname": os.environ.get("HOSTNAME"),
        "vllm_version": version_of("vllm"),
        "vllm_source": source_of("vllm"),
        "vllm_ascend_version": version_of("vllm_ascend", "vllm-ascend"),
        "vllm_ascend_source": source_of("vllm_ascend", "vllm-ascend"),
        "torch_version": version_of("torch"),
        "torch_npu_version": version_of("torch_npu", "torch-npu"),
        "cann_toolkit_env": os.environ.get("ASCEND_TOOLKIT_HOME"),
        "profile_wrapper": _installed_wrapper_export_expectation(),
    }


def completion_payload(
    model: str,
    token_ids: list[int],
    *,
    max_tokens: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a token-id completion request that needs no tokenizer."""
    if not token_ids or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in token_ids):
        raise ValueError("token_ids must be a non-empty list of non-negative integers")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    return {
        "model": model,
        "prompt": list(token_ids),
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": int(seed),
        "ignore_eos": True,
        "return_token_ids": True,
        "add_special_tokens": False,
        "stream": False,
    }


def validate_budget(
    *,
    total_timeout: float,
    readiness_timeout: float,
    control_timeout: float,
    request_timeout: float,
    profile_requests: int,
    db_export_timeout: float,
    cleanup_grace: float,
    cleanup_kill_timeout: float,
    max_db_export_roots: int = 1,
    overhead: float = 30.0,
    memory_phase_overhead: float = 0.0,
    pre_request_timeout: float | None = None,
) -> dict[str, Any]:
    """Require outer budget >= all sequential stages plus cleanup."""
    values = {
        "total_timeout": total_timeout,
        "readiness_timeout": readiness_timeout,
        "control_timeout": control_timeout,
        "request_timeout": request_timeout,
        "db_export_timeout": db_export_timeout,
        "cleanup_grace": cleanup_grace,
        "cleanup_kill_timeout": cleanup_kill_timeout,
        "overhead": overhead,
        "memory_phase_overhead": memory_phase_overhead,
        "pre_request_timeout": request_timeout if pre_request_timeout is None else pre_request_timeout,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise OnlineEvidenceError("invalid_budget", f"{name} must be non-negative")
        if name != "memory_phase_overhead" and value <= 0:
            raise OnlineEvidenceError("invalid_budget", f"{name} must be positive")
    if not isinstance(profile_requests, int) or isinstance(profile_requests, bool) or profile_requests <= 0:
        raise OnlineEvidenceError("invalid_budget", "profile_requests must be a positive integer")
    if not isinstance(max_db_export_roots, int) or isinstance(max_db_export_roots, bool) or max_db_export_roots <= 0:
        raise OnlineEvidenceError("invalid_budget", "max_db_export_roots must be a positive integer")
    stage_sum = (
        readiness_timeout
        + control_timeout * 2  # start + stop
        + values["pre_request_timeout"]  # first request can compile runtime kernels
        + request_timeout * (profile_requests + 2)  # during + post + post-stop health
        + db_export_timeout * max_db_export_roots
        + cleanup_grace
        + cleanup_kill_timeout
        + overhead
        + memory_phase_overhead
    )
    plan = {
        "total_timeout_s": float(total_timeout),
        "stage_sum_s": float(stage_sum),
        "margin_s": float(total_timeout) - float(stage_sum),
        "stages": {
            **{name: float(value) for name, value in values.items() if name != "total_timeout"},
            "profile_requests": profile_requests,
            "request_count": profile_requests + 3,
            "control_calls": 2,
            "max_db_export_roots": max_db_export_roots,
        },
    }
    if float(total_timeout) < float(stage_sum):
        raise OnlineEvidenceError(
            "budget_too_small",
            f"outer total_timeout {total_timeout}s < stage sum {stage_sum:.1f}s",
        )
    return plan


def build_vllm_argv(
    *,
    python: str,
    model_dir: str | Path,
    served_model_name: str,
    host: str,
    port: int,
    profiler_dir: str | Path,
    max_model_len: int = 128,
    gpu_memory_utilization: float = 0.2,
    dtype: str = "float16",
    seed: int = 0,
    profile_memory: bool = False,
    worker_cls: str | None = None,
) -> list[str]:
    """Exact installed CLI: dummy weights, no tokenizer, eager, profile hooks."""
    profiler_config = {
        "profiler": "torch",
        "torch_profiler_dir": str(Path(profiler_dir).resolve()),
        "torch_profiler_with_stack": False,
        "torch_profiler_with_memory": profile_memory,
        "torch_profiler_record_shapes": False,
        "torch_profiler_with_flops": False,
        "torch_profiler_use_gzip": False,
    }
    return [
        python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(Path(model_dir).resolve()),
        "--served-model-name", served_model_name,
        "--host", host,
        "--port", str(int(port)),
        "--load-format", "dummy",
        "--skip-tokenizer-init",
        "--enforce-eager",
        "--dtype", dtype,
        "--max-model-len", str(int(max_model_len)),
        "--max-num-seqs", "2",
        "--gpu-memory-utilization", str(float(gpu_memory_utilization)),
        "--seed", str(int(seed)),
        "--profiler-config", json.dumps(profiler_config, separators=(",", ":")),
        *(["--worker-cls", worker_cls] if worker_cls else []),
    ]


def scan_profile_artifacts(root: str | Path, *, max_files: int = MAX_ARTIFACT_FILES) -> dict[str, Any]:
    """Classify actual profile artifacts; never infer a DB from text files."""
    base = Path(root)
    result: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA,
        "root": str(base),
        "artifacts": [],
        "by_kind": {},
        "db": [],
        "text": [],
        "ok": False,
        "error": None,
    }
    if not base.is_dir():
        result["error"] = "profile root does not exist"
        return result
    seen = 0
    for directory, _dirs, files in os.walk(base):
        for name in files:
            seen += 1
            if seen > max_files:
                result["error"] = f"artifact scan exceeded {max_files} files"
                return result
            if name.endswith(".db") and "ascend_pytorch_profiler" in name:
                kind = "db"
            elif name == "kernel_details.csv":
                kind = "kernel_details"
            elif name == "trace_view.json":
                kind = "trace_view"
            elif name.startswith("npu_module_mem") and name.endswith(".csv"):
                kind = "npu_module_mem"
            elif name.startswith("memory_record") and name.endswith(".csv"):
                kind = "memory_record"
            elif name.startswith("profiler_info") and name.endswith(".json"):
                kind = "profiler_info"
            else:
                continue
            path = Path(directory) / name
            try:
                size = path.stat().st_size
            except OSError as exc:
                size = None
                entry_error = str(exc)
            else:
                entry_error = None
            entry = {
                "kind": kind,
                "path": str(path),
                "relative_path": str(path.relative_to(base)),
                "size_bytes": size,
                "non_empty": bool(size and size > 0),
                "error": entry_error,
            }
            result["artifacts"].append(entry)
            result["by_kind"][kind] = result["by_kind"].get(kind, 0) + 1
            if kind == "db":
                result["db"].append(entry)
            elif kind in ("kernel_details", "trace_view"):
                result["text"].append(entry)
    result["ok"] = True
    return result


def find_profile_roots(profile_dir: str | Path, *, max_dirs: int = 5000) -> list[Path]:
    """Find actual raw ``*_ascend_pt`` roots under the configured directory."""
    base = Path(profile_dir)
    if not base.is_dir():
        return []
    roots: list[Path] = []
    walked = 0
    for directory, dirs, files in os.walk(base):
        walked += 1
        if walked > max_dirs:
            break
        path = Path(directory)
        is_root = path.name.endswith("_ascend_pt")
        if not is_root and "ASCEND_PROFILER_OUTPUT" in dirs:
            is_root = True
        if not is_root and any(name.startswith("profiler_info") and name.endswith(".json") for name in files):
            is_root = True
        if is_root:
            roots.append(path)
            dirs[:] = []
    return sorted(set(roots))


def export_db_for_roots(
    roots: list[Path],
    *,
    timeout_s: float,
    python: str | None = None,
    cwd: str | Path | None = None,
    max_output_chars: int = 8000,
) -> list[dict[str, Any]]:
    """Ask installed torch_npu to export DB from actual raw roots, bounded."""
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    code = (
        "import sys\n"
        "from torch_npu.profiler import ExportType\n"
        "from torch_npu.profiler.profiler import analyse\n"
        "analyse(sys.argv[1], export_type=ExportType.Db)\n"
    )
    receipts: list[dict[str, Any]] = []
    for root in roots:
        started = time.monotonic()
        receipt: dict[str, Any] = {
            "schema_version": DB_EXPORT_SCHEMA,
            "root": str(root),
            "command": [python or sys.executable, "-c", "<installed torch_npu analyse db>", str(root)],
            "status": "failed",
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "elapsed_s": None,
        }
        try:
            completed = subprocess.run(
                [python or sys.executable, "-c", code, str(root)],
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
                check=False,
                cwd=str(cwd) if cwd is not None else None,
                env=os.environ.copy(),
            )
            receipt.update({
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-max_output_chars:],
                "stderr_tail": completed.stderr[-max_output_chars:],
                "status": "ok" if completed.returncode == 0 else "failed",
            })
        except subprocess.TimeoutExpired as exc:
            receipt.update({
                "status": "timeout",
                "stdout_tail": (exc.stdout or "")[-max_output_chars:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-max_output_chars:] if isinstance(exc.stderr, str) else "",
            })
        except OSError as exc:
            receipt.update({"status": "spawn_failed", "stderr_tail": str(exc)})
        receipt["elapsed_s"] = time.monotonic() - started
        receipts.append(receipt)
    return receipts


def terminate_process_group(
    process: subprocess.Popen[Any],
    *,
    grace_s: float,
    kill_timeout_s: float,
) -> dict[str, Any]:
    """TERM then KILL only the process group created for this server."""
    receipt: dict[str, Any] = {
        "pid": process.pid,
        "term_sent": False,
        "kill_sent": False,
        "exit_code": process.poll(),
        "cleanup_error": None,
    }
    if process.poll() is not None:
        return receipt
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        receipt["exit_code"] = process.poll()
        return receipt
    try:
        os.killpg(pgid, signal.SIGTERM)
        receipt["term_sent"] = True
    except ProcessLookupError:
        pass
    except OSError as exc:
        receipt["cleanup_error"] = f"TERM failed: {exc}"
    try:
        receipt["exit_code"] = process.wait(timeout=grace_s)
        return receipt
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
        receipt["kill_sent"] = True
    except ProcessLookupError:
        pass
    except OSError as exc:
        receipt["cleanup_error"] = (receipt["cleanup_error"] + "; " if receipt["cleanup_error"] else "") + f"KILL failed: {exc}"
    try:
        receipt["exit_code"] = process.wait(timeout=kill_timeout_s)
    except subprocess.TimeoutExpired as exc:
        receipt["cleanup_error"] = (receipt["cleanup_error"] + "; " if receipt["cleanup_error"] else "") + f"wait after KILL failed: {exc}"
    return receipt


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)


__all__ = [
    "OnlineEvidenceError",
    "build_vllm_argv",
    "completion_payload",
    "export_db_for_roots",
    "find_profile_roots",
    "load_source_bundle",
    "load_source_pair",
    "runtime_identity",
    "scan_profile_artifacts",
    "terminate_process_group",
    "tiny_llama_config",
    "validate_budget",
    "write_json_atomic",
    "write_tiny_llama_model",
]
