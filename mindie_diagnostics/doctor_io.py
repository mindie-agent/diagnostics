"""Read-only inspection helpers for the unified MindIE Doctor report.

[Role]
  Read the adapter profile, installed sibling distributions, consent and
  community authorities, endpoint aliases, SSH auth presence, GitHub CLI
  channel, and the optional reporting policy without mutating any state.

[Invariants]
  * Every filesystem read is bounded; missing optional files are normal states.
  * Endpoint/consent documents are parsed with a strict allowlist; password or
    credential values are never copied into a caller-visible structure.
  * No sibling package import is required.  `mindie_diagnostics` stays 100%
    stdlib at runtime.

[Timeouts & Locks]
  * `gh` is capped at 2.5s and SSH at 8.0s; both receive the remaining outer
    monotonic deadline (at least 0.5s).  No lock is acquired.

[Side-Effect Policy]
  * No persistent filesystem mutation.  The optional `gh` check uses a
    disposable temp dir removed before return; SSH uses `BatchMode=yes`, a null
    known-hosts file, and no control socket.
"""
from __future__ import annotations

import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any

CHOICES = {"contribute", "local-only", "read-only", "later", "disabled"}
PACKAGES = (
    "mindie-harness", "mindie-diagnostics", "mindie-knowledge", "remote-dev",
    "mindie-debugger", "mindie-profiler", "mindie-sim", "mindie-coordinator", "npu-top",
)
JSON_LIMIT = 262144
GH_LIMIT = 2.5
SSH_LIMIT = 8.0


def absolute(value: str | Path) -> Path:
    path = Path(os.path.expanduser(os.fspath(value)))
    if not path.is_absolute():
        path = Path.cwd() / path
    return Path(os.path.abspath(path))


def text(data: Any, *keys: str, limit: int = 4096) -> str | None:
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value and len(value) <= limit and not any(ord(ch) < 32 for ch in value):
                return value
    return None


def read_json(path: Path) -> tuple[str, dict[str, Any] | None]:
    """Return (ok|missing|unreadable|corrupt, object) using only bounded reads."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    except FileNotFoundError:
        return "missing", None
    except TimeoutError:
        raise
    except OSError:
        return "unreadable", None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > JSON_LIMIT:
            return "corrupt", None
        raw = os.read(fd, JSON_LIMIT + 1)
        if len(raw) > JSON_LIMIT:
            return "corrupt", None
    except TimeoutError:
        raise
    except OSError:
        return "unreadable", None
    finally:
        os.close(fd)
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return "corrupt", None
    return ("ok", data) if isinstance(data, dict) else ("corrupt", None)


def config_home() -> Path:
    explicit = os.environ.get("MINDIE_CONFIG_HOME")
    if explicit:
        return absolute(explicit)
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        return (absolute(appdata) if appdata else Path.home() / "AppData" / "Roaming") / "mindie-agent"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return absolute(xdg) / "mindie-agent" if xdg else Path.home() / ".config" / "mindie-agent"


def adapter(adapter_config: str | Path | None) -> tuple[Path | None, dict[str, Any] | None, str]:
    raw = adapter_config or os.environ.get("MINDIE_AGENT_CONFIG")
    if not raw:
        return None, None, "unconfigured"
    path = absolute(raw)
    state, data = read_json(path)
    return path, data if state == "ok" else None, state


def path_from(data: Any, key: str, base: Path) -> Path | None:
    value = text(data, key)
    if value is None:
        return None
    path = Path(value)
    return absolute(path if path.is_absolute() else base / path)


def dist_revision(dist: Any) -> str | None:
    try:
        raw = dist.read_text("direct_url.json")
    except TimeoutError:
        raise
    except OSError:
        return None
    try:
        data = json.loads(raw) if raw else None
        revision = (data.get("vcs_info") or {}).get("commit_id") if isinstance(data, dict) else None
        return revision if isinstance(revision, str) and 0 < len(revision) <= 64 else None
    except (ValueError, TypeError):
        return None


def packages() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in PACKAGES:
        try:
            dist = metadata.distribution(name)
            version = dist.version if isinstance(dist.version, str) else None
        except metadata.PackageNotFoundError:
            result[name] = {"installed": False, "version": None, "revision": None}
            continue
        except TimeoutError:
            raise
        except OSError:
            result[name] = {"installed": False, "version": None, "revision": None}
            continue
        result[name] = {"installed": version is not None, "version": version,
                        "revision": dist_revision(dist)}
    return result


def runtime(adapter_path: Path | None, profile: dict[str, Any] | None, state: str) -> dict[str, Any]:
    host = text(profile, "host", "harness")
    return {
        "adapter_config": str(adapter_path) if adapter_path else None,
        "adapter_state": state,
        "host": host,
        "harness": text(profile, "harness", "host"),
        "version": text(profile, "version"),
        "native_version": text(profile, "native_version"),
        "python": text(profile, "python") or sys.executable,
        "sys_executable": sys.executable,
        "packages": packages(),
    }


def valid_consent(state: str, data: dict[str, Any] | None) -> tuple[str, dict[str, Any] | None]:
    if state != "ok":
        return state, None
    if data.get("schema") != "mindie-consent/1":
        return "corrupt", None
    choice = data.get("choice")
    if choice is not None and choice not in CHOICES:
        return "corrupt", None
    revision = data.get("choice_revision")
    if revision is not None and (
        not isinstance(revision, str) or len(revision) != 32
        or any(ch not in "0123456789abcdef" for ch in revision)
    ):
        return "corrupt", None
    return "ok", data


def roots(*documents: Any) -> list[str]:
    found: list[str] = []
    for data in documents:
        values = data.get("project_roots") if isinstance(data, dict) else None
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, str) or not item or "\x00" in item:
                continue
            path = Path(os.path.expanduser(item))
            if not path.is_absolute():
                continue
            canonical = os.path.realpath(path)
            if canonical not in found:
                found.append(canonical)
    return found


def in_scope(cwd: str | Path | None, project_roots: list[str]) -> bool:
    if not project_roots:
        return False
    target = os.path.normcase(os.path.realpath(cwd or os.getcwd()))
    for root in project_roots:
        canonical = os.path.normcase(os.path.realpath(root))
        try:
            if os.path.commonpath([target, canonical]) == canonical:
                return True
        except ValueError:
            continue
    return False


def privacy(adapter_path: Path | None, profile: dict[str, Any] | None,
            cwd: str | Path | None) -> dict[str, Any]:
    base = adapter_path.parent if adapter_path else config_home()
    community_path = path_from(profile, "community_config", base)
    explicit_community = community_path is not None
    community_path = community_path or base / "mindie-community.json"
    community_state, community = read_json(community_path)
    if community_state == "missing" and not explicit_community and base != config_home():
        fallback = config_home() / "mindie-community.json"
        if fallback != community_path:
            fallback_state, fallback_data = read_json(fallback)
            if fallback_state != "missing":
                community_path, community_state, community = fallback, fallback_state, fallback_data
    consent_path = path_from(profile, "consent_config", base)
    explicit_consent = consent_path is not None
    if consent_path is None and community_state == "ok":
        consent_path = path_from(community, "consent_config", community_path.parent)
        explicit_consent = consent_path is not None
    consent_path = consent_path or base / "mindie-consent.json"
    consent_state, consent = read_json(consent_path)
    if consent_state == "missing" and not explicit_consent and base != config_home():
        fallback = config_home() / "mindie-consent.json"
        if fallback != consent_path:
            fallback_state, fallback_data = read_json(fallback)
            if fallback_state != "missing":
                consent_path, consent_state, consent = fallback, fallback_state, fallback_data
    if community_state == "ok" and community.get("schema") != "mindie-community-config/1":
        community_state, community = "corrupt", None
    consent_state, consent = valid_consent(consent_state, consent)
    choice = text(consent, "choice") if consent else None
    if consent_state == "missing":
        tier = "unconfigured"
    elif consent_state in {"corrupt", "unreadable"}:
        tier = "corrupt"
    else:
        tier = choice if choice in CHOICES else "unconfigured"
    project_roots = roots(community, consent, profile)
    community_allows_capture = (
        community_state == "missing"
        or (community_state == "ok" and community.get("enabled") is True)
    )
    publish = (
        tier == "contribute" and community_state == "ok"
        and community.get("enabled") is True
        and not bool(community.get("local_only"))
        and all(text(community, key) for key in ("repository", "fork", "account"))
    )
    domains: list[str] = []
    if isinstance(profile, dict):
        configured = profile.get("domain_configs")
        if isinstance(configured, dict):
            domains.extend(key for key in configured if isinstance(key, str) and key)
        single = text(profile, "domain")
        if single and single not in domains:
            domains.append(single)
    return {
        "tier": tier,
        "consent_state": consent_state,
        "consent_path": str(consent_path),
        "community_config_path": str(community_path),
        "community_state": community_state,
        "community_enabled": bool(isinstance(community, dict) and community.get("enabled") is True),
        "local_capture_enabled": tier in {"local-only", "contribute"} and bool(project_roots) and community_allows_capture,
        "community_publish_enabled": bool(publish),
        "project_roots": project_roots,
        "cwd_in_scope": in_scope(cwd, project_roots),
        "registered_domains": domains[:32],
    }


def endpoint(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    host = text(entry, "host", limit=255)
    user = text(entry, "user", limit=255) or "root"
    port = entry.get("port")
    if isinstance(port, str) and port.isdigit():
        port = int(port)
    if (
        host is None or host.startswith("-") or user.startswith("-")
        or isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535
    ):
        return None
    out: dict[str, Any] = {
        "host": host, "port": port, "user": user,
        "container": text(entry, "container", limit=255),
        "auth_ref": text(entry, "auth_ref", limit=255),
    }
    ident = text(entry, "identity_file", limit=1024)
    if ident:
        out["identity_file"] = ident
    return out


def ssh_auth_status() -> str:
    raw = os.environ.get("REMOTE_DEV_AUTH_DIR")
    auth_dir = absolute(raw) if raw else Path.home() / ".mindie" / "ssh"
    if (auth_dir / "id_ed25519.pub").is_file():
        return "key_present"
    if (auth_dir / "credentials.json").is_file():
        return "credentials_present"
    return "unconfigured"


def endpoints() -> dict[str, Any]:
    configured = os.environ.get("REMOTE_DEV_ENDPOINTS_FILE")
    if configured:
        paths = [absolute(item.strip()) for item in configured.split(os.pathsep) if item.strip()]
    else:
        paths = [Path.home() / ".mindie" / "endpoints.json",
                 Path.home() / ".mindie-coordinator" / "endpoints.json"]
    entries: dict[str, Any] = {}
    default_alias: str | None = None
    status, errors = "missing", []
    for path in paths:
        state, data = read_json(path)
        if state == "missing":
            continue
        if state != "ok":
            status = state
            errors.append({"path": str(path), "status": state})
            continue
        if status == "missing":
            status = "ok"
        raw_default = text(data, "default_alias", limit=255)
        if raw_default:
            default_alias = raw_default
        group = data.get("endpoints") if isinstance(data.get("endpoints"), dict) else data
        if not isinstance(group, dict):
            status = "corrupt"
            errors.append({"path": str(path), "status": "corrupt"})
            continue
        for alias, value in group.items():
            if alias != "default_alias" and isinstance(alias, str) and isinstance(value, dict):
                entries[alias] = value
    if default_alias is None and "default" in entries:
        default_alias = "default"
    if default_alias is None and len(entries) == 1:
        default_alias = next(iter(entries))
    return {"configured": bool(entries), "status": status, "default_alias": default_alias,
            "default_endpoint": endpoint(entries.get(default_alias)) if default_alias else None,
            "ssh_auth_status": ssh_auth_status(),
            "sources": [str(path) for path in paths], "errors": errors}


def probe(selected: dict[str, Any], deadline: float) -> dict[str, Any]:
    started = time.monotonic()
    remaining = deadline - started
    if remaining <= 0:
        raise TimeoutError("doctor deadline exceeded before remote probe")
    ssh = shutil.which("ssh")
    if ssh is None:
        return {"attempted": True, "reachable": False, "error": "ssh_not_installed",
                "duration_ms": 0}
    timeout = min(SSH_LIMIT, remaining)
    null_hosts = "NUL" if os.name == "nt" else os.devnull
    auth_raw = os.environ.get("REMOTE_DEV_AUTH_DIR")
    auth_dir = absolute(auth_raw) if auth_raw else Path.home() / ".mindie" / "ssh"
    ident_raw = selected.get("identity_file")
    ident_path = absolute(ident_raw) if isinstance(ident_raw, str) and ident_raw else (auth_dir / "id_ed25519")
    argv = [ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", "StrictHostKeyChecking=no", "-o", f"UserKnownHostsFile={null_hosts}",
            "-o", "ControlMaster=no", "-o", "ControlPath=none"]
    if ident_path.is_file():
        argv.extend(["-i", str(ident_path)])
    argv.extend([
        "-p", str(selected["port"]), "--", f"{selected['user']}@{selected['host']}",
        "printf '===MINDIE_PROBE===\\n'; uname -srm; npu-smi info -l 2>/dev/null || true",
    ])
    kwargs = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, errors="replace",
                              timeout=timeout, check=False, env=dict(os.environ), **kwargs)
    except subprocess.TimeoutExpired:
        return {"attempted": True, "reachable": False, "error": "timeout",
                "duration_ms": int((time.monotonic() - started) * 1000)}
    except TimeoutError:
        raise
    except OSError:
        return {"attempted": True, "reachable": False, "error": "ssh_unavailable",
                "duration_ms": int((time.monotonic() - started) * 1000)}
    stdout = (proc.stdout if isinstance(proc.stdout, str) else "")[:8192]
    marker_idx = stdout.find("===MINDIE_PROBE===")
    body = stdout[marker_idx + len("===MINDIE_PROBE==="):].lstrip("\r\n") if marker_idx >= 0 else stdout
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    result: dict[str, Any] = {
        "attempted": True, "reachable": proc.returncode == 0,
        "transport": "ssh", "returncode": proc.returncode,
        "duration_ms": int((time.monotonic() - started) * 1000),
    }
    platform = lines[0] if lines else ""
    if platform and len(platform) <= 128 and not any(ord(ch) < 32 for ch in platform):
        result["platform"] = platform
    for ln in lines[1:]:
        if "Total Count" in ln and ":" in ln:
            tail = ln.split(":", 1)[1].strip()
            if tail.isdigit():
                result["npu_count"] = int(tail)
            break
    if not result["reachable"]:
        result["error"] = "unreachable"
    return result


def remote(probe_remote: bool, deadline: float) -> dict[str, Any]:
    report = endpoints()
    if probe_remote and report["default_endpoint"] is not None:
        report["probe"] = probe(report["default_endpoint"], deadline)
    else:
        report["probe"] = {"attempted": False}
    return report


def github(tier: str, deadline: float) -> dict[str, Any]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("doctor deadline exceeded before GitHub check")
    gh = shutil.which("gh")
    required = "required_unauthenticated" if tier == "contribute" else "optional_unconfigured"
    report: dict[str, Any] = {"gh_installed": gh is not None, "status": required, "account": None}
    if gh is None:
        return report
    timeout = min(GH_LIMIT, remaining)
    kwargs = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}
    configured = os.environ.get("GH_CONFIG_DIR")
    if configured:
        config_dir = absolute(configured)
    elif os.environ.get("XDG_CONFIG_HOME"):
        config_dir = absolute(os.environ["XDG_CONFIG_HOME"]) / "gh"
    elif os.name == "nt":
        config_dir = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "GitHub CLI"
    else:
        config_dir = Path.home() / ".config" / "gh"
    hosts_yml = config_dir / "hosts.yml"
    try:
        # Always run `gh` against a disposable GH_CONFIG_DIR copy so `gh` migrations
        # or device-id writes never mutate the user's real configuration directory.
        with tempfile.TemporaryDirectory(prefix="mindie-doctor-gh-") as scratch:
            if hosts_yml.is_file():
                shutil.copyfile(hosts_yml, Path(scratch) / "hosts.yml")
                cfg_yml = config_dir / "config.yml"
                if cfg_yml.is_file():
                    shutil.copyfile(cfg_yml, Path(scratch) / "config.yml")
            env = {**os.environ, "GH_PROMPT_DISABLED": "1", "GIT_TERMINAL_PROMPT": "0",
                   "NO_COLOR": "1", "GH_CONFIG_DIR": scratch,
                   "XDG_STATE_HOME": scratch, "XDG_CACHE_HOME": scratch}
            if os.name == "nt":
                env["LOCALAPPDATA"] = scratch
            proc = subprocess.run([gh, "api", "user", "--jq", ".login"], capture_output=True,
                                  text=True, errors="replace", timeout=timeout, check=False, env=env, **kwargs)
    except subprocess.TimeoutExpired:
        report["status"] = "unreachable"
        return report
    except TimeoutError:
        raise
    except OSError:
        return report
    login = (proc.stdout or "").strip().splitlines()
    login = login[0].strip() if login else ""
    if proc.returncode == 0 and login and len(login) <= 128 and not any(ord(ch) < 32 for ch in login):
        report.update(status="authenticated", account=login)
        return report
    stderr_lower = (proc.stderr or "").lower()
    if any(tok in stderr_lower for tok in ("dial tcp", "tls", "timed out", "connection refused", "no such host", "eof", "502", "503", "504")):
        report["status"] = "unreachable"
    return report


def diagnostics(reporting_config: str | Path | None, state_dir: str | Path | None) -> dict[str, Any]:
    from .fallback import policy_path
    path = absolute(policy_path(reporting_config))
    try:
        os.lstat(path)
    except FileNotFoundError:
        result: dict[str, Any] = {"status": "not_configured", "configuration_path": str(path)}
    except TimeoutError:
        raise
    except OSError:
        result = {"status": "configuration_unavailable", "category": "unreadable_configuration",
                  "configuration_path": str(path)}
    else:
        try:
            from .integration import reporting_status
            result = reporting_status(config=str(path))
        except TimeoutError:
            raise
        except Exception as exc:
            result = {"status": "configuration_unavailable", "category": "status_failed",
                      "error_type": type(exc).__name__}
    if state_dir is not None:
        result = {**result, "state_dir": str(absolute(state_dir))}
    return result
