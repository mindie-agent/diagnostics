"""Unified read-only MindIE Doctor report (`mindie-doctor/1`).

[Role]
  Inspect one local MindIE Agent installation (adapter profile, optional sibling
  distributions, consent/community authorities, endpoint aliases, SSH auth
  presence, GitHub CLI channel, and optional diagnostics) and return one
  structured, secret-free health report.

[Invariants]
  * Strictly read-only: never creates directories, JSON files, SQLite queues,
    WAL/SHM sidecars, worker health, consent documents, or background workers.
  * Consent/endpoint documents are bounded reads; password and credential
    values are never copied into the report.
  * Missing optional tools (`gh`, `ssh`, endpoint aliases, reporting policy)
    are normal states, not failures.  A corrupt authority is `degraded`.
  * No sibling package import is required; `mindie_diagnostics` remains 100%
    stdlib at runtime.

[Timeouts & Locks]
  * Outer monotonic budget: 5.0s local-only, 15.0s when probing a remote host
    (5.0s local + 8.0s remote + 2.0s cleanup).  `gh` is capped at 2.5s and SSH
    at 8.0s; both receive the remaining deadline (at least 0.5s).  No lock is
    acquired.

[Side-Effect Policy]
  * No persistent filesystem or process mutation.  The optional `gh` check
    uses a disposable temp dir removed before return; SSH uses `BatchMode=yes`,
    a null known-hosts file, and no control socket.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from . import doctor_io

SCHEMA = "mindie-doctor/1"
LOCAL_BUDGET = 5.0
REMOTE_BUDGET = 15.0


def _overall(privacy: dict[str, Any], remote: dict[str, Any], github: dict[str, Any],
             diagnostics: dict[str, Any], adapter_state: str) -> str:
    if privacy["tier"] == "corrupt" or remote["status"] in {"corrupt", "unreadable"}:
        return "degraded"
    if github["status"] == "required_unauthenticated":
        return "degraded"
    if remote["probe"].get("attempted") and not remote["probe"].get("reachable"):
        return "degraded"
    if diagnostics.get("status") == "configuration_unavailable":
        return "degraded"
    if adapter_state in {"corrupt", "unreadable"}:
        return "degraded"
    tier = privacy["tier"]
    if tier in {"unconfigured", "later"}:
        return "attention_needed"
    if tier in {"local-only", "contribute"} and not privacy["local_capture_enabled"]:
        return "attention_needed"
    if tier == "contribute" and github["status"] == "unreachable":
        return "attention_needed"
    if tier == "contribute" and not privacy["community_publish_enabled"]:
        return "attention_needed"
    return "healthy"


def _remediations(privacy: dict[str, Any], remote: dict[str, Any], github: dict[str, Any],
                  diagnostics: dict[str, Any], adapter_state: str, probe_remote: bool) -> list[str]:
    items: list[str] = []
    tier = privacy["tier"]
    if tier == "corrupt":
        items.append("Repair or remove the unreadable/corrupt consent file; doctor never rewrites it.")
    elif tier in {"unconfigured", "later"}:
        items.append("Choose a contribution tier (`local-only`, `read-only`, or `contribute`) and save it in `mindie-consent.json`.")
    if tier in {"local-only", "contribute"} and not privacy["local_capture_enabled"]:
        items.append("Add at least one absolute `project_roots` path (and ensure `mindie-community.json` is not disabled) so local capture is in scope.")
    if tier == "contribute":
        if github["status"] == "unreachable":
            items.append("GitHub CLI check timed out or was unreachable; verify network/VPN connectivity.")
        elif github["status"] != "authenticated":
            items.append("Install the GitHub CLI (`gh`) and run `gh auth login` because `contribute` requires an authenticated channel.")
        if not privacy["community_publish_enabled"]:
            items.append("Enable `mindie-community.json` (`mindie-community-config/1`) with `repository`, `fork`, and `account` to publish.")
    if remote["status"] in {"corrupt", "unreadable"}:
        items.append("Repair the endpoint alias JSON (valid object with `host` and `port`) or remove the damaged file.")
    elif probe_remote and not remote["configured"]:
        items.append("Connect an NPU host with the `remote_auth` tool, or configure an endpoint alias file.")
    if remote["probe"].get("attempted") and not remote["probe"].get("reachable"):
        items.append("Remote probe failed; continue the task with native `ssh` or local scripts while repairing the endpoint.")
    if adapter_state in {"corrupt", "unreadable"}:
        items.append("Repair or regenerate the optional adapter profile JSON.")
    if diagnostics.get("status") == "configuration_unavailable":
        items.append("Repair the reporting policy JSON (`diagnostics.json`); local diagnostics remain available without it.")
    if items and not any("native `ssh`" in item for item in items):
        items.append("If a helper tool has issues, agents can continue via native `ssh` or local scripts.")
    return items


def run_doctor(*, adapter_config: str | Path | None = None, cwd: str | Path | None = None,
               reporting_config: str | Path | None = None, state_dir: str | Path | None = None,
               probe_remote: bool = False, timeout_sec: float | None = None) -> dict[str, Any]:
    """Return one `mindie-doctor/1` report without creating or mutating anything."""
    started = time.monotonic()
    budget = float(timeout_sec) if timeout_sec is not None else (
        REMOTE_BUDGET if probe_remote else LOCAL_BUDGET)
    deadline = started + max(0.5, budget)
    adapter_path, adapter, adapter_state = doctor_io.adapter(adapter_config)
    privacy = doctor_io.privacy(adapter_path, adapter, cwd)
    github = doctor_io.github(privacy["tier"], deadline)
    remote = doctor_io.remote(bool(probe_remote), deadline)
    diagnostics = doctor_io.diagnostics(reporting_config, state_dir)
    return {
        "schema": SCHEMA,
        "overall_status": _overall(privacy, remote, github, diagnostics, adapter_state),
        "runtime": doctor_io.runtime(adapter_path, adapter, adapter_state),
        "experience_and_privacy": privacy,
        "remote_npu": remote,
        "github_channel": github,
        "diagnostics": diagnostics,
        "remediations": _remediations(privacy, remote, github, diagnostics, adapter_state,
                                      bool(probe_remote)),
    }
