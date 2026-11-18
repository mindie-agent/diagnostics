"""Unified `mindie_doctor` regression tests.

[Role]
  Verify `run_doctor` and `mindie-diagnostics doctor` produce a stable
  `mindie-doctor/1` report, stay strictly read-only, never echo endpoint
  secrets, and classify healthy / attention_needed / degraded correctly.

[Invariants]
  * HOME, USERPROFILE, APPDATA, LOCALAPPDATA, XDG_*, MINDIE_*, REMOTE_DEV_*,
    TMPDIR, and PATH are isolated per test.
  * A completely empty installation creates zero files, directories, or SQLite
    sidecars anywhere under the isolated root.

[Timeouts & Locks]
  * The CLI roundtrip uses a 30s harness timeout; doctor's own budgets are
    exercised by the module under test.

[Side-Effect Policy]
  * Test-only fixtures write under pytest `tmp_path`; production code is read-only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mindie_diagnostics.doctor import run_doctor

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _doctor_isolated(tmp_path, monkeypatch):
    base = tmp_path
    for key, value in (
        ("HOME", base / "home"),
        ("USERPROFILE", base / "home"),
        ("APPDATA", base / "appdata"),
        ("LOCALAPPDATA", base / "localappdata"),
        ("XDG_CONFIG_HOME", base / "xdg-config"),
        ("XDG_DATA_HOME", base / "xdg-data"),
        ("XDG_STATE_HOME", base / "xdg-state"),
        ("XDG_CACHE_HOME", base / "xdg-cache"),
        ("MINDIE_CONFIG_HOME", base / "config"),
        ("MINDIE_DATA_HOME", base / "data"),
        ("MINDIE_DIAGNOSTICS_ROOT", base / "diag"),
        ("MINDIE_DIAGNOSTICS_CONFIG", base / "reporting.json"),
        ("REMOTE_DEV_AUTH_DIR", base / "auth"),
        ("REMOTE_DEV_ENDPOINTS_FILE", base / "endpoints.json"),
        ("TMPDIR", base / "tmp"),
    ):
        monkeypatch.setenv(key, str(value))
    for key in ("MINDIE_AGENT_CONFIG", "GH_TOKEN", "GITHUB_TOKEN", "MINDIE_COMMUNITY_POLICY"):
        monkeypatch.delenv(key, raising=False)
    bin_dir = base / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))


def _write(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if not isinstance(data, str) else data,
                    encoding="utf-8")
    return path


def _snapshot(root: Path) -> list[tuple[str, bool, int]]:
    rows = []
    for path in root.rglob("*"):
        rel = str(path.relative_to(root))
        size = path.stat().st_size if path.is_file() else 0
        rows.append((rel, path.is_dir(), size))
    return sorted(rows)


def _consent(choice: str, roots: list[Path] | None = None) -> Path:
    payload = {"schema": "mindie-consent/1", "choice": choice}
    if roots is not None:
        payload["project_roots"] = [str(root) for root in roots]
    return _write(Path(os.environ["MINDIE_CONFIG_HOME"]) / "mindie-consent.json", payload)


def test_empty_installation_is_attention_needed_and_creates_nothing(tmp_path):
    before = _snapshot(tmp_path)
    report = run_doctor()
    after = _snapshot(tmp_path)

    assert before == after
    assert not list(tmp_path.rglob("*.sqlite3"))
    assert not list(tmp_path.rglob("*-wal"))
    assert not list(tmp_path.rglob("*-shm"))
    assert not list(tmp_path.rglob("worker-health.json"))
    assert report["schema"] == "mindie-doctor/1"
    assert report["overall_status"] == "attention_needed"
    assert report["experience_and_privacy"]["tier"] == "unconfigured"
    assert report["experience_and_privacy"]["consent_state"] == "missing"
    assert report["github_channel"] == {
        "gh_installed": False, "status": "optional_unconfigured", "account": None,
    }
    assert report["remote_npu"]["configured"] is False
    assert report["remote_npu"]["probe"] == {"attempted": False}
    assert report["diagnostics"]["status"] == "not_configured"


def test_local_only_consent_scopes_cwd_and_allows_missing_gh(tmp_path):
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    _consent("local-only", [project])

    inside = run_doctor(cwd=project)
    scope = inside["experience_and_privacy"]
    assert inside["overall_status"] == "healthy"
    assert scope["tier"] == "local-only"
    assert scope["local_capture_enabled"] is True
    assert scope["community_publish_enabled"] is False
    assert scope["cwd_in_scope"] is True
    assert inside["github_channel"]["status"] == "optional_unconfigured"
    assert inside["remediations"] == []

    outside_report = run_doctor(cwd=outside)
    assert outside_report["overall_status"] == "healthy"
    assert outside_report["experience_and_privacy"]["cwd_in_scope"] is False
    assert outside_report["experience_and_privacy"]["local_capture_enabled"] is True
    assert outside_report["remediations"] == []


def test_contribute_without_gh_is_degraded_with_actionable_remediation(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _consent("contribute", [project])

    report = run_doctor(cwd=project)
    assert report["overall_status"] == "degraded"
    assert report["experience_and_privacy"]["tier"] == "contribute"
    assert report["experience_and_privacy"]["local_capture_enabled"] is True
    assert report["github_channel"]["gh_installed"] is False
    assert report["github_channel"]["status"] == "required_unauthenticated"
    text = " ".join(report["remediations"]).lower()
    assert "gh" in text and "auth login" in text


def test_endpoint_password_is_never_serialized(tmp_path):
    endpoints = Path(os.environ["REMOTE_DEV_ENDPOINTS_FILE"])
    _write(endpoints, {
        "default_alias": "npu",
        "npu": {"host": "192.0.2.10", "port": 22, "user": "root",
                "password": "secret123", "container": "cann"},
    })

    report = run_doctor()
    encoded = json.dumps(report)
    assert "secret123" not in encoded
    assert '"password"' not in encoded
    remote = report["remote_npu"]
    assert remote["configured"] is True
    assert remote["default_alias"] == "npu"
    assert remote["default_endpoint"] == {
        "host": "192.0.2.10", "port": 22, "user": "root",
        "container": "cann", "auth_ref": None,
    }


def test_cli_doctor_roundtrip_prints_one_json_document(tmp_path):
    config = tmp_path / "config"
    project = tmp_path / "project"
    project.mkdir()
    _write(config / "adapter.json", {
        "schema": "mindie-adapter-config/1", "host": "codex", "harness": "codex",
        "version": "1.2.3", "native_version": "9.9.9", "python": sys.executable,
        "domain_configs": {"vllm-ascend": str(tmp_path / "engine.json")},
    })
    _consent("local-only", [project])
    state = tmp_path / "state"

    proc = subprocess.run(
        [sys.executable, "-B", "-m", "mindie_diagnostics.cli", "doctor",
         "--adapter-config", str(config / "adapter.json"), "--cwd", str(project),
         "--reporting-config", str(tmp_path / "missing-reporting.json"),
         "--state", str(state)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""
    assert "\n" not in proc.stdout.strip()
    report = json.loads(proc.stdout)
    assert report["schema"] == "mindie-doctor/1"
    assert report["overall_status"] == "healthy"
    assert report["runtime"]["host"] == "codex"
    assert report["runtime"]["version"] == "1.2.3"
    assert report["runtime"]["native_version"] == "9.9.9"
    assert report["experience_and_privacy"]["tier"] == "local-only"
    assert report["experience_and_privacy"]["cwd_in_scope"] is True
    assert report["experience_and_privacy"]["registered_domains"] == ["vllm-ascend"]
    assert report["diagnostics"]["state_dir"] == str(state)
    assert not state.exists()


def test_probe_without_endpoint_only_recommends_remote_auth(tmp_path):
    report = run_doctor(probe_remote=True, timeout_sec=1.0)
    assert report["overall_status"] == "attention_needed"
    assert report["remote_npu"]["probe"] == {"attempted": False}
    assert any("remote_auth" in item for item in report["remediations"])


def test_reporting_status_stays_zero_create_on_empty_installation(tmp_path):
    from mindie_diagnostics.integration import reporting_status

    before = _snapshot(tmp_path)
    result = reporting_status()
    assert _snapshot(tmp_path) == before
    assert result["status"] == "not_configured"


def test_corrupt_authorities_are_degraded_without_mutation(tmp_path):
    config = tmp_path / "config"
    _write(config / "mindie-consent.json", "{not-json")
    endpoints = Path(os.environ["REMOTE_DEV_ENDPOINTS_FILE"])
    _write(endpoints, "{not-json")

    before = _snapshot(tmp_path)
    report = run_doctor()
    assert _snapshot(tmp_path) == before
    assert report["overall_status"] == "degraded"
    assert report["experience_and_privacy"]["tier"] == "corrupt"
    assert report["remote_npu"]["status"] == "corrupt"
    assert any("endpoint" in item.lower() for item in report["remediations"])


def test_authenticated_contribute_is_healthy(tmp_path, monkeypatch):
    from mindie_diagnostics import doctor_io

    project = tmp_path / "project"
    project.mkdir()
    _consent("contribute", [project])
    _write(Path(os.environ["MINDIE_CONFIG_HOME"]) / "mindie-community.json", {
        "schema": "mindie-community-config/1", "enabled": True,
        "repository": "owner/repo", "fork": "owner/repo", "account": "octocat",
        "project_roots": [str(project)],
    })
    calls = {}

    def fake_run(argv, **kwargs):
        calls.update(argv=argv, kwargs=kwargs)
        return subprocess.CompletedProcess(argv, 0, "octocat\n", "")

    monkeypatch.setattr(doctor_io.shutil, "which",
                        lambda name: "/fake/gh" if name == "gh" else None)
    monkeypatch.setattr(doctor_io.subprocess, "run", fake_run)

    report = run_doctor(cwd=project)
    assert report["overall_status"] == "healthy"
    assert report["github_channel"] == {
        "gh_installed": True, "status": "authenticated", "account": "octocat",
    }
    assert report["experience_and_privacy"]["community_publish_enabled"] is True
    assert report["remediations"] == []
    assert calls["argv"] == ["/fake/gh", "api", "user", "--jq", ".login"]
    assert calls["kwargs"]["env"]["GH_PROMPT_DISABLED"] == "1"


def test_downgrade_from_contribute_to_local_only_disables_reporting_cleanly(tmp_path):
    from mindie_diagnostics.integration import configure_reporting

    project = tmp_path / "project"
    project.mkdir()
    consent_path = _write(
        Path(os.environ["MINDIE_CONFIG_HOME"]) / "mindie-consent.json",
        {
            "schema": "mindie-consent/1",
            "choice": "contribute",
            "choice_revision": "0" * 32,
            "project_roots": [str(project)],
        },
    )
    reporting_cfg = Path(os.environ["MINDIE_DIAGNOSTICS_CONFIG"])
    cfg_res = configure_reporting(
        True,
        consent_config=str(consent_path),
        config=str(reporting_cfg),
        roots=[str(project)],
        repository="mindie-agent/mindie-agent",
    )
    assert cfg_res["status"] == "configured"

    _write(
        consent_path,
        {
            "schema": "mindie-consent/1",
            "choice": "local-only",
            "choice_revision": "1" * 32,
            "project_roots": [str(project)],
        },
    )
    report = run_doctor(cwd=project, reporting_config=reporting_cfg)
    assert report["overall_status"] == "healthy"
    assert report["diagnostics"]["status"] == "configured"
    assert report["diagnostics"]["enabled"] is False
    off_res = configure_reporting(
        False,
        consent_config=str(consent_path),
        config=str(reporting_cfg),
        roots=[str(project)],
        repository="mindie-agent/mindie-agent",
    )
    assert off_res["status"] == "configured"

