"""Storage lifecycle regressions: self-log pruning, atomic retirement, staging."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from mindie_diagnostics import reporting_commands as commands
from mindie_diagnostics import reporting_runtime as runtime
from mindie_diagnostics import runtime_retirement as retirement
from mindie_diagnostics import service
from mindie_diagnostics.integration import configure_reporting
from mindie_diagnostics.outbox import Outbox


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _config(tmp_path: Path) -> Path:
    config = tmp_path / "diagnostics.json"
    configure_reporting(
        True, config=config, repository="owner/repo", roots=[str(tmp_path / "logs")],
    )
    return config


def _runtime_root(config: Path) -> Path:
    root = config.with_suffix(".runtime")
    root.mkdir(mode=0o700)
    return root


def _generation(generations: Path, source_hash: str) -> Path:
    entry = generations / source_hash
    entry.mkdir(mode=0o700)
    (entry / "source.json").write_text('{"schema":1}\n')
    nested = entry / "venv" / "Scripts"
    nested.mkdir(parents=True)
    (nested / "pythonw.exe").write_bytes(b"locked-binary")
    return entry


def _exited_pid() -> int:
    windowless = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    proc = subprocess.Popen([sys.executable, "-I", "-c", "pass"], **windowless)
    proc.wait(timeout=10)
    return proc.pid


def test_maintain_prunes_worker_self_logs_without_ingesting_them(tmp_path):
    config = _config(tmp_path)
    state = config.with_suffix(".state")
    queue = Outbox(state / "reporter.sqlite3")
    events = state / "diagnostics" / "events" / "mindie-diagnostics"
    events.mkdir(parents=True)
    segment = events / f"{_exited_pid()}-{'a' * 32}.jsonl"
    segment.write_text('{"schema":1,"event":"worker.cycle"}\n')
    os.utime(segment, (1, 1))

    result = commands.maintain(config, budget_seconds=20)

    assert not segment.exists()
    assert not events.exists()
    assert any(row.get("removed_files") for row in result["retention"])
    # The self-log root is pruned locally and must never gain an ingestion
    # cursor or an incident that could be filed upstream.
    with queue.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM cursors").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM incidents").fetchone()[0] == 0


def test_publish_retains_exactly_one_previous_source_hash(tmp_path):
    config = _config(tmp_path)
    root = _runtime_root(config)
    old, new = _hash("old"), _hash("new")

    runtime._publish(root, {"schema": 1, "source_hash": old, "version": "0.5.13"})
    assert "previous_source_hash" not in runtime._read_current(root)

    runtime._publish(root, {"schema": 1, "source_hash": new, "version": "0.5.14"})
    assert runtime._read_current(root)["previous_source_hash"] == old

    # A no-op republish of the same current must not erase the rollback hash.
    runtime._publish(root, {"schema": 1, "source_hash": new, "version": "0.5.14"})
    assert runtime._read_current(root)["previous_source_hash"] == old


def test_retirement_is_two_phase_and_never_half_deletes_a_locked_generation(tmp_path, monkeypatch):
    config = _config(tmp_path)
    root = _runtime_root(config)
    generations = root / "generations"
    generations.mkdir(mode=0o700)
    hashes = [_hash(f"generation-{index}") for index in range(12)]
    entries = {source_hash: _generation(generations, source_hash) for source_hash in hashes}
    read_only = entries[hashes[5]] / "read-only.dll"
    read_only.write_bytes(b"read-only")
    read_only.chmod(0o400)
    stale = generations / ".staging-stale"
    stale.mkdir()
    os.utime(stale, (time.time() - 3600, time.time() - 3600))
    fresh = generations / ".staging-fresh"
    fresh.mkdir()
    interrupted = generations / ".trash-interrupted"
    interrupted.mkdir()
    (interrupted / "leftover.dll").write_bytes(b"x")

    locked = hashes[3]
    real_replace = os.replace

    def replace(source, destination):
        if Path(source).name == locked:
            raise PermissionError("pythonw.exe is still open")
        return real_replace(source, destination)

    monkeypatch.setattr(runtime.os, "replace", replace)
    retired = runtime.retire_other_generations(
        root, hashes[0], extra_keep={hashes[1], hashes[2]},
    )

    assert set(retired) == set(hashes[4:])
    assert retired.staging == (".staging-stale",)
    assert set(retired.deferred) == {locked}
    assert not interrupted.exists() and not stale.exists()
    assert fresh.is_dir()
    for kept in hashes[:3]:
        assert entries[kept].is_dir()
    # The rename failed, so the original generation remains complete: no
    # half-deleted .py tree and no .trash-* entry for it.
    assert (entries[locked] / "source.json").is_file()
    assert (entries[locked] / "venv" / "Scripts" / "pythonw.exe").is_file()
    assert not list(generations.glob(f".trash-{locked[:16]}-*"))
    for source_hash in hashes[4:]:
        assert not entries[source_hash].exists()


def test_maintain_retroactively_retires_eleven_legacy_generations(tmp_path, monkeypatch):
    config = _config(tmp_path)
    root = _runtime_root(config)
    generations = root / "generations"
    generations.mkdir(mode=0o700)
    hashes = [_hash(f"legacy-{index}") for index in range(11)]
    entries = {source_hash: _generation(generations, source_hash) for source_hash in hashes}
    (root / "current.json").write_text(json.dumps({
        "schema": 1,
        "source_hash": hashes[0],
        "version": "0.5.14",
        "revision": None,
        "previous_source_hash": hashes[1],
    }))
    stale = generations / ".staging-dead"
    stale.mkdir()
    os.utime(stale, (time.time() - 3600, time.time() - 3600))
    monkeypatch.setattr(commands, "_worker_view", lambda config: {"runtime_source": hashes[2]})

    result = commands.maintain(config, budget_seconds=75)

    assert set(result["retired_generations"]) == set(hashes[3:])
    assert not stale.exists()
    for kept in hashes[:3]:
        assert entries[kept].is_dir()
    for source_hash in hashes[3:]:
        assert not entries[source_hash].exists()


def _patch_ensure(monkeypatch, config: Path, root: Path, new_hash: str):
    old = runtime._read_current(root)
    monkeypatch.setattr(commands.shutil, "which", lambda name: sys.executable)
    monkeypatch.setattr(commands, "_worker_view", lambda config: {
        "healthy": True, "runtime_source": new_hash,
    })
    monkeypatch.setattr(service, "service_status", lambda **kwargs: {"status": "active"})
    monkeypatch.setattr(service, "remove_service", lambda **kwargs: {"status": "absent"})
    monkeypatch.setattr(service, "ensure_reporter_service", lambda *args, **kwargs: {"status": "installed"})

    def prepare(*args, **kwargs):
        runtime._publish(root, {"schema": 1, "source_hash": new_hash, "version": "0.5.14"})
        return {"python": sys.executable, "source_hash": new_hash}

    monkeypatch.setattr(runtime, "_prepare_locked", prepare)
    return old


def test_ensure_retires_only_beyond_the_rollback_closure(tmp_path, monkeypatch):
    config = _config(tmp_path)
    root = _runtime_root(config)
    generations = root / "generations"
    generations.mkdir(mode=0o700)
    previous, legacy, current = _hash("previous"), _hash("legacy"), _hash("new")
    for source_hash in (previous, legacy):
        _generation(generations, source_hash)
    (root / "current.json").write_text(json.dumps({
        "schema": 1, "source_hash": previous, "version": "0.5.13",
        "revision": None,
    }))
    _patch_ensure(monkeypatch, config, root, current)

    result = commands.ensure(config)

    assert result["status"] == "running"
    assert result["retired_generations"] == [legacy]
    assert (generations / previous).is_dir()
    assert not (generations / legacy).exists()


def test_ensure_reports_retirement_warning_without_downgrading_running(tmp_path, monkeypatch):
    config = _config(tmp_path)
    root = _runtime_root(config)
    generations = root / "generations"
    generations.mkdir(mode=0o700)
    previous, legacy, current = _hash("previous"), _hash("legacy"), _hash("new")
    for source_hash in (previous, legacy):
        _generation(generations, source_hash)
    (root / "current.json").write_text(json.dumps({
        "schema": 1, "source_hash": previous, "version": "0.5.13",
        "revision": None,
    }))
    _patch_ensure(monkeypatch, config, root, current)

    def refuse(*args, **kwargs):
        raise PermissionError("open handle")

    monkeypatch.setattr(runtime, "retire_other_generations", refuse)
    result = commands.ensure(config)

    assert result["status"] == "running"
    assert result["retirement_warning"] == "PermissionError"
    assert (generations / legacy).is_dir()


def test_orphan_profile_cleanup_skips_active_live_and_unmarked_dirs(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    active_name = "a" * 20
    active = profiles / active_name
    active.mkdir()
    dead = profiles / ("b" * 20)
    dead.mkdir()
    (dead / "profile.json").write_text(json.dumps({
        "schema": 1, "root": str(tmp_path / "missing-root"),
    }))
    live = profiles / ("c" * 20)
    live.mkdir()
    live_root = tmp_path / "live-root"
    live_root.mkdir()
    (live / "profile.json").write_text(json.dumps({
        "schema": 1, "root": str(live_root),
    }))
    unmarked = profiles / ("d" * 20)
    unmarked.mkdir()
    busy = profiles / ("e" * 20)
    busy.mkdir()
    (busy / "profile.json").write_text(json.dumps({
        "schema": 1, "root": str(tmp_path / "busy-root"),
    }))
    executable = busy / "venv" / "Scripts" / "pythonw.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"mapped")

    class _WindowsOS:
        name = "nt"

        def __getattr__(self, attribute):
            return getattr(os, attribute)

    monkeypatch.setattr(retirement, "os", _WindowsOS())
    monkeypatch.setattr(runtime, "_generations", lambda root: profiles / active_name)
    monkeypatch.setattr(retirement, "_image_available", lambda path: False)

    result = retirement.prune_orphan_windows_profiles(tmp_path / "active-root")

    assert list(result) == [dead.name]
    assert list(result.deferred) == [busy.name]
    assert not dead.exists()
    assert active.is_dir() and live.is_dir() and unmarked.is_dir() and busy.is_dir()


def test_service_publish_removes_staging_with_leftover_files(tmp_path, monkeypatch):
    unit_dir = tmp_path / "units"
    unit_dir.mkdir()
    path = unit_dir / service.UNIT
    real_mkstemp = service.tempfile.mkstemp

    def mkstemp_with_leftover(*args, **kwargs):
        (Path(kwargs["dir"]) / "leftover.tmp").write_text("kept by a failed helper")
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(service.tempfile, "mkstemp", mkstemp_with_leftover)
    monkeypatch.setattr(service, "which_native", lambda *args, **kwargs: None)

    assert service._publish(path, "[Unit]\nDescription=x\n", None) == "unavailable"
    assert path.is_file()
    assert not list(unit_dir.glob(".mindie-unit-*"))
