"""Storage lifecycle regressions: self-log pruning, atomic retirement, staging."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import pytest

from mindie_diagnostics import fallback as policy
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


def _linked_config(tmp_path: Path) -> Path:
    consent = tmp_path / "consent.json"
    consent.write_text(json.dumps({
        "schema": "mindie-consent/1",
        "choice": "contribute",
        "choice_revision": "e" * 32,
    }))
    config = tmp_path / "diagnostics.json"
    result = configure_reporting(
        True, config=config, repository="owner/repo",
        roots=[str(tmp_path / "logs")], consent_config=consent,
    )
    assert result["status"] == "configured"
    return config


def _generation(generations: Path, source_hash: str) -> Path:
    entry = generations / source_hash
    entry.mkdir(mode=0o700)
    (entry / "source.json").write_text('{"schema":1}\n')
    nested = entry / "venv" / "Scripts"
    nested.mkdir(parents=True)
    (nested / "pythonw.exe").write_bytes(b"locked-binary")
    return entry


def _service_definition(unit_dir: Path, generation: Path) -> Path:
    unit_dir.mkdir(exist_ok=True)
    path = unit_dir / "mindie-worker.service"
    path.write_text(
        f"ExecStart={generation / 'venv' / 'bin' / 'python'} "
        "-m mindie_diagnostics.worker\n"
    )
    return path


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
    executable = busy / ("f" * 64) / "venv" / "Scripts" / "pythonw.exe"
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


def test_maintain_defers_generation_retirement_when_worker_source_is_unconfirmed(tmp_path, monkeypatch):
    config = _config(tmp_path)
    root = _runtime_root(config)
    generations = root / "generations"
    generations.mkdir(mode=0o700)
    live, previous, current = _hash("live-a"), _hash("previous-b"), _hash("current-c")
    entries = {source_hash: _generation(generations, source_hash)
               for source_hash in (live, previous, current)}
    (root / "current.json").write_text(json.dumps({
        "schema": 1, "source_hash": current, "version": "0.5.14",
        "revision": None, "previous_source_hash": previous,
    }))
    monkeypatch.setattr(commands, "_worker_view",
                        lambda config: {"status": "unavailable", "healthy": False})
    monkeypatch.setattr(service, "service_status", lambda **kwargs: {"status": "absent"})

    result = commands.maintain(config, budget_seconds=20)

    assert result["retirement"]["status"] == "deferred"
    assert result["retirement"]["reason"] == "worker_generation_unconfirmed"
    assert result["retirement_warning"] == "worker_generation_unconfirmed"
    # The failed ensure history advanced current/previous, but the live worker
    # generation must survive an unreadable health record.
    for source_hash in (live, previous, current):
        assert entries[source_hash].is_dir()


def _three_generations(tmp_path):
    config = _config(tmp_path)
    root = _runtime_root(config)
    generations = root / "generations"
    generations.mkdir(mode=0o700)
    live, previous, current = _hash("live-a"), _hash("previous-b"), _hash("current-c")
    entries = {source_hash: _generation(generations, source_hash)
               for source_hash in (live, previous, current)}
    (root / "current.json").write_text(json.dumps({
        "schema": 1, "source_hash": current, "version": "0.5.14",
        "revision": None, "previous_source_hash": previous,
    }))
    return config, root, entries, live, previous, current


def test_maintain_keeps_live_generation_when_worker_health_is_missing(tmp_path, monkeypatch):
    config, root, entries, live, previous, current = _three_generations(tmp_path)
    unit_dir = tmp_path / "units"
    _service_definition(unit_dir, entries[current])
    monkeypatch.setattr(commands, "_worker_view",
                        lambda config: {"status": "not_started"})
    monkeypatch.setattr(service, "service_status", lambda **kwargs: {"status": "absent"})
    monkeypatch.setattr(retirement, "live_generation_hashes",
                        lambda roots, deadline=None: ({live}, True))

    result = commands.maintain(config, budget_seconds=20, unit_dir=unit_dir)

    # The service definition names C, but the live process still runs A.
    assert result["retired_generations"] == []
    for source_hash in (live, previous, current):
        assert entries[source_hash].is_dir()


def test_maintain_defers_when_service_unit_exists_without_healthy_worker(tmp_path, monkeypatch):
    config, root, entries, live, previous, current = _three_generations(tmp_path)
    unit_dir = tmp_path / "units"
    _service_definition(unit_dir, entries[current])
    monkeypatch.setattr(commands, "_worker_view",
                        lambda config: {"status": "not_started"})
    monkeypatch.setattr(service, "service_status", lambda **kwargs: {"status": "active"})

    result = commands.maintain(config, budget_seconds=20, unit_dir=unit_dir)

    # A service unit without a healthy worker cannot prove that A is gone, even
    # though the unit definition already names C.
    assert result["retirement"]["status"] == "deferred"
    assert result["retirement"]["reason"] == "worker_generation_unconfirmed"
    for source_hash in (live, previous, current):
        assert entries[source_hash].is_dir()


def test_maintain_defers_when_live_process_scan_is_incomplete(tmp_path, monkeypatch):
    config, root, entries, live, previous, current = _three_generations(tmp_path)
    monkeypatch.setattr(commands, "_worker_view",
                        lambda config: {"status": "not_started"})
    monkeypatch.setattr(service, "service_status", lambda **kwargs: {"status": "absent"})
    monkeypatch.setattr(retirement, "live_generation_hashes",
                        lambda roots, deadline=None: (set(), False))

    result = commands.maintain(config, budget_seconds=20)

    assert result["retirement"]["status"] == "deferred"
    for source_hash in (live, previous, current):
        assert entries[source_hash].is_dir()


def test_maintain_resolves_linked_profile_service_directory_for_retirement(tmp_path, monkeypatch):
    config = _linked_config(tmp_path)
    root = _runtime_root(config)
    generations = root / "generations"
    generations.mkdir(mode=0o700)
    live, previous, current = _hash("service-a"), _hash("previous-b"), _hash("current-c")
    entries = {source_hash: _generation(generations, source_hash)
               for source_hash in (live, previous, current)}
    (root / "current.json").write_text(json.dumps({
        "schema": 1, "source_hash": current, "version": "0.5.14",
        "revision": None, "previous_source_hash": previous,
    }))
    unit_dir = policy.service_directory(config)
    assert unit_dir is not None
    unit_dir.mkdir(parents=True)
    _service_definition(unit_dir, entries[live])
    monkeypatch.setattr(commands, "_worker_view",
                        lambda config: {"status": "not_started"})
    monkeypatch.setattr(retirement, "live_generation_hashes",
                        lambda roots, deadline=None: (set(), True))
    seen = []

    def status(*, unit_dir=None, runner=None):
        seen.append(unit_dir)
        return ({"status": "idle"} if unit_dir == policy.service_directory(config)
                else {"status": "absent"})

    monkeypatch.setattr(service, "service_status", status)

    result = commands.maintain(config, budget_seconds=20)

    assert seen and seen[0] == policy.service_directory(config)
    assert result["retirement"]["status"] == "deferred"
    assert result["retirement"]["reason"] == "worker_generation_unconfirmed"
    # The installed service definition still names A while the live worker is
    # unconfirmed: A must survive even though current/previous are C/B.
    for source_hash in (live, previous, current):
        assert entries[source_hash].is_dir()


def test_maintain_keeps_a_generation_named_by_the_service_definition(tmp_path, monkeypatch):
    config, root, entries, live, previous, current = _three_generations(tmp_path)
    unit_dir = tmp_path / "units"
    _service_definition(unit_dir, entries[live])
    monkeypatch.setattr(commands, "_worker_view",
                        lambda config: {"healthy": True, "runtime_source": current})
    monkeypatch.setattr(service, "service_status", lambda **kwargs: {"status": "active"})
    monkeypatch.setattr(retirement, "live_generation_hashes",
                        lambda roots, deadline=None: (set(), True))

    result = commands.maintain(config, budget_seconds=20, unit_dir=unit_dir)

    assert result["retired_generations"] == []
    for source_hash in (live, previous, current):
        assert entries[source_hash].is_dir()


@pytest.mark.skipif(os.name != "posix", reason="POSIX ps process inspection")
def test_live_generation_scan_finds_a_real_process_path(tmp_path):
    source_hash = "a" * 64
    generation = tmp_path / "generations" / source_hash
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(30)", str(generation)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        found, complete = set(), False
        while time.monotonic() < deadline:
            found, complete = retirement.live_generation_hashes(
                (tmp_path / "generations",), deadline=time.monotonic() + 2)
            if found:
                break
            time.sleep(0.05)
        assert complete is True
        assert source_hash in found
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.mark.skipif(os.name != "posix", reason="POSIX ps process inspection")
def test_posix_scan_uses_unlimited_width_and_strips_columns(tmp_path, monkeypatch):
    source_hash = "b" * 64
    command = (
        f"{tmp_path}/generations/{source_hash}/"
        + "nested/" * 30
        + "venv/bin/python -m mindie_diagnostics.worker"
    )
    assert len(command) > 200
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["env"] = dict(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0, stdout=command.encode())

    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setattr(retirement.subprocess, "run", fake_run)

    found, complete = retirement.live_generation_hashes(
        (tmp_path / "generations",), deadline=time.monotonic() + 5)

    assert os.environ["COLUMNS"] == "80"
    assert captured["argv"] == ["ps", "-ww", "-axo", "command="]
    assert "COLUMNS" not in captured["env"]
    assert found == {source_hash}
    assert complete is True


@pytest.mark.skipif(os.name != "posix", reason="POSIX ps process inspection")
def test_posix_scan_treats_a_capped_read_as_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(retirement, "_PROCESS_TEXT_LIMIT", 64)
    monkeypatch.setattr(retirement.subprocess, "run",
                        lambda argv, **kwargs: subprocess.CompletedProcess(
                            argv, 0, stdout=b"x" * 64))

    assert retirement._posix_process_text(time.monotonic() + 5) is None
    found, complete = retirement.live_generation_hashes(
        (tmp_path / "generations",), deadline=time.monotonic() + 5)

    assert found == set()
    assert complete is False


def test_service_definition_scan_counts_every_entry_and_stops_at_cap(tmp_path, monkeypatch):
    unit_dir = tmp_path / "units"
    unit_dir.mkdir()

    class _Entry:
        def __init__(self, index):
            self.path = str(unit_dir / f"{index:03d}.file")

        def is_file(self, follow_symlinks=False):
            return True

    class _Scan:
        def __enter__(self):
            return iter(_Entry(index) for index in range(100))

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(commands.os, "scandir", lambda path: _Scan())

    files = commands._service_definition_files(unit_dir, None, time.monotonic() + 5)

    scanned = [path for path in files
               if path.parent == unit_dir and path.name.endswith(".file")]
    assert len(scanned) == commands._SERVICE_ENTRY_LIMIT


def test_service_definition_scan_stops_at_deadline(tmp_path, monkeypatch):
    unit_dir = tmp_path / "units"
    unit_dir.mkdir()

    class _Entry:
        def __init__(self, index):
            self.path = str(unit_dir / f"{index:03d}.file")

        def is_file(self, follow_symlinks=False):
            return True

    class _Scan:
        def __enter__(self):
            return iter(_Entry(index) for index in range(10))

        def __exit__(self, *exc):
            return False

    clock = iter((0.0, 10.0))
    monkeypatch.setattr(commands.os, "scandir", lambda path: _Scan())
    monkeypatch.setattr(commands.time, "monotonic", lambda: next(clock))

    files = commands._service_definition_files(unit_dir, None, 5.0)

    scanned = [path for path in files
               if path.parent == unit_dir and path.name.endswith(".file")]
    assert len(scanned) == 1


def test_maintain_runs_bounded_windows_orphan_profile_cleanup(tmp_path, monkeypatch):
    config = _config(tmp_path)
    calls = []

    def cleanup(active_root, *, budget_seconds=0):
        calls.append((active_root, budget_seconds))
        return retirement._Retirement(["dead-profile"], swept=[".trash-old"])

    monkeypatch.setattr(retirement, "prune_orphan_windows_profiles", cleanup)

    result = commands.maintain(config, budget_seconds=20)

    assert len(calls) == 1
    assert calls[0][1] <= 3.0
    assert result["orphan_profiles"]["retired_profiles"] == ["dead-profile"]
    assert result["orphan_profiles"]["swept_trash"] == [".trash-old"]


def test_remove_tree_detects_windows_junction_without_os_path_isjunction(tmp_path, monkeypatch):
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    keep = sentinel / "keep.txt"
    keep.write_text("outside")
    trash = tmp_path / ".trash-junction"
    trash.mkdir()
    junction = trash / "junction"
    junction.mkdir()

    real_lstat = os.lstat
    real_scandir = os.scandir
    scans = []

    class _JunctionStat:
        st_mode = stat.S_IFDIR
        st_file_attributes = stat.S_IFDIR | 0x410  # DIRECTORY | REPARSE_POINT
        st_reparse_tag = 0xA0000003  # IO_REPARSE_TAG_MOUNT_POINT

    class _PathWithoutIsjunction:
        def __getattr__(self, attribute):
            if attribute == "isjunction":
                raise AttributeError(attribute)
            return getattr(os.path, attribute)

    class _WindowsOS:
        name = "nt"
        path = _PathWithoutIsjunction()

        def lstat(self, path):
            if Path(path) == junction:
                return _JunctionStat()
            return real_lstat(path)

        def scandir(self, path):
            scans.append(Path(path))
            return real_scandir(path)

        def __getattr__(self, attribute):
            return getattr(os, attribute)

    monkeypatch.setattr(retirement, "os", _WindowsOS())

    # Simulate Python 3.11, where os.path has no isjunction at all.
    assert not hasattr(retirement.os.path, "isjunction")
    assert retirement._is_junction(junction) is True
    assert retirement._remove_tree(trash, time.monotonic() + 5) is True
    assert junction not in scans  # a junction is never traversed
    assert not trash.exists()
    assert keep.read_text() == "outside"
    assert sentinel.is_dir()


def test_remove_tree_unlinks_a_symlink_without_touching_its_target(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    keep = target / "keep.txt"
    keep.write_text("outside")
    trash = tmp_path / ".trash-link"
    trash.mkdir()
    link = trash / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")

    assert retirement._remove_tree(trash, time.monotonic() + 5) is True
    assert keep.read_text() == "outside"
    assert target.is_dir() and not trash.exists()


def test_lstat_timeout_propagates_from_junction_detection(tmp_path, monkeypatch):
    def timeout_lstat(path):
        raise TimeoutError("simulated lstat timeout")

    monkeypatch.setattr(retirement.os, "lstat", timeout_lstat)

    with pytest.raises(TimeoutError):
        retirement._is_junction(tmp_path / "entry")
    with pytest.raises(TimeoutError):
        retirement._remove_tree(tmp_path / "entry", time.monotonic() + 5)


class _WindowsOS:
    name = "nt"

    def __getattr__(self, attribute):
        return getattr(os, attribute)


def test_remove_tree_continues_after_a_child_disappears(tmp_path, monkeypatch):
    root = tmp_path / "tree"
    root.mkdir()
    first = root / "first.txt"
    first.write_text("first")
    second = root / "second.txt"
    second.write_text("second")
    third = root / "third.txt"
    third.write_text("third")
    real_unlink = retirement._unlink
    disappeared = []

    def unlink_once(path):
        path = Path(path)
        if path == first and not disappeared:
            disappeared.append(path)
            real_unlink(path)
            raise FileNotFoundError(path)
        return real_unlink(path)

    monkeypatch.setattr(retirement, "_unlink", unlink_once)

    assert retirement._remove_tree(root, time.monotonic() + 5) is True
    assert disappeared == [first]
    assert not first.exists() and not second.exists() and not third.exists()
    assert not root.exists()


def test_remove_tree_returns_false_when_the_root_survives(tmp_path, monkeypatch):
    root = tmp_path / "tree"
    root.mkdir()
    child = root / "child.txt"
    child.write_text("child")
    real_rmdir = retirement._rmdir

    def fail_root(path):
        if Path(path) == root:
            raise OSError("root is held open")
        return real_rmdir(path)

    monkeypatch.setattr(retirement, "_rmdir", fail_root)

    assert retirement._remove_tree(root, time.monotonic() + 5) is False
    assert root.is_dir()
    assert not child.exists()


def test_profile_idle_requires_a_complete_scan(tmp_path, monkeypatch):
    profile = tmp_path / ("a" * 20)
    profile.mkdir()
    monkeypatch.setattr(retirement, "_bounded_entries",
                        lambda path, deadline: ([], False))

    assert retirement._profile_idle(profile, time.monotonic() + 5) is False


def test_orphan_profile_cleanup_rechecks_deadline_before_replace(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    active_name = "a" * 20
    (profiles / active_name).mkdir()
    dead = profiles / ("b" * 20)
    dead.mkdir()
    (dead / "profile.json").write_text(json.dumps({
        "schema": 1, "root": str(tmp_path / "missing-root"),
    }))
    monkeypatch.setattr(retirement, "os", _WindowsOS())
    monkeypatch.setattr(runtime, "_generations", lambda root: profiles / active_name)
    monkeypatch.setattr(retirement, "_profile_idle",
                        lambda path, deadline: time.sleep(0.3) or True)
    replaced = []
    monkeypatch.setattr(retirement.os, "replace",
                        lambda source, destination: replaced.append(source))

    result = retirement.prune_orphan_windows_profiles(
        tmp_path / "active-root", budget_seconds=0.2)

    # The scan consumed the whole budget: do not rename the profile to trash.
    assert replaced == []
    assert dead.is_dir()
    assert dead.name in result.deferred


def test_orphan_profile_defers_when_the_target_volume_anchor_is_unavailable(tmp_path, monkeypatch):
    from mindie_diagnostics import fallback

    profiles = tmp_path / "profiles"
    profiles.mkdir()
    active_name = "a" * 20
    (profiles / active_name).mkdir()
    dead = profiles / ("b" * 20)
    dead.mkdir()
    unmounted_drive = next(
        letter for letter in "QZYXWVUTSRPONMLKJIHGFE"
        if not Path(f"{letter}:/").exists() and not Path(f"{letter}:\\").exists()
    )
    (dead / "profile.json").write_text(json.dumps({
        "schema": 1, "root": f"{unmounted_drive}:/unmounted-policy-root",
    }))
    monkeypatch.setattr(fallback, "_as_local_absolute", lambda value: Path(value))
    monkeypatch.setattr(retirement, "os", _WindowsOS())
    monkeypatch.setattr(runtime, "_generations", lambda root: profiles / active_name)

    result = retirement.prune_orphan_windows_profiles(tmp_path / "active-root")

    assert list(result) == []
    assert result.deferred == (dead.name,)
    assert result.volume_unavailable == (dead.name,)
    assert dead.is_dir()


def test_orphan_profile_trash_sweep_resumes_an_interrupted_pass(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    active_name = "a" * 20
    (profiles / active_name).mkdir()
    dead = profiles / ("b" * 20)
    dead.mkdir()
    (dead / "profile.json").write_text(json.dumps({
        "schema": 1, "root": str(tmp_path / "missing-root"),
    }))
    monkeypatch.setattr(retirement, "os", _WindowsOS())
    monkeypatch.setattr(runtime, "_generations", lambda root: profiles / active_name)
    real_remove = retirement._remove_tree
    calls = []

    def timed_out_once(path, deadline):
        calls.append(Path(path).name)
        if len(calls) == 1:
            return False
        return real_remove(path, deadline)

    monkeypatch.setattr(retirement, "_remove_tree", timed_out_once)

    first = retirement.prune_orphan_windows_profiles(tmp_path / "active-root")
    trash = list(profiles.glob(".trash-*"))
    assert list(first) == [dead.name]
    assert len(trash) == 1 and trash[0].name in first.deferred

    second = retirement.prune_orphan_windows_profiles(tmp_path / "active-root")

    assert not trash[0].exists()
    assert trash[0].name in second.swept


def test_blocked_writer_recovers_after_stale_marker_and_writes_its_event(tmp_path):
    from mindie_diagnostics import logging as log

    events = tmp_path / "events" / "blocked-recovery"
    events.mkdir(parents=True)
    old = events / f"{_exited_pid()}-{'a' * 32}.jsonl"
    old.write_text('{"schema":1}\n')
    os.utime(old, (1, 1))
    marker = tmp_path / "retention-state.json"
    marker.write_text(json.dumps({
        "schema": 1, "blocked": True, "limited": True,
        "checked_at": time.time() - 3600,
    }))
    marker.chmod(0o600)
    recorder = log.configure("blocked-recovery", root=tmp_path)
    try:
        recorder.event("INFO", "recovered")
        path = Path(recorder.record_ref)
        assert path.is_file()
        assert "recovered" in path.read_text()
    finally:
        recorder.close()

    assert not old.exists()
    state = json.loads(marker.read_text())
    assert state["blocked"] is False
