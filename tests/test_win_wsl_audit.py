"""Windows PowerShell 5.1, WSL2 PATH/DrvFS, non-systemd service, and TLS transport regressions."""
from __future__ import annotations

import base64
import http.client
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import time

import pytest

from mindie_diagnostics import fallback, filesystem, platform_service, reporting_commands, reporting_runtime, service
from mindie_diagnostics.github_transport import TransportError, _run_gh
from mindie_diagnostics.ingestion import ingest
from mindie_diagnostics.integration import configure_reporting, record_failure, reporting_status
from mindie_diagnostics.outbox import Outbox
from windows_launcher_fixture import write_launcher


@pytest.mark.parametrize(
    "error",
    [
        ssl.SSLError(1, "[SSL: UNEXPECTED_EOF_WHILE_READING] PRIVATE-HOST"),
        ssl.SSLEOFError(8, "EOF occurred in violation of protocol PRIVATE-HOST"),
        socket.gaierror(-2, "Name or service not known PRIVATE-HOST"),
        socket.herror(1, "Unknown host PRIVATE-HOST"),
        http.client.IncompleteRead(b"PRIVATE-BODY", 1024),
        http.client.RemoteDisconnected("Remote end closed connection without response"),
        http.client.BadStatusLine("PRIVATE-LINE"),
    ],
)
def test_midstream_tls_dns_and_http_drops_are_connectivity_not_internal_bugs(tmp_path, reporting_consent, error):
    assert fallback.update_failure_policy(error) == ("connectivity", False, "update_transport_unavailable")
    wrapped = RuntimeError("download stream aborted")
    wrapped.__cause__ = error
    assert fallback.update_failure_policy(wrapped) == ("connectivity", False, "update_transport_unavailable")

    ref = record_failure(
        "mindie-agent-codex",
        "release.update",
        stage="download",
        category="update",
        version="0.1.5",
        revision="a" * 40,
        exception=wrapped,
        root=tmp_path,
    )
    queue = Outbox(tmp_path / "queue.db")
    assert ingest(tmp_path, queue, config=reporting_consent["config_file"])["enqueued"] == 0
    assert queue.rows() == []
    view = reporting_status(config=reporting_consent["config_file"])["local"]["recent"][0]
    assert view["incident_id"] == ref["incident_id"]
    assert view["reporting_eligibility"] == "local_only"
    assert "PRIVATE" not in json.dumps(view)


def test_tls_cert_verification_and_http_client_state_bugs_keep_distinct_policies(tmp_path, reporting_consent):
    cert_err = ssl.SSLCertVerificationError(1, "certificate verify failed: self-signed certificate")
    assert fallback.update_failure_policy(cert_err) == ("credentials", False, "update_credentials_required")
    record_failure(
        "mindie-agent-cc",
        "release.update",
        stage="download",
        category="update",
        version="0.1.5",
        revision="b" * 40,
        exception=cert_err,
        root=tmp_path,
    )
    queue = Outbox(tmp_path / "queue.db")
    assert ingest(tmp_path, queue, config=reporting_consent["config_file"])["enqueued"] == 0

    for internal_bug in (
        http.client.InvalidURL("http://bad url"),
        http.client.ImproperConnectionState("cannot send"),
        http.client.NotConnected(),
    ):
        assert fallback.update_failure_policy(internal_bug) == ("internal", True, "update_internal_failure")


@pytest.mark.skipif(os.name == "nt", reason="POSIX / WSL2 PATH shadowing behavior")
def test_wsl_which_native_and_gh_transport_skip_pe_and_drvfs_shims(tmp_path, monkeypatch):
    monkeypatch.setattr(filesystem.sys, "platform", "linux")
    drvfs_bin = tmp_path / "mnt_c" / "Program Files" / "GitHub CLI"
    scoop_shims = tmp_path / "mnt_c" / "Users" / "alice" / "scoop" / "shims"
    linux_bin = tmp_path / "usr_bin"
    drvfs_bin.mkdir(parents=True)
    scoop_shims.mkdir(parents=True)
    linux_bin.mkdir(parents=True)

    # 1. Real PE header disguised as `gh` and `gh.exe`
    pe_gh = write_launcher(drvfs_bin / "gh")
    pe_gh.chmod(0o755)
    pe_gh_exe = write_launcher(drvfs_bin / "gh.exe")
    pe_gh_exe.chmod(0o755)

    # 2. POSIX-looking shell shim living on a WSL2 DrvFS 9p mount
    drvfs_shim = scoop_shims / "gh"
    drvfs_shim.write_text("#!/bin/sh\nexec gh.exe \"$@\"\n", encoding="utf-8")
    drvfs_shim.chmod(0o755)

    mountinfo = tmp_path / "mountinfo"
    escaped_root = str((tmp_path / "mnt_c").resolve()).replace(" ", "\\040")
    mountinfo.write_text(
        f"36 28 0:31 / / rw,relatime - ext4 /dev/sdb rw\n"
        f"92 36 0:54 / {escaped_root} rw,noatime - 9p C:\\134 rw,dirsync,aname=drvfs;path=C:\\;uid=1000;gid=1000\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(filesystem, "_MOUNTINFO_PATH", str(mountinfo))

    # When only Windows PE + DrvFS shims exist on PATH, native lookup returns None
    shadow_only_path = os.pathsep.join([str(drvfs_bin), str(scoop_shims)])
    monkeypatch.setenv("PATH", shadow_only_path)
    assert filesystem.is_pe_executable(pe_gh) is True
    assert filesystem.is_wsl_drvfs(drvfs_shim) is True
    assert filesystem.which_native("gh") is None
    assert filesystem.which_native("gh.exe") is None

    with pytest.raises(TransportError) as exc_info:
        _run_gh(["gh", "api", "rate_limit"], None, deadline=time.monotonic() + 3.0, max_bytes=4096)
    assert exc_info.value.code == "github_client_missing"
    assert exc_info.value.permanent is True and exc_info.value.uncertain is False

    config = tmp_path / "policy.json"
    configure_reporting(True, config=config, roots=[str(tmp_path / "logs")])
    ensured = reporting_commands.ensure(config)
    assert ensured == {"status": "configuration_unavailable", "category": "gh_unavailable"}

    # Once a native Linux `gh` exists later in PATH, `which_native` selects it cleanly
    linux_gh = linux_bin / "gh"
    linux_gh.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    linux_gh.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join([shadow_only_path, str(linux_bin)]))
    assert filesystem.which_native("gh") == str(linux_gh)


@pytest.mark.skipif(os.name == "nt", reason="POSIX / WSL2 DrvFS mode bit checks")
def test_wsl_drvfs_and_native_fail_closed_for_worker_runtime_and_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(filesystem.sys, "platform", "linux")
    euid = os.geteuid()
    drvfs_root = tmp_path / "mnt_c" / "workspace"
    drvfs_meta_root = tmp_path / "mnt_d" / "workspace"
    native_root = tmp_path / "ext4_home"
    drvfs_root.mkdir(parents=True)
    drvfs_meta_root.mkdir(parents=True)
    native_root.mkdir(parents=True)

    mountinfo = tmp_path / "mountinfo"
    escaped_drvfs = str((tmp_path / "mnt_c").resolve()).replace(" ", "\\040")
    escaped_meta = str((tmp_path / "mnt_d").resolve()).replace(" ", "\\040")
    mountinfo.write_text(
        f"36 28 0:31 / / rw,relatime - ext4 /dev/sdb rw\n"
        f"92 36 0:54 / {escaped_drvfs} rw,noatime - 9p C:\\134 rw,aname=drvfs;path=C:\\;uid={euid};gid={euid}\n"
        f"93 36 0:55 / {escaped_meta} rw,noatime - 9p D:\\134 rw,aname=drvfs;path=D:\\;uid={euid};gid={euid};metadata\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(filesystem, "_MOUNTINFO_PATH", str(mountinfo))

    # Unmetadata'd DrvFS synthetic mount UID does NOT establish private access -> 0777/0666 MUST fail closed!
    drvfs_dir = drvfs_root / "runtime_dir"
    drvfs_dir.mkdir()
    drvfs_dir.chmod(0o777)
    with pytest.raises(RuntimeError, match="runtime_untrusted"):
        reporting_runtime._require_dir(drvfs_dir, fallback._owned)

    drvfs_env = drvfs_root / "mindie-network.env"
    drvfs_env.write_text('HTTPS_PROXY="http://proxy.example.com:8080"\n', encoding="utf-8")
    drvfs_env.chmod(0o777)
    with pytest.raises(service.ServiceError, match="unsafe_environment_file"):
        service._environment_file(drvfs_env)

    # DrvFS mounts with `metadata` enabled succeed ONLY when truly 0700/0600, and fail closed on 0777/0666
    meta_dir = drvfs_meta_root / "runtime_dir"
    meta_dir.mkdir()
    meta_dir.chmod(0o777)
    with pytest.raises(RuntimeError, match="runtime_untrusted"):
        reporting_runtime._require_dir(meta_dir, fallback._owned)
    meta_dir.chmod(0o700)
    reporting_runtime._require_dir(meta_dir, fallback._owned)

    meta_env = drvfs_meta_root / "mindie-network.env"
    meta_env.write_text('HTTPS_PROXY="http://proxy.example.com:8080"\n', encoding="utf-8")
    meta_env.chmod(0o666)
    with pytest.raises(service.ServiceError, match="unsafe_environment_file"):
        service._environment_file(meta_env)
    meta_env.chmod(0o600)
    assert service._environment_file(meta_env) == meta_env.absolute()

    # Symlinks on DrvFS must still be rejected
    drvfs_link = drvfs_root / "linked.env"
    drvfs_link.symlink_to(drvfs_env)
    with pytest.raises(service.ServiceError, match="unsafe_environment_file"):
        service._environment_file(drvfs_link)

    # Native ext4 directories and env files with broad permissions still fail closed
    native_dir = native_root / "runtime_dir"
    native_dir.mkdir()
    native_dir.chmod(0o755)
    with pytest.raises(RuntimeError, match="runtime_untrusted"):
        reporting_runtime._require_dir(native_dir, fallback._owned)

    native_env = native_root / "mindie-network.env"
    native_env.write_text('HTTPS_PROXY="http://proxy.example.com:8080"\n', encoding="utf-8")
    native_env.chmod(0o644)
    with pytest.raises(service.ServiceError, match="unsafe_environment_file"):
        service._environment_file(native_env)


@pytest.mark.skipif(os.name == "nt", reason="POSIX native resolution execution test")
def test_run_gh_executes_resolved_native_path_not_shadowing_drvfs_shim(tmp_path, monkeypatch):
    monkeypatch.setattr(filesystem.sys, "platform", "linux")
    drvfs_bin = tmp_path / "mnt_c" / "gh_cli"
    native_bin = tmp_path / "usr_bin"
    drvfs_bin.mkdir(parents=True)
    native_bin.mkdir(parents=True)

    mountinfo = tmp_path / "mountinfo"
    escaped = str((tmp_path / "mnt_c").resolve()).replace(" ", "\\040")
    mountinfo.write_text(
        f"36 28 0:31 / / rw,relatime - ext4 /dev/sdb rw\n"
        f"92 36 0:54 / {escaped} rw,noatime - drvfs C: rw,uid={os.geteuid()},gid={os.geteuid()}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(filesystem, "_MOUNTINFO_PATH", str(mountinfo))

    shim_gh = drvfs_bin / "gh"
    shim_gh.write_text("#!/bin/sh\necho SHIM_EXECUTED\nexit 99\n", encoding="utf-8")
    shim_gh.chmod(0o755)

    real_gh = native_bin / "gh"
    real_gh.write_text("#!/bin/sh\necho NATIVE_GH_OK\nexit 0\n", encoding="utf-8")
    real_gh.chmod(0o755)

    monkeypatch.setenv("PATH", f"{drvfs_bin}{os.pathsep}{native_bin}{os.pathsep}/bin:/usr/bin")
    code, raw, _ = _run_gh(["gh", "version"], None, deadline=time.monotonic() + 5.0, max_bytes=4096)
    assert code == 0
    assert b"NATIVE_GH_OK" in raw
    assert b"SHIM_EXECUTED" not in raw


@pytest.mark.skipif(os.name == "nt", reason="POSIX detached process supervision")
def test_non_systemd_linux_local_worker_lifecycle_and_pid_reuse_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(service.sys, "platform", "linux")
    monkeypatch.setattr(platform_service.sys, "platform", "linux")

    venv_dir = tmp_path / "gen-venv"
    bin_dir = venv_dir / "bin"
    bin_dir.mkdir(parents=True)
    fake_py = bin_dir / "python"
    fake_py.write_text(
        f"#!{sys.executable}\nimport time\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    fake_py.chmod(0o755)
    fake_gh = bin_dir / "gh"
    fake_gh.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_gh.chmod(0o755)

    def non_systemd_runner(argv, **kwargs):
        if Path(argv[0]).name == "systemctl":
            return subprocess.CompletedProcess(argv, 1, "", "Failed to connect to bus: No medium found")
        if argv[0] == str(fake_py) and argv[1:3] == ["-I", "-c"]:
            facts = {
                "prefix": str(venv_dir),
                "base": "/usr",
                "version": "0.5.12",
                "editable": False,
                "package": str(venv_dir / "lib/python3.11/site-packages/mindie_diagnostics"),
            }
            return subprocess.CompletedProcess(argv, 0, json.dumps(facts), "")
        return subprocess.run(argv, **kwargs)

    non_systemd_runner._is_real_runner = True
    config = tmp_path / "reporting.json"
    roots = [tmp_path / "logs"]
    state = tmp_path / "state"
    unit_dir = tmp_path / "wsl-service"
    configure_reporting(True, config=config, roots=[str(roots[0])])
    revision = fallback.read_policy(config)["revision"]

    try:
        installed = service.install_service(
            roots,
            state,
            "mindie-agent/mindie-agent",
            python=fake_py,
            gh=fake_gh,
            unit_dir=unit_dir,
            runner=non_systemd_runner,
            reporting_config=config,
            expected_consent_revision=revision,
            since="2026-10-01T00:00:00Z",
        )
        assert installed["status"] == "installed"
        assert installed["platform"] == "linux"
        status_1 = service.service_status(unit_dir=unit_dir, runner=non_systemd_runner)
        assert status_1["status"] == "active"
        pid_1 = status_1["pid"]
        assert pid_1 > 1

        # Reconfigure interval -> stops pid_1 cleanly and launches pid_2 while keeping `since`
        updated = service.install_service(
            roots,
            state,
            "mindie-agent/mindie-agent",
            python=fake_py,
            gh=fake_gh,
            interval=120,
            unit_dir=unit_dir,
            runner=non_systemd_runner,
            reporting_config=config,
            expected_consent_revision=revision,
            since="2029-01-01T00:00:00Z",
        )
        assert updated["changed"] is True
        assert updated["since"] == installed["since"]
        status_2 = service.service_status(unit_dir=unit_dir, runner=non_systemd_runner)
        assert status_2["status"] == "active"
        pid_2 = status_2["pid"]
        assert pid_2 != pid_1
        with pytest.raises(ProcessLookupError):
            os.kill(pid_1, 0)

        # PID-reuse & missing-identity safety: mismatched or missing identity never signals bystander
        bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            assert platform_service._linux_worker_alive(bystander.pid, "proc:0:1") is False
            assert platform_service._linux_worker_alive(bystander.pid, None) is False
            assert platform_service._linux_worker_alive(bystander.pid, "") is False
        finally:
            bystander.terminate()
            bystander.wait(timeout=5)

        marker = state / "keep.sqlite3"
        marker.write_bytes(b"preserve-outbox")
        removed = service.remove_service(unit_dir=unit_dir, runner=non_systemd_runner)
        assert removed == {"status": "removed", "unit": updated["unit"], "state_preserved": True}
        assert marker.read_bytes() == b"preserve-outbox"
        assert service.service_status(unit_dir=unit_dir, runner=non_systemd_runner)["status"] == "absent"
        with pytest.raises(ProcessLookupError):
            os.kill(pid_2, 0)
    finally:
        manifest = platform_service._paths(unit_dir)[0]
        pid_file = manifest.with_name("worker-process.json")
        if pid_file.is_file():
            try:
                os.kill(json.loads(pid_file.read_text())["pid"], 9)
            except Exception:
                pass


@pytest.mark.skipif(os.name == "nt", reason="POSIX registration cleanup and rollback checks")
def test_worker_registration_cleanup_and_timeout_transaction_rollback(tmp_path, monkeypatch):
    manifest = tmp_path / "service.json"
    unit = tmp_path / "mindie-diagnostics.service"
    unit.write_bytes(b"old-unit")
    manifest.write_bytes(b"old-manifest")

    real_write = platform_service._write
    calls = {"n": 0}

    def timeout_on_manifest(path, data):
        calls["n"] += 1
        if path == manifest and calls["n"] == 2:
            raise TimeoutError("manifest_write_timeout")
        return real_write(path, data)

    monkeypatch.setattr(platform_service, "_write", timeout_on_manifest)
    with pytest.raises(TimeoutError, match="manifest_write_timeout"):
        platform_service._publish_unit_and_manifest(unit, b"new-unit", manifest, b"new-manifest")
    assert unit.read_bytes() == b"old-unit"
    assert manifest.read_bytes() == b"old-manifest"

    # Inject PID-file write failure -> spawned child MUST be terminated and reaped!
    spawned = []
    real_popen = subprocess.Popen

    def tracking_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        if args and list(args[0])[:1] == [str(sleeper)]:
            spawned.append(proc)
        return proc

    def fail_pid_write(path, data):
        raise TimeoutError("pid_write_timeout")

    monkeypatch.setattr(platform_service.subprocess, "Popen", tracking_popen)
    monkeypatch.setattr(platform_service, "_write", fail_pid_write)
    sleeper = tmp_path / "sleeper"
    sleeper.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n", encoding="utf-8")
    sleeper.chmod(0o755)
    with pytest.raises(TimeoutError, match="pid_write_timeout"):
        platform_service._linux_start_worker(manifest, sleeper, [], config_sha256="a" * 64)
    assert len(spawned) == 1
    assert spawned[0].poll() is not None


def test_powershell_encoded_prefix_binds_utf8_standard_streams():
    captured = []

    def fake_runner(argv, **kwargs):
        captured.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "ok\n", "")

    platform_service._ps(fake_runner, "'ok'", action="test.ps")
    argv, _ = captured[0]
    decoded = base64.b64decode(argv[-1]).decode("utf-16-le")
    assert "$u=[Text.UTF8Encoding]::new($false);$OutputEncoding=$u;" in decoded
    assert "[Console]::OpenStandardOutput()" in decoded
    assert "[Console]::SetOut(" in decoded
    assert "[Console]::OpenStandardError()" in decoded
    assert "[Console]::SetError(" in decoded


@pytest.mark.skipif(os.name != "nt", reason="Real Windows headless PowerShell 5.1 UTF-8 check")
def test_windows_detached_headless_powershell_51_preserves_utf8_non_ascii():
    code = (
        "import json, subprocess; from mindie_diagnostics import platform_service as p; "
        "r = p._ps(subprocess.run, '@{sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value;"
        "label=\"\\u4e2d\\u6587\\u8def\\u5f84\\u2014\\u03bcs\"}|ConvertTo-Json -Compress', action='sid'); "
        "d = json.loads(r.stdout); "
        "assert d['sid'].startswith('S-1-') and d['label'] == '\\u4e2d\\u6587\\u8def\\u5f84\\u2014\\u03bcs', d; "
        "print('UTF8_OK')"
    )
    flags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        creationflags=flags,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "UTF8_OK" in proc.stdout


@pytest.mark.skipif(os.name == "nt", reason="POSIX native lookup and DrvFS symlink resolution")
def test_round2_systemctl_rejection_and_symlink_drvfs_and_stat_timeout(tmp_path, monkeypatch):
    # 1. Rejected PE/DrvFS systemctl shim must raise command_unavailable rather than invoking bare 'systemctl'
    shim_dir = tmp_path / "drvfs_bin"
    shim_dir.mkdir()
    pe_systemctl = shim_dir / "systemctl"
    pe_header = bytearray(128)
    pe_header[0:2] = b"MZ"
    pe_header[0x3C:0x40] = (64).to_bytes(4, "little")
    pe_header[64:68] = b"PE\x00\x00"
    pe_systemctl.write_bytes(bytes(pe_header))
    pe_systemctl.chmod(0o755)
    with pytest.MonkeyPatch.context() as mp_path:
        mp_path.setenv("PATH", str(shim_dir))
        invoked = []

        def real_like_runner(argv, **kwargs):
            invoked.append(argv)
            return subprocess.CompletedProcess(argv, 0, "", "")

        real_like_runner._is_real_runner = True
        with pytest.raises(service.ServiceError) as exc_info:
            service._systemctl(real_like_runner, "show", "mindie-diagnostics.service", check=False)
        assert exc_info.value.category == "command_unavailable"
        assert invoked == []

    # 2. Strict realpath resolution in is_wsl_drvfs propagates os.lstat TimeoutError
    uid = os.geteuid()
    drvfs_root = tmp_path / "mnt_c"
    drvfs_root.mkdir()
    fake_mountinfo = tmp_path / "mountinfo"
    fake_mountinfo.write_text(
        f"20 1 8:1 / / rw,relatime - ext4 /dev/sda1 rw\n"
        f"90 20 0:42 / {drvfs_root} rw,noatime - 9p C:\\134 rw,dirsync,aname=drvfs;path=C:\\134;uid={uid};gid={uid};symlinkroot=/mnt/\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(filesystem, "sys", type("S", (), {"platform": "linux"})())
    monkeypatch.setattr(filesystem, "_MOUNTINFO_PATH", str(fake_mountinfo))
    assert filesystem.is_wsl_drvfs(drvfs_root / "gh") is True
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(filesystem.os, "lstat", lambda p, *a, **kw: (_ for _ in ()).throw(TimeoutError("lstat_deadline")))
        with pytest.raises(TimeoutError, match="lstat_deadline"):
            filesystem.is_wsl_drvfs(drvfs_root / "link" / "gh")

    # 3. TimeoutError during os.stat inside which_native must propagate
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(filesystem.os, "stat", lambda p: (_ for _ in ()).throw(TimeoutError("stat_deadline")))
        with pytest.raises(TimeoutError, match="stat_deadline"):
            filesystem.which_native("gh", path=str(tmp_path))

    # 4. Insufficient remaining deadline fails closed BEFORE signaling worker or unlinking worker-process.json
    assert platform_service._WORKER_STOP_DRAIN >= 5.0
    manifest = tmp_path / "mindie-diagnostics-service.json"
    manifest.write_text("{}", encoding="utf-8")
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        identity = platform_service._linux_worker_identity(sleeper.pid)
        expected_prefix = "proc:" if sys.platform.startswith("linux") else "ps:"
        assert identity and identity.startswith(expected_prefix)
        pid_file = platform_service._pid_path(manifest)
        pid_file.write_text(
            json.dumps({"manifest": str(manifest), "pid": sleeper.pid, "identity": identity, "config_sha256": "a" * 64}),
            encoding="utf-8",
        )
        with pytest.raises(service.ServiceError) as stop_exc:
            platform_service._linux_stop_worker(manifest, {"ok": True}, deadline=time.monotonic() + 0.25)
        assert stop_exc.value.category == "command_timeout"
        assert sleeper.poll() is None
        assert pid_file.is_file()
    finally:
        sleeper.terminate()
        sleeper.wait(timeout=5)


def test_reporter_reexports_create_no_window_guard():
    from mindie_diagnostics import github_transport, reporter

    assert reporter._CREATE_NO_WINDOW == github_transport._CREATE_NO_WINDOW
    assert reporter._CREATE_NO_WINDOW == getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
