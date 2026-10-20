"""Tier 2 standing A2/A3 live NPU hardware-probe integration.

These tests are opt-in with ``MINDIE_SIM_LIVE_NPU=1`` so an offline CI runner
does not silently pretend to have hardware.  When enabled, each host gets a
unique ``/tmp/mindie-sim-test-<uuid>`` sandbox, the trap cleans it on exit, and
the local assertions run against the real ``npu-smi`` and CANN ``.ini`` output.
"""
from __future__ import annotations

import os
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

from mindie_sim import hardware as hw
from mindie_sim import subgraph as sg

pytestmark = [
    pytest.mark.live_npu,
    pytest.mark.skipif(
        os.environ.get("MINDIE_SIM_LIVE_NPU") != "1",
        reason="set MINDIE_SIM_LIVE_NPU=1 to run standing A2/A3 live tests",
    ),
]

A2_HOST = os.environ.get("MINDIE_SIM_A2_HOST", "root@173.125.1.2")
A3_HOST = os.environ.get("MINDIE_SIM_A3_HOST", "root@192.168.13.165")
SSH_TIMEOUT_S = 12.0
SSH_OPTIONS = (
    "-o", "ConnectTimeout=8",
    "-o", "BatchMode=yes",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "StrictHostKeyChecking=no",
    "-o", "LogLevel=ERROR",
)
MARKERS = ("@@@INFO@@@", "@@@COMMON@@@", "@@@BOARD@@@", "@@@TOPO@@@", "@@@INI@@@")


class LiveNpuError(RuntimeError):
    """A standing A2/A3 probe failed or returned an unexpected schema."""


def _ssh(host: str, command: str, *, timeout: float = SSH_TIMEOUT_S, check: bool = True) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        ["ssh", *SSH_OPTIONS, host, command],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        raise LiveNpuError(f"ssh_failed:{host}:{' '.join(detail[-3:])}")
    return completed


def _sections(text: str) -> dict[str, str]:
    lines = text.splitlines()
    result: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in lines:
        if line in MARKERS:
            if current is not None:
                result[current] = "\n".join(buffer).strip() + "\n"
            current = line
            buffer = []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        result[current] = "\n".join(buffer).strip() + "\n"
    missing = [marker for marker in MARKERS if marker not in result]
    if missing:
        raise LiveNpuError("live_capture_missing_sections:" + ",".join(missing))
    return result


@contextmanager
def _capture(host: str, ini_name: str) -> Iterator[dict[str, Any]]:
    sandbox = f"/tmp/mindie-sim-test-{uuid.uuid4().hex}"
    if not sandbox.startswith("/tmp/mindie-sim-test-") or len(Path(sandbox).name) != len("mindie-sim-test-") + 32:
        raise LiveNpuError("sandbox_name_invalid")
    script = f"""
set -eu
sandbox={sandbox}
mkdir -p "$sandbox"
trap 'rm -rf "$sandbox"' EXIT
npu-smi info > "$sandbox/info.txt"
npu-smi info -t common -i 0 > "$sandbox/common.txt"
npu-smi info -t board -i 0 > "$sandbox/board.txt"
npu-smi info -t topo > "$sandbox/topo.txt"
ini=$(find /usr/local/Ascend -path '*platform_config/{ini_name}.ini' 2>/dev/null | head -1 || true)
if [ -z "$ini" ]; then
  ini=$(find /usr/local/Ascend -path '*platform_config/Ascend910*.ini' 2>/dev/null | head -1 || true)
fi
if [ -z "$ini" ] || [ ! -f "$ini" ]; then
  echo "platform_ini_not_found" >&2
  exit 4
fi
cp "$ini" "$sandbox/platform.ini"
echo '@@@INFO@@@'
cat "$sandbox/info.txt"
echo '@@@COMMON@@@'
cat "$sandbox/common.txt"
echo '@@@BOARD@@@'
cat "$sandbox/board.txt"
echo '@@@TOPO@@@'
cat "$sandbox/topo.txt"
echo '@@@INI@@@'
cat "$sandbox/platform.ini"
""".strip()
    try:
        completed = _ssh(host, script)
        capture = {
            "host": host,
            "sandbox": sandbox,
            "sections": _sections(completed.stdout),
        }
        yield capture
    finally:
        # The remote trap normally removes the sandbox.  This best-effort
        # second cleanup also covers a dropped connection after mkdir.
        _ssh(host, f"rm -rf {sandbox} && test ! -e {sandbox}", check=False, timeout=10.0)


@pytest.fixture(scope="module")
def a2_capture() -> Iterator[dict[str, Any]]:
    with _capture(A2_HOST, "Ascend910B4") as capture:
        yield capture


@pytest.fixture(scope="module")
def a3_capture() -> Iterator[dict[str, Any]]:
    with _capture(A3_HOST, "Ascend910_9391") as capture:
        yield capture


def _parse_capture(capture: dict[str, Any]) -> tuple[hw.PlatformConfig, hw.NpuSmiSnapshot]:
    sections = capture["sections"]
    config = hw.parse_platform_ini_text(sections["@@@INI@@@"], source=f"{capture['host']}:platform.ini")
    snapshot = hw.parse_npu_smi_snapshot(
        sections["@@@INFO@@@"],
        common_text=sections["@@@COMMON@@@"],
        board_text=sections["@@@BOARD@@@"],
        source=capture["host"],
    )
    return config, snapshot


def test_a2_live_npu_smi_and_cann_platform_cross_check(a2_capture):
    config, snapshot = _parse_capture(a2_capture)
    assert config.short_soc_version == "Ascend910B"
    assert config.ai_core_cnt == 20
    assert config.platform_config_hbm_mib == 32768.0
    assert config.l2_size_bytes == 96 * 1024 * 1024
    assert config.platform_allocatable_ub_kib == 192.0
    assert snapshot.chip_count == 1
    assert len(snapshot.devices) == 8
    assert snapshot.hbm_totals_mib == (32768,)
    assert snapshot.chips and snapshot.chips[0].aicore_count == 20

    cross = hw.cross_check_capacities(
        config,
        snapshot,
        simulator_ub_kib=None,
        sustained_mhz_range=(1250, 1300),
    )
    assert cross["platform_config_hbm_mib"] == 32768.0
    assert cross["verified_runtime_hbm_mib"] == 32768.0
    assert cross["allocatable_ub_kib"] == 192.0
    assert cross["capacity_authority"] == "verified_runtime"
    assert "runtime_configured_clock_differs_from_platform_config" in cross["flags"]
    assert "platform_clock_is_not_sustained_clock" in cross["flags"]

    report = hw.hardware_report(config, snapshot=snapshot, simulator_ub_kib=None)
    assert report["status"] == "modeled"
    assert report["capacity_cross_check"]["verified_runtime_hbm_mib"] == 32768.0
    assert report["soc"]["platform_config_hbm_mib"] == 32768.0


def test_a3_live_dual_die_and_cann_platform_cross_check(a3_capture):
    config, snapshot = _parse_capture(a3_capture)
    assert config.short_soc_version == "Ascend910_93"
    assert config.npu_arch == "2201"
    assert config.ai_core_cnt == 24
    assert config.platform_config_hbm_mib == 65536.0
    assert config.l2_size_bytes == 192 * 1024 * 1024
    assert config.cube_freq_mhz == 1850.0
    assert snapshot.chip_count == 2
    assert snapshot.dual_die is True
    assert len(snapshot.devices) == 16
    assert snapshot.phy_ids == tuple(range(16))
    assert snapshot.hbm_totals_mib == (65536,)
    assert [chip.aicore_count for chip in snapshot.chips] == [20, 20]

    cross = hw.cross_check_capacities(
        config,
        snapshot,
        simulator_ub_kib=None,
        sustained_mhz_range=(1250, 1300),
    )
    assert cross["platform_config_hbm_mib"] == 65536.0
    assert cross["verified_runtime_hbm_mib"] == 65536.0
    assert "runtime_aicore_count_differs_from_platform_config" in cross["flags"]
    assert "runtime_configured_clock_differs_from_platform_config" in cross["flags"]
    assert "platform_clock_is_not_sustained_clock" in cross["flags"]

    report = hw.hardware_report(config, snapshot=snapshot, simulator_ub_kib=None)
    assert report["status"] == "modeled"
    assert report["capacity_cross_check"]["verified_runtime_hbm_mib"] == 65536.0
    assert report["capacity_cross_check"]["platform_config_hbm_mib"] == 65536.0


def test_live_subgraph_capacity_gates_use_measured_cluster_ub(a2_capture, a3_capture):
    for capture in (a2_capture, a3_capture):
        config, snapshot = _parse_capture(capture)
        capacities = {
            "L1": config.l1_kib,
            "L0A": config.l0_a_kib,
            "L0B": config.l0_b_kib,
            "L0C": config.l0_c_kib,
            "UB": config.platform_allocatable_ub_kib,
        }
        dag = {
            "name": f"live-{config.short_soc_version}",
            "nodes": [
                {"id": "producer", "kind": "matmul", "duration_us": None},
                {"id": "cast", "kind": "aiv_cast", "duration_us": None},
            ],
            "edges": [
                {"src": "producer", "dst": "cast", "tensor": {"bytes": 4.5 * 1024 * 1024}},
            ],
            "capacities_kib": capacities,
        }
        candidate = {
            "producer": "producer",
            "consumer": "cast",
            "remove_nodes": ["cast"],
            "remove_edges": [{"src": "producer", "dst": "cast"}],
            "live_buffers": [{"name": "live UB scratch", "buffer": "UB", "size_kib": 180}],
        }
        fits = sg.analyze_subgraph_fusion(dag, candidate)
        assert fits["removable_gm_bytes"] == 9 * 1024 * 1024
        assert fits["feasibility"]["feasible"] is True
        assert fits["status"] == "unmodeled_latency"
        assert fits["latency"]["max_saving_us"] is None
        assert fits["buffer_capacities_kib"]["UB"] == config.platform_allocatable_ub_kib

        overflow_candidate = dict(candidate)
        overflow_candidate["live_buffers"] = [{"name": "live UB scratch", "buffer": "UB", "size_kib": 200}]
        overflow = sg.analyze_subgraph_fusion(dag, overflow_candidate)
        assert overflow["status"] == "infeasible"
        assert "live_buffer_limit" in overflow["feasibility"]["failures"]
        assert snapshot.hbm_totals_mib == (int(config.platform_config_hbm_mib),)


def test_live_a2_and_a3_interconnect_topology_and_transfer_channels(a2_capture, a3_capture):
    a2_config, a2_snapshot = _parse_capture(a2_capture)
    a2_topo = hw.parse_npu_smi_topo(a2_capture["sections"]["@@@TOPO@@@"], source=f"{a2_capture['host']}:topo")
    assert a2_topo.has_sio_intra_package is False
    assert a2_topo.intra_node_fabric == "HCCS"
    assert len(a2_topo.numa_groups) == 4
    a2_prof = hw.resolve_device_capability_profile(config=a2_config, snapshot=a2_snapshot, topo_snapshot=a2_topo)
    assert a2_prof.memory_hierarchy_spec.transfer_channels["d2d_intra_package"].effective_uni_bw_gbps is None
    assert a2_prof.memory_hierarchy_spec.transfer_channels["d2d_intra_node"].protocol == "HCCS"

    a3_config, a3_snapshot = _parse_capture(a3_capture)
    a3_topo = hw.parse_npu_smi_topo(a3_capture["sections"]["@@@TOPO@@@"], source=f"{a3_capture['host']}:topo")
    assert a3_topo.has_sio_intra_package is True
    assert a3_topo.intra_node_fabric == "HCCS_SW"
    assert a3_topo.link_type_counts.get("SIO") == 16
    a3_prof = hw.resolve_device_capability_profile(config=a3_config, snapshot=a3_snapshot, topo_snapshot=a3_topo)
    assert a3_prof.memory_hierarchy_spec.transfer_channels["d2d_intra_package"].protocol == "SIO"
    assert a3_prof.memory_hierarchy_spec.transfer_channels["d2d_intra_node"].protocol == "HCCS_SW"

