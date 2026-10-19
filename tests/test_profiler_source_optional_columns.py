"""Absent optional compute columns must not invalidate supported DB queries."""
import sqlite3

import pytest

from mindie_profiler import analyze_profile
from test_profiler_core import STEP4, build_db, rank0_events


@pytest.mark.parametrize("columns", [("blockDim",), ("mixBlockDim",), ("blockDim", "mixBlockDim")])
def test_optional_block_columns_preserve_interval_accounting(tmp_path, columns):
    source = build_db(tmp_path / "rank_0" / "ascend_pytorch_profiler.db", rank0_events())
    with sqlite3.connect(source) as connection:
        for column in columns:
            connection.execute(f"ALTER TABLE COMPUTE_TASK_INFO DROP COLUMN {column}")
    report = analyze_profile(source, windows=[STEP4])
    assert report["window_accounting"][0]["busy_union_ms"] == 0.04
    assert report["window_accounting"][0]["kernel_sum_ms"] == 0.061
