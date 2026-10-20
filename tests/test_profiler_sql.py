"""Tier-1 tests for the read-only bounded SQL surface."""

from __future__ import annotations

import pytest

from mindie_profiler.events import ProfilerQueryError
from mindie_profiler.sql import run_sql
from tests.test_profiler_core import build_db, rank0_events


@pytest.fixture()
def db_path(tmp_path):
    return build_db(tmp_path / "rank_0" / "ascend_pytorch_profiler_0.db", rank0_events())


def test_select_resolves_string_ids_additively(db_path):
    result = run_sql(db_path, "SELECT name, blockDim FROM COMPUTE_TASK_INFO ORDER BY globalTaskId LIMIT 2")
    assert result["row_count"] == 2
    assert isinstance(result["rows"][0]["name"], int)
    assert result["rows"][0]["name_resolved"] == "aclnnFlashAttention"
    assert "blockDim" in result["columns"]


@pytest.mark.parametrize(
    "query",
    [
        "INSERT INTO STRING_IDS VALUES (999, 'x')",
        "UPDATE META_DATA SET value='9.9.9'",
        "DELETE FROM TASK",
        "PRAGMA table_info(TASK)",
        "ATTACH DATABASE ':memory:' AS other",
        "SELECT load_extension('evil')",
    ],
)
def test_write_ddl_pragma_attach_and_extension_are_rejected(db_path, query):
    with pytest.raises(ProfilerQueryError):
        run_sql(db_path, query, timeout_s=2.0)


def test_cte_delete_is_blocked_by_authorizer(db_path):
    with pytest.raises(ProfilerQueryError):
        run_sql(db_path, "WITH doomed AS (SELECT 1) DELETE FROM TASK", timeout_s=2.0)


def test_row_limit_overflow_is_explicit_not_truncated(db_path):
    with pytest.raises(ProfilerQueryError, match="row limit"):
        run_sql(db_path, "SELECT id FROM STRING_IDS", limit=2, timeout_s=2.0)


def test_progress_handler_interrupts_a_runaway_query(db_path):
    query = (
        "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt) "
        "SELECT x FROM cnt"
    )
    with pytest.raises(ProfilerQueryError, match="deadline"):
        run_sql(db_path, query, limit=10_000_000, timeout_s=0.05)


def test_database_file_is_not_modified(db_path):
    before = db_path.read_bytes()
    run_sql(db_path, "SELECT COUNT(*) AS n FROM TASK")
    assert db_path.read_bytes() == before
