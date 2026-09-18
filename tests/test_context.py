import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys

from mindie_diagnostics import bind_context, current_context, wrap_context


def test_context_is_only_diagnostic_ids_and_restored(monkeypatch):
    monkeypatch.delenv("MINDIE_DIAGNOSTICS_CONTEXT", raising=False)
    assert current_context() == {}
    with bind_context({"trace_id": "a" * 32, "operation_id": "b" * 32, "task_id": "other-task", "user": "someone"}):
        assert current_context() == {"trace_id": "a" * 32, "operation_id": "b" * 32}
        with bind_context({"trace_id": "c" * 32}):
            assert current_context() == {"trace_id": "c" * 32}
        assert current_context()["trace_id"] == "a" * 32
    assert current_context() == {}


def test_thread_helper_supports_concurrent_invocations():
    with bind_context({"trace_id": "a" * 32}):
        wrapped = wrap_context(current_context)
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(lambda _: wrapped(), range(12))) == [{"trace_id": "a" * 32}] * 12


def test_async_contexts_do_not_leak():
    async def job(char):
        with bind_context({"trace_id": char * 32}):
            await asyncio.sleep(0)
            return current_context()
    async def run():
        return await asyncio.gather(job("a"), job("b"))
    assert asyncio.run(run()) == [{"trace_id": "a" * 32}, {"trace_id": "b" * 32}]


def test_environment_round_trip_and_bad_input(monkeypatch):
    env = {**os.environ, "MINDIE_DIAGNOSTICS_CONTEXT": json.dumps({"trace_id": "a" * 32, "token": "never"})}
    result = subprocess.run([sys.executable, "-c", "import json; from mindie_diagnostics import current_context; print(json.dumps(current_context()))"],
                            env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {"trace_id": "a" * 32}
    for raw in ("[]", "{broken", '{"trace_id": "not-id"}', "x" * 2049):
        monkeypatch.setenv("MINDIE_DIAGNOSTICS_CONTEXT", raw)
        assert current_context() == {}
