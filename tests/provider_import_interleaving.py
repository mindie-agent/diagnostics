"""[Role] Reproduce a competing actual TensorCast import at lock acquisition.
[Invariants] Both source arguments are clean checkouts of the actual pinned upstream.
[Timeouts & Locks] The parent bounds this process; the real import lock stays in use.
[Side-Effect Policy] Imports in this isolated child only; never writes upstream sources.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mindie_profiler import tensor_cast_provider as module  # noqa: E402
from mindie_profiler.theoretical_model import KernelInputs, ProviderUnmodeledError  # noqa: E402


def main() -> None:
    first_source, second_source = sys.argv[1:]
    original_lock = module._IMPORT_LOCK
    executed = False

    class CompetingImport:
        def __enter__(self):
            nonlocal executed
            # Another constructor completes while this constructor awaits the lock.
            module._IMPORT_LOCK = original_lock
            first = module.TensorCastAnalyticProvider(first_source, device_name="TEST_DEVICE")
            first.predict(KernelInputs(
                shape=(8, 16, 4), dtype="bfloat16", layout="row_major",
                hardware=first.hardware_spec("bfloat16", "matmul"), op_kind="matmul",
            ))
            executed = first.executions == 1
            original_lock.acquire()

        def __exit__(self, *args):
            original_lock.release()

    module._IMPORT_LOCK = CompetingImport()
    try:
        module.TensorCastAnalyticProvider(second_source, device_name="TEST_DEVICE")
    except ProviderUnmodeledError as exc:
        status = exc.reason
    else:
        status = "incorrectly_accepted_a_different_import_source"
    finally:
        module._IMPORT_LOCK = original_lock
    print(json.dumps({"first_upstream_executed": executed, "second_status": status}))
    if not executed or status != "tensor_cast_already_imported_from_other_source":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
