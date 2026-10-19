"""[Role] Measure a repeatable public-input component workload for releases.
[Invariants] Assert business output before reporting latency; no productivity claim.
[Timeouts & Locks] Bounded iterations, no network or background services.
[Side-Effect Policy] Temporary synthetic files and an explicit result JSON only.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mindie_diagnostics.redact import redact_text, scan_text
text = ("torch_npu 2.10.0: Authorization: Bearer abcDEF123456\n"
        "Trace in /home/exampleuser/project; NPU index 4, 8192x8192 matmul.\n") * 64
sanitized = redact_text(text)
assert "abcDEF123456" not in sanitized and "torch_npu" in sanitized
assert not scan_text(sanitized)
iterations = 100
started = time.perf_counter()
for _ in range(iterations):
    result = redact_text(text)
elapsed = time.perf_counter() - started
assert result == sanitized
METRIC = "redact_8k_latency"

Path(os.environ["MINDIE_BENCH_RESULT_PATH"]).write_text(json.dumps({
    "status": "pass", "metrics": [{"name": METRIC, "unit": "us/op",
    "value": elapsed * 1_000_000 / iterations, "iterations": iterations}]
}) + "\n", encoding="utf-8")
