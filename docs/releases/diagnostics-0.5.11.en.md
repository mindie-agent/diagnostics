# mindie-diagnostics 0.5.11: Dual-Key Deduplication, Triage Card v2 & Centralized Filing

[中文](diagnostics-0.5.11.md)

Upgrades the automatic fault-diagnostics and reporting pipeline (Diagnostics 2.0), centralizing default GitHub Issue filing into `mindie-agent/mindie-agent` while eliminating cross-version duplicate Issues and adding commit-exact root-cause localization:

- **Dual-Key Deduplication & Occurrence/Regression Comments (`fingerprint` + `fault_signature`)**: Retains per-build `fingerprint` local outbox deduplication while adding a version-independent `fault_signature` (invariant across package versions, revisions, and line-number drift). Matches against `OPEN` Issues append an **Additional Occurrence** comment and matches against `CLOSED` Issues append a **Regression Observed After Issue Closure** comment instead of opening duplicate Issues, with deletion-resilient multi-cycle pagination continuity.
- **Triage Card v2 & Exact-Commit Source Permalinks**: Issue bodies and occurrence comments render structured root-cause triage fields (`reason_code`, `substage`, `contract_name`, `soc_generation`, `harness`, `diagnostics_version`, and bounded `phases`) plus exact 40-SHA GitHub source permalinks into the owning component/plugin repository (distinguishing `module.py` from `module/__init__.py`).
- **64-Link Causal-Chain Traversal & Atomic Build Metadata**: Both `record_failure` and `Operation` logging walk up to 64 causal exception links and mark unvisited tails `stack_incomplete=True` so wrapper frames are never misattributed as fault sites. Explicit partial build metadata or `infer_build_meta=False` suppresses parent-environment enrichment.
- **Zero-Free-Text Privacy Boundary & Local Invocation Token**: `MINDIE_DIAGNOSTICS_INVOCATION_TOKEN` binds child crash receipts strictly locally and is stripped before bundle export or GitHub publication without inflating `attributes_omitted`; exception messages, `__notes__`, paths, environment variables, and credentials remain excluded.

The package `__version__` is synchronized to `0.5.11`. Actual test and benchmark results for this release are recorded in the bundled `benchmark-results.json`, `benchmark-results.md`, and CI runs.
