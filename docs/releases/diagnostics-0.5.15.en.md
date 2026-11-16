# mindie-diagnostics 0.5.15

## Update Failure Policy Classification Hardening

- **DNS Resolver Failures Classified as Connectivity (`resolver`)**: `update_failure_policy()` now maps `category="resolver"` from subprocess/transport classifiers to `("connectivity", False, "update_transport_unavailable")`, preventing local DNS or offline failures from falling through to `internal / update_internal_failure` auto-filed GitHub Issues.
- **Lock Contention & Prerequisite Deferrals Marked Local-Only (`busy` / `deferred` / `prerequisite`)**: `update_failure_policy()` recognizes `category in {"busy", "deferred", "prerequisite"}` across the exception chain as `("prerequisite", False, "update_prerequisite_unavailable")` (`reportable=False`), keeping concurrent update lock contention visible locally without filing false internal bug reports.
- **Version**: `0.5.15`.
