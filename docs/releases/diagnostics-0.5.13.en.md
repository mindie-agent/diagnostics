# mindie-diagnostics 0.5.13

## Fixes & Compatibility Hardening

- **Windows Zero-Console-Popup Gate (`R4`) Export Alignment**: Explicitly define `_CREATE_NO_WINDOW` in `mindie_diagnostics.github_transport` and re-export it from `mindie_diagnostics.reporter`, restoring compatibility with `check_auto_update_readiness.py` `R4` validation after the `v0.5.11` `github_transport.py` module split.
