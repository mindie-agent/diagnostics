# `mindie-diagnostics` Documentation

[中文](./README.md) · **English** · [Back to Project Homepage](../README.en.md)

Welcome to the `mindie-diagnostics` documentation hub. Here you will find the living specifications for allowlist-only privacy projection, co-owned stack attribution, Scale-to-Zero on-demand queue draining, consent separation, and bounded ring-buffer logging.

## Guides & References

| Topic | 中文 | English | What You'll Learn |
| --- | --- | --- | --- |
| Architecture, Scale-to-Zero Drain, Privacy Allowlist & Log Rotation | [architecture-and-privacy.md](./architecture-and-privacy.md) | [architecture-and-privacy.en.md](./architecture-and-privacy.en.md) | Allowlisted vs. stripped fields, `mindie-harness` co-owned stack frames, Scale-to-Zero on-demand worker lifecycle (zero 24/7 idle daemon or 15s heartbeat `fsync`s), `7d` expiry strictly for unsent `pending` rows while preserving `uncertain` POSTs and dedup receipts, and 4 × 1 MiB log rotation |
| Archived Historical Notes | [archive/README.md](./archive/README.md) | [archive/README.md](./archive/README.md) | Historical outbound reconciliation follow-up notes preserved for provenance |
