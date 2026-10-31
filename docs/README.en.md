# `mindie-diagnostics` Documentation

[中文](./README.md) · **English** · [Back to Project Homepage](../README.en.md)

Welcome to the `mindie-diagnostics` documentation hub. Here you will find details on how privacy allowlisting, failure attribution, user consent handling, and bounded ring-buffer logging work.

## Guides & References

| Topic | 中文 | English | What You'll Learn |
| --- | --- | --- | --- |
| Architecture, Privacy Allowlist & Log Rotation | [architecture-and-privacy.md](./architecture-and-privacy.md) | [architecture-and-privacy.en.md](./architecture-and-privacy.en.md) | What data is allowlisted vs. stripped, how caller errors are separated from product bugs, how consent and outbox storage are guarded, and how 4 × 1 MiB log rotation works |
| Bounded Outbound Reconciliation & Prune Budget Alignment | [outbound-reconciliation-followup.md](./outbound-reconciliation-followup.md) | [outbound-reconciliation-followup.en.md](./outbound-reconciliation-followup.en.md) | Why uncertain POSTs are never replayed, how `attempt_timestamp` + positive search + bounded listing reconcile within the 8 MiB cycle budget, and how `prune` aligns with reader cursor budgets |
