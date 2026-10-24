# `mindie-diagnostics` Documentation

[中文](./README.md) · **English** · [Back to Project Homepage](../README.en.md)

Welcome to the `mindie-diagnostics` documentation hub. Here you will find details on how privacy allowlisting, failure attribution, user consent handling, and bounded ring-buffer logging work.

## Guides & References

| Topic | 中文 | English | What You'll Learn |
| --- | --- | --- | --- |
| Architecture, Privacy Allowlist & Log Rotation | [architecture-and-privacy.md](./architecture-and-privacy.md) | [architecture-and-privacy.en.md](./architecture-and-privacy.en.md) | What data is allowlisted vs. stripped, how caller errors are separated from product bugs, how consent and outbox storage are guarded, and how 4 × 1 MiB log rotation works |

## Design Follow-Ups (Not Implemented)

| Topic | 中文 | English | What You'll Learn |
| --- | --- | --- | --- |
| Bounding the Outbound Reconciliation Read | [outbound-reconciliation-followup.md](./outbound-reconciliation-followup.md) | [outbound-reconciliation-followup.en.md](./outbound-reconciliation-followup.en.md) | Why the 8 MiB cycle budget and the 20-page reconciliation window contradict each other, why a search miss cannot prove absence, and which single-owner intent metadata a bounded window needs |
