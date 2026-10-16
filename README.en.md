<h1 align="center">MindIE Agent Diagnostics (<code>mindie-diagnostics</code>)</h1>

<p align="center">
  <strong>Bounded local structured logs, offline diagnostic bundles, and opt-in GitHub Issue reporting for MindIE Agent components</strong>
</p>

<p align="center">
  <a href="./README.md">中文</a> · <strong>English</strong> · <a href="./docs/README.en.md">Documentation</a> · <a href="https://github.com/mindie-agent/mindie-agent">MindIE Agent Main Repo</a>
</p>

---

## Overview

`mindie-diagnostics` is the shared observability and fault-reporting package of [MindIE Agent](https://github.com/mindie-agent/mindie-agent) (distribution `mindie-diagnostics`, import package `mindie_diagnostics`). It provides model-free local structured log rotation, failure attribution, offline redacted diagnostic bundle export, and user-authorized automatic GitHub Issue reporting across `remote-dev`, `coordinator`, `npu-top`, `knowledge`, and native client adapters.

- **100% Model-Free Execution Paths**: Diagnostics recording, maintenance, bundle export, and Issue reporting never invoke an LLM, never capture conversation transcripts, and never replay business commands.
- **Bounded Local Log Rotation**: Active writers use four fixed 1 MiB JSONL segments per process/component, recycling closed segments only after all registered readers have consumed them, with offline retention targets of 7 days, 128 MiB, and 512 files per root.
- **Strict Allowlist Privacy**: Public Issues and offline diagnostic bundles include only static component names, loaded versions/revisions, operation stages, static error categories, monotonic durations, exit/cleanup facts, and product code frames—excluding raw messages, commands, output, prompts, paths, endpoints, and credentials.
- **Explicit Opt-In Reporting**: Automatic GitHub Issue reporting is off by default, independent of knowledge contribution, and immediately revokes pending remote actions when disabled.

---

## Quick Start

### 1. Requirements & Installation

Requires Python `>= 3.11`. Automatic GitHub Issue reporting additionally requires an authenticated `gh` CLI installation.

```bash
uv pip install -e .
```

### 2. Reporting Status & Configuration (CLI)

Native client adapters expose built-in reporting commands; standalone components can invoke the CLI directly:

```bash
# Inspect read-only status (recent local incidents, authorization, queue, and worker health)
python -m mindie_diagnostics.cli reporting status

# Enable automatic reporting (default repository: mindie-agent/mindie-agent)
python -m mindie_diagnostics.cli reporting configure --enabled true

# Prepare the isolated reporter runtime and start the background worker
python -m mindie_diagnostics.cli reporting ensure

# Disable automatic reporting (immediately invalidates pending remote actions)
python -m mindie_diagnostics.cli reporting configure --enabled false
```

### 3. Log Maintenance & Offline Bundle Export

```bash
# Run offline log retention maintenance
python -m mindie_diagnostics.cli reporting maintain

# Upgrade a healthy running worker to a newer semantic version (never revives stopped workers)
python -m mindie_diagnostics.cli reporting maintain --update-running

# Export a local, redacted diagnostic bundle for a specific operation ID (offline only)
python -m mindie_diagnostics.cli bundle --root /absolute/log-root --operation-id INCIDENT_ID
```

---

## Python Integration

Record failures only at the component boundary that owns the failure:

```python
from mindie_diagnostics.integration import record_failure

result = record_failure(
    "remote-dev",
    "mcp.request",
    stage="decode",
    category="protocol_mismatch",
    exception=error,
)
```

- Pass static product metadata—never pass user input as `category` or `operation`.
- Ordinary caller errors, missing configuration, permissions, network/auth failures, and non-zero business command exits are not product defects and remain local with `reportable=False`.
- Normal operations can use `configure(...).operation(...)` to record bounded stages, monotonic timings, and cleanup evidence.

---

## Configuration Reference

| Variable / Path | Default | Purpose |
| --- | --- | --- |
| `MINDIE_DIAGNOSTICS_CONFIG` | `~/.config/mindie-agent/diagnostics.json` (or `$XDG_CONFIG_HOME/mindie-agent/diagnostics.json`) | Reporting policy path (queue, receipts, health, and isolated runtime live alongside this file) |
| `MINDIE_DIAGNOSTICS_ROOT` | Platform user state directory | Local structured diagnostic log root |
| `MINDIE_LOG_LEVEL` | `INFO` | Log verbosity (`DEBUG` / `INFO` / `WARNING` / `ERROR`) |

---

## Documentation

| Topic | 中文 | English |
| --- | --- | --- |
| Documentation Index | [docs/README.md](./docs/README.md) | [docs/README.en.md](./docs/README.en.md) |
| Architecture, Privacy & Log Retention | [docs/architecture-and-privacy.md](./docs/architecture-and-privacy.md) | [docs/architecture-and-privacy.en.md](./docs/architecture-and-privacy.en.md) |

---

## Development & Testing

```bash
python -m pytest
```

---

## License

Released under the [MIT License](./LICENSE).
