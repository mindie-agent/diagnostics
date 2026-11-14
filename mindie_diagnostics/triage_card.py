"""Deterministic Triage Card v2 rendering, source permalinks, and fault signatures.

[Role]
Computes version-independent root-cause identities (`fault_signature`), maps owned
component stack frames to exact-commit GitHub source permalinks, and renders
structured Markdown triage cards (`render_triage_card`, `render_occurrence_card`).

[Invariants]
- Operates strictly on re-projected `issue_payload` dicts; never reads local files
  or accepts raw unprojected event dictionaries.
- Every rendered Markdown title, issue body, and occurrence comment passes
  `scan_text` (`REDACTION_PROFILE = "r2"`) and fits within GitHub's 60 KB limit.
- Avoids prose `key: value` patterns that collide with privacy assignment rules,
  using pipe-delimited Markdown tables instead.

[Timeouts & Locks]
- Pure CPU/memory formatting with zero locks, subprocesses, or I/O.

[Side-Effect Policy]
- Pure functions returning strings and SHA-256 hex digests.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .evidence import component_module_name
from .redact import scan_text

MARKER = "<!-- mindie-incident:"
FAULT_MARKER = "<!-- mindie-fault:"
_HEX_SHA_RE = re.compile(r"[0-9a-f]{7,40}\Z")
_MODULE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,159}\Z")

_COMPONENT_REPOS: dict[str, tuple[str, str]] = {
    "mindie-diagnostics": ("mindie-agent/diagnostics", "package"),
    "mindie-knowledge": ("mindie-agent/knowledge", "package"),
    "remote-dev": ("mindie-agent/remote-dev", "package"),
    "mindie-coordinator": ("mindie-agent/coordinator", "package"),
    "mindie-profiler": ("mindie-agent/profiler", "package"),
    "mindie-sim": ("mindie-agent/sim", "package"),
    "mindie-debugger": ("mindie-agent/debugger", "package"),
    "npu-top": ("mindie-agent/npu-top", "package"),
    "mindie-bench": ("mindie-agent/mindie-bench", "src"),
    "mindie-agent-codex": ("mindie-agent/mindie-agent-codex", "plugins/mindie-agent/scripts"),
    "mindie-agent-agy": ("mindie-agent/mindie-agent-agy", "plugins/mindie-agent/scripts"),
    "mindie-agent-wb": ("mindie-agent/mindie-agent-wb", "plugins/mindie-agent/scripts"),
    "mindie-agent-oc": ("mindie-agent/mindie-agent-oc", "plugins/mindie-agent/scripts"),
    "mindie-agent-cc": ("mindie-agent/mindie-agent-cc", "scripts"),
    "mindie-agent-kimi": ("mindie-agent/mindie-agent-kimi", "scripts"),
    "mindie-agent-dsh": ("mindie-agent/mindie-agent-dsh", "scripts"),
}


def owned_frames(component: str, raw_frames: Any) -> list[dict[str, Any]]:
    module = component_module_name(component)
    if not module or not isinstance(raw_frames, list):
        return []
    return [
        frame for frame in raw_frames
        if isinstance(frame, dict)
        and (frame.get("module") == module or str(frame.get("module", "")).startswith(module + "."))
    ]


def _failure_event(bundle: dict[str, Any]) -> dict[str, Any]:
    from .bundle import export_issue_event

    events = bundle.get("events", [])
    failures = [
        safe for event in events
        if (safe := export_issue_event(event)) is not None and safe.get("status") == "error"
    ]
    return failures[-1] if failures else {}


def legacy_fingerprint(bundle: dict[str, Any]) -> str:
    """Pre-0.5.11 per-build fingerprint (without v2 reason/contract/substage/error_code)."""
    event = _failure_event(bundle)
    attrs = event.get("attributes", {})
    component = event.get("component", "unknown")
    frames = [
        {k: v for k, v in frame.items() if k != "init"}
        for frame in owned_frames(component, attrs.get("stack_frames", []))
    ]
    identity = {
        "component": component,
        "package_version": event.get("package_version"),
        "package_revision": event.get("package_revision"),
        "operation": event.get("operation"),
        "stage": attrs.get("stage"),
        "category": attrs.get("category"),
        "error_type": attrs.get("error_type"),
        "frames": frames,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def build_fingerprint(bundle: dict[str, Any]) -> str:
    """Per-build public fault identity including v2 root-cause discriminators."""
    event = _failure_event(bundle)
    attrs = event.get("attributes", {})
    component = event.get("component", "unknown")
    frames = owned_frames(component, attrs.get("stack_frames", []))
    identity: dict[str, Any] = {
        "component": component,
        "package_version": event.get("package_version"),
        "package_revision": event.get("package_revision"),
        "operation": event.get("operation"),
        "stage": attrs.get("stage"),
        "category": attrs.get("category"),
        "error_type": attrs.get("error_type"),
        "frames": frames,
    }
    for key in ("substage", "error_code", "reason_code", "contract_name"):
        if attrs.get(key) is not None:
            identity[key] = attrs[key]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def item_public_fingerprint(item: dict[str, Any]) -> str:
    """Resolve an Outbox item's public fingerprint, preserving pre-0.5.11 rows."""
    from .reporting import scope_key

    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    saved = payload.get("public_fingerprint") if not payload.get("events") else None
    if isinstance(saved, str) and len(saved) == 64 and all(c in "0123456789abcdef" for c in saved):
        return saved
    current = build_fingerprint(payload)
    row_fp = item.get("fingerprint")
    consent = item.get("consent")
    if isinstance(row_fp, str) and row_fp not in {current, scope_key(current, consent)}:
        legacy = legacy_fingerprint(payload)
        if row_fp in {legacy, scope_key(legacy, consent)}:
            return legacy
    return current


def fault_signature(bundle: dict[str, Any]) -> str:
    """Version-independent root-cause hash for cross-release deduplication."""
    event = _failure_event(bundle)
    attrs = event.get("attributes", {})
    component = event.get("component", "unknown")
    incomplete = attrs.get("stack_incomplete") is True
    frames = [] if incomplete else [
        {
            "module": frame.get("module"),
            "function": frame.get("function"),
            **({"init": True} if frame.get("init") is True else {}),
        }
        for frame in owned_frames(component, attrs.get("stack_frames", []))[-1:]
    ]
    identity = {
        "component": component,
        "operation": event.get("operation"),
        "stage": attrs.get("stage"),
        "substage": attrs.get("substage"),
        "category": attrs.get("category"),
        "error_code": attrs.get("error_code"),
        "reason_code": attrs.get("reason_code"),
        "contract_name": attrs.get("contract_name"),
        "error_type": attrs.get("error_type"),
        "call_chain": frames,
        **({"stack_incomplete": True} if incomplete else {}),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def component_commit_url(component: str, revision: str | None) -> tuple[str | None, str | None]:
    spec = _COMPONENT_REPOS.get(component)
    if spec is None or not isinstance(revision, str) or _HEX_SHA_RE.fullmatch(revision) is None:
        return (spec[0] if spec else None), None
    repo = spec[0]
    return repo, f"https://github.com/{repo}/tree/{revision}"


def component_source_permalink(
    component: str,
    revision: str | None,
    module: str | None,
    line: int | None,
    *,
    is_init: bool = False,
) -> tuple[str | None, str | None]:
    """Resolve an owned Python frame to `(repo_rel_path, github_permalink_url)`."""
    spec = _COMPONENT_REPOS.get(component)
    if spec is None or not isinstance(module, str) or _MODULE_RE.fullmatch(module) is None:
        return None, None
    repo, layout = spec
    pkg = component_module_name(component)
    if not pkg:
        return None, None
    clean_mod = module[:-9] if module.endswith(".__init__") else module
    pkg_init = is_init or module.endswith(".__init__")
    if layout == "package":
        if clean_mod == pkg:
            rel_path = f"{pkg}/__init__.py"
        elif clean_mod.startswith(pkg + "."):
            base = clean_mod.replace(".", "/")
            rel_path = f"{base}/__init__.py" if pkg_init else f"{base}.py"
        else:
            return None, None
    else:
        if clean_mod == pkg and pkg_init:
            rel_path = f"{layout}/__init__.py"
        elif clean_mod.startswith(pkg + "."):
            tail = clean_mod[len(pkg) + 1:].replace(".", "/")
            rel_path = f"{layout}/{tail}/__init__.py" if pkg_init else f"{layout}/{tail}.py"
        else:
            return None, None
    if isinstance(revision, str) and _HEX_SHA_RE.fullmatch(revision) and type(line) is int and 0 < line < 10**7:
        return rel_path, f"https://github.com/{repo}/blob/{revision}/{rel_path}#L{line}"
    return rel_path, None


def _summary_rows(failure: dict[str, Any], occurrences: int) -> list[tuple[str, str]]:
    component = str(failure.get("component", "mindie"))
    operation = str(failure.get("operation", "operation"))
    version = failure.get("package_version")
    revision = failure.get("package_revision")
    attrs = failure.get("attributes", {})
    repo, tree_url = component_commit_url(component, revision)
    rows: list[tuple[str, str]] = [("Component", f"`{component}`" + (f" (`{repo}`)" if repo else ""))]
    rows.append(("Operation", f"`{operation}`"))
    if version or revision:
        ver_part = f"`{version}`" if version else "unpinned"
        if revision and tree_url:
            rev_part = f"[`{revision[:12]}`]({tree_url})"
        elif revision:
            rev_part = f"`{revision[:12]}`"
        else:
            rev_part = "no commit SHA"
        rows.append(("Build", f"{ver_part} @ {rev_part}"))
    stage = attrs.get("stage")
    substage = attrs.get("substage")
    if stage:
        rows.append(("Stage", f"`{stage}`" + (f" / `{substage}`" if substage else "")))
    category = attrs.get("category")
    if category:
        rows.append(("Category", f"`{category}`"))
    signals = []
    for key in ("error_type", "reason_code", "contract_name", "path", "soc_generation"):
        if attrs.get(key) is not None:
            signals.append(f"`{key}={attrs[key]}`" if key not in {"error_type"} else f"`{attrs[key]}`")
    for key in ("error_code", "exit_code", "errno", "http_status"):
        if attrs.get(key) is not None:
            signals.append(f"`{key}={attrs[key]}`")
    if signals:
        rows.append(("Fault Signals", ", ".join(signals)))
    env_parts = [f"`{attrs[k]}`" for k in ("platform", "arch", "py_version", "harness", "diagnostics_version") if attrs.get(k)]
    if env_parts:
        rows.append(("Runtime Envelope", " / ".join(env_parts)))
    rows.append(("Local Occurrences", str(int(occurrences))))
    return rows


def _frames_section(
    component: str,
    revision: str | None,
    frames: list[dict[str, Any]],
    *,
    truncated: bool = False,
    incomplete: bool = False,
) -> str:
    if not frames:
        return ""
    lines = ["### Owned Call Chain (Outer -> Fault Site)\n"]
    if incomplete:
        lines.append("_(Stack depth exceeded traversal cap; innermost fault site was not reached.)_\n")
    elif truncated or len(frames) > 12:
        lines.append("_(Outer frames omitted; showing innermost frames ending at the fault site.)_\n")
    shown = frames[-12:]
    for idx, frame in enumerate(shown, 1):
        mod = frame.get("module", "unknown")
        fn = frame.get("function", "<unknown>")
        ln = frame.get("line", 0)
        rel_path, url = component_source_permalink(
            component, revision, mod, ln, is_init=(frame.get("init") is True)
        )
        loc_label = f"{rel_path}#L{ln}" if rel_path else f"{mod}#L{ln}"
        link = f"[`{loc_path_escape(loc_label)}`]({url})" if url else f"`{loc_path_escape(loc_label)}`"
        marker = " **(fault site)**" if (idx == len(shown) and not incomplete) else ""
        lines.append(f"{idx}. {link} in `{fn}`{marker}")
    return "\n".join(lines) + "\n\n"


def loc_path_escape(text: str) -> str:
    return text.replace("`", "")


def _phases_section(payload: dict[str, Any], failure: dict[str, Any]) -> str:
    attrs = failure.get("attributes", {})
    phases = attrs.get("phases")
    rows: list[tuple[str, str, str]] = []
    if isinstance(phases, list) and phases:
        for item in phases[-12:]:
            if isinstance(item, dict) and item.get("phase"):
                dur = f"{item['duration_ms']} ms" if "duration_ms" in item else "-"
                rows.append((str(item["phase"]), str(item.get("status", "-")), dur))
    elif len(payload.get("events", [])) > 1:
        for ev in payload["events"][-12:]:
            label = ev.get("phase") or ev.get("operation") or ev.get("event", "-")
            dur = f"{ev['duration_ms']} ms" if "duration_ms" in ev else "-"
            rows.append((f"{ev.get('event', '-')}: {label}", str(ev.get("status", "-")), dur))
    if not rows:
        return ""
    out = [
        "### Preceding Phase Breadcrumbs\n",
        "| Phase | Status | Duration |",
        "| --- | --- | --- |",
    ]
    for phase, status, dur in rows:
        out.append(f"| `{phase}` | `{status}` | `{dur}` |")
    return "\n".join(out) + "\n\n"


def render_triage_card(
    payload: dict[str, Any], *, incident_fp: str, fault_sig: str, occurrences: int
) -> tuple[str, str]:
    """Render deterministic `(title, body)` for a new GitHub Issue."""
    failure = next(e for e in reversed(payload["events"]) if e.get("status") == "error")
    component = str(failure.get("component", "mindie"))
    operation = str(failure.get("operation", "operation"))
    attrs = failure.get("attributes", {})
    verb = "contract violation" if attrs.get("category") == "contract_violation" else "failed"
    detail_tokens = [
        str(x) for x in (
            attrs.get("stage"),
            attrs.get("reason_code") or attrs.get("contract_name") or attrs.get("error_type") or attrs.get("error_code"),
        ) if x is not None
    ]
    suffix = f" ({' / '.join(detail_tokens[:2])})" if detail_tokens else ""
    if suffix and scan_text(suffix):
        suffix = ""
    title = f"[automatic diagnostic] {component}: {operation} {verb}{suffix}"[:200]

    summary_lines = ["| Field | Value |", "| --- | --- |"]
    for field, val in _summary_rows(failure, occurrences):
        summary_lines.append(f"| {field} | {val} |")

    frames = owned_frames(component, attrs.get("stack_frames", []))
    frames_md = _frames_section(
        component,
        failure.get("package_revision"),
        frames,
        truncated=bool(attrs.get("stack_truncated")),
        incomplete=bool(attrs.get("stack_incomplete")),
    )
    phases_md = _phases_section(payload, failure)

    body = (
        f"{MARKER}{incident_fp} -->\n"
        f"{FAULT_MARKER}{fault_sig} -->\n\n"
        "MindIE recorded an automatic diagnostic incident from sanitized, allowlist-projected events.\n\n"
        "### Triage Summary\n\n"
        + "\n".join(summary_lines)
        + "\n\n"
        + frames_md
        + phases_md
        + "<details><summary>Sanitized diagnostic evidence (JSON)</summary>\n\n```json\n"
        + json.dumps(payload, ensure_ascii=True, indent=2)
        + "\n```\n</details>\n"
    )
    if len(body.encode("utf-8")) > 60000:
        raise ValueError("issue body exceeds publication limit")
    if scan_text(title) or scan_text(body):
        raise ValueError("final diagnostic leak scan rejected publication")
    return title, body


def render_occurrence_card(
    payload: dict[str, Any],
    *,
    incident_fp: str,
    fault_sig: str,
    occurrences: int,
    issue_state: str | None = None,
) -> str:
    """Render deterministic comment Markdown for a cross-version repeat or regression."""
    failure = next(e for e in reversed(payload["events"]) if e.get("status") == "error")
    component = str(failure.get("component", "mindie"))
    attrs = failure.get("attributes", {})
    heading = (
        "### Regression Observed After Issue Closure"
        if str(issue_state or "").lower() == "closed"
        else "### Additional Occurrence on Newer Build"
    )
    summary_lines = ["| Field | Value |", "| --- | --- |"]
    for field, val in _summary_rows(failure, occurrences):
        summary_lines.append(f"| {field} | {val} |")

    frames = owned_frames(component, attrs.get("stack_frames", []))
    frames_md = _frames_section(
        component,
        failure.get("package_revision"),
        frames,
        truncated=bool(attrs.get("stack_truncated")),
        incomplete=bool(attrs.get("stack_incomplete")),
    )
    body = (
        f"{MARKER}{incident_fp} -->\n"
        f"{FAULT_MARKER}{fault_sig} -->\n\n"
        f"{heading}\n\n"
        + "\n".join(summary_lines)
        + "\n\n"
        + frames_md
        + "<details><summary>Sanitized diagnostic evidence (JSON)</summary>\n\n```json\n"
        + json.dumps(payload, ensure_ascii=True, indent=2)
        + "\n```\n</details>\n"
    )
    if len(body.encode("utf-8")) > 60000:
        raise ValueError("comment body exceeds publication limit")
    if scan_text(body):
        raise ValueError("final diagnostic leak scan rejected publication")
    return body
