"""Tier 0 & Tier 1 tests for Diagnostics 2.0 triage cards, permalinks, and comment dedup."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindie_diagnostics import cli, configure
from mindie_diagnostics.bundle import export_issue_event
from mindie_diagnostics.evidence import exception_evidence
from mindie_diagnostics.ingestion import ingest
from mindie_diagnostics.integration import record_failure, reporting_status
from mindie_diagnostics.outbox import Outbox
from mindie_diagnostics.redact import scan_text
from mindie_diagnostics.reporter import GitHub, TransportError, fingerprint, publish_one, render_issue
from mindie_diagnostics.reporting import scope_key
from mindie_diagnostics.triage_card import component_source_permalink, fault_signature


def _cc_event(*, version: str, revision: str, line: int, reason_code: str = "wheel_hash_mismatch"):
    return {
        "events": [
            {
                "schema": 1,
                "timestamp": "2026-11-13T10:00:00Z",
                "monotonic_ns": 100,
                "component": "mindie-agent-cc",
                "severity": "ERROR",
                "event": "operation.end",
                "operation_id": "a" * 32,
                "trace_id": "a" * 32,
                "process_instance_id": "b" * 32,
                "operation": "release.update",
                "status": "error",
                "package_version": version,
                "package_revision": revision,
                "attributes": {
                    "stage": "component_inheritance",
                    "substage": "verify_wheel",
                    "category": "internal",
                    "error_type": "RuntimeError",
                    "reason_code": reason_code,
                    "exit_code": 1,
                    "platform": "darwin",
                    "arch": "arm64",
                    "py_version": "py3.12",
                    "stack_frames": [
                        {
                            "module": "mindie_agent_cc.release_update",
                            "function": "verify_component_wheel",
                            "line": line,
                        }
                    ],
                },
            }
        ]
    }


def test_fault_signature_groups_across_versions_and_line_shifts_while_fingerprint_tracks_build():
    v1 = _cc_event(version="0.1.16", revision="1" * 40, line=84)
    v2 = _cc_event(version="0.1.17", revision="2" * 40, line=92)
    different_bug = _cc_event(version="0.1.17", revision="2" * 40, line=92, reason_code="manifest_schema_invalid")

    assert fingerprint(v1) != fingerprint(v2)
    assert fault_signature(v1) == fault_signature(v2)
    assert fault_signature(v1) != fault_signature(different_bug)


@pytest.mark.parametrize(
    "component,module,expected_rel",
    [
        ("mindie-diagnostics", "mindie_diagnostics.reporter", "mindie_diagnostics/reporter.py"),
        ("mindie-knowledge", "mindie_knowledge.loop.dfx", "mindie_knowledge/loop/dfx.py"),
        ("remote-dev", "remote_dev.observability", "remote_dev/observability.py"),
        ("mindie-coordinator", "mindie_coordinator.cli", "mindie_coordinator/cli.py"),
        ("mindie-profiler", "mindie_profiler.cli", "mindie_profiler/cli.py"),
        ("mindie-sim", "mindie_sim.cli", "mindie_sim/cli.py"),
        ("mindie-debugger", "mindie_debugger.cli", "mindie_debugger/cli.py"),
        ("npu-top", "npu_top.cli", "npu_top/cli.py"),
        ("mindie-agent-codex", "mindie_agent_codex.release_activation", "plugins/mindie-agent/scripts/release_activation.py"),
        ("mindie-agent-agy", "mindie_agent_agy.mcp_server", "plugins/mindie-agent/scripts/mcp_server.py"),
        ("mindie-agent-wb", "mindie_agent_wb.mcp_server", "plugins/mindie-agent/scripts/mcp_server.py"),
        ("mindie-agent-oc", "mindie_agent_oc.mcp_server", "plugins/mindie-agent/scripts/mcp_server.py"),
        ("mindie-agent-cc", "mindie_agent_cc.release_update", "scripts/release_update.py"),
        ("mindie-agent-kimi", "mindie_agent_kimi.updater", "scripts/updater.py"),
        ("mindie-agent-dsh", "mindie_agent_dsh.dsh_diagnostics", "scripts/dsh_diagnostics.py"),
    ],
)
def test_all_ecosystem_components_map_to_exact_github_permalinks_and_retain_frames(component, module, expected_rel):
    sha = "a1b2c3d4" * 5
    rel, url = component_source_permalink(component, sha, module, 128)
    assert rel == expected_rel
    assert url is not None and url.endswith(f"/blob/{sha}/{expected_rel}#L128")

    projected = export_issue_event(
        {
            "schema": 1,
            "timestamp": "2026-11-13T10:00:00Z",
            "component": component,
            "severity": "ERROR",
            "event": "operation.end",
            "operation": "self.check",
            "status": "error",
            "package_revision": sha,
            "attributes": {
                "stage": "run",
                "category": "internal",
                "stack_frames": [{"module": module, "function": "entry", "line": 128}],
            },
        }
    )
    assert projected is not None
    assert len(projected["attributes"]["stack_frames"]) == 1
    assert projected["attributes"]["stack_frames"][0]["module"] == module


def test_cross_version_dedup_appends_comment_and_regression_note_without_duplicates(tmp_path, reporting_consent):
    now = [1000.0]
    queue = Outbox(tmp_path / "queue.db", clock=lambda: now[0])
    repo = reporting_consent["repository"]

    class InMemoryGitHub(GitHub):
        def __init__(self):
            super().__init__(repo)
            self.issues: list[dict] = []
            self.comments: dict[int, list[dict]] = {}
            self.fail_next_comment_uncertain = False
            self.drop_next_comment_before_server = False

        def request(self, method, path, payload=None):
            if method == "GET":
                if path.startswith("search/issues"):
                    return {"items": list(self.issues), "total_count": len(self.issues), "incomplete_results": False}
                if "/comments?" in path:
                    num = int(path.split("/issues/")[1].split("/comments")[0])
                    return list(self.comments.get(num, []))
                return list(self.issues)
            assert method == "POST"
            if path.endswith("/issues"):
                num = 100 + len(self.issues)
                created = {
                    "number": num,
                    "html_url": f"https://github.com/{self.repository}/issues/{num}",
                    "title": payload["title"],
                    "body": payload["body"],
                    "state": "open",
                    "created_at": "2026-11-13T10:00:00Z",
                    "updated_at": "2026-11-13T10:00:00Z",
                }
                self.issues.append(created)
                self.comments[num] = []
                return created
            num = int(path.split("/issues/")[1].split("/comments")[0])
            if self.drop_next_comment_before_server:
                self.drop_next_comment_before_server = False
                raise TransportError("github_timeout", uncertain=True)
            cid = 5000 + len(self.comments.setdefault(num, []))
            comment = {
                "id": cid,
                "html_url": f"https://github.com/{self.repository}/issues/{num}#issuecomment-{cid}",
                "body": payload["body"],
            }
            self.comments[num].append(comment)
            if self.fail_next_comment_uncertain:
                self.fail_next_comment_uncertain = False
                raise TransportError("github_timeout", uncertain=True)
            return comment

    gh = InMemoryGitHub()

    # 1. v0.1.16 opens the root issue with Triage Card v2 and source permalink.
    b1 = _cc_event(version="0.1.16", revision="1" * 40, line=84)
    queue.enqueue(scope_key(fingerprint(b1), reporting_consent), "1" * 32, b1, consent=reporting_consent)
    r1 = publish_one(queue, gh)
    assert r1["status"] == "published"
    assert len(gh.issues) == 1
    assert f"blob/{'1' * 40}/scripts/release_update.py#L84" in gh.issues[0]["body"]
    assert not scan_text(gh.issues[0]["body"])

    # 2. v0.1.17 hits the same unfixed bug -> appends a comment instead of opening a 2nd issue!
    now[0] += 60
    b2 = _cc_event(version="0.1.17", revision="2" * 40, line=92)
    queue.enqueue(scope_key(fingerprint(b2), reporting_consent), "2" * 32, b2, consent=reporting_consent)
    r2 = publish_one(queue, gh)
    assert r2["status"] == "published" and r2.get("publication_kind") == "comment"
    assert len(gh.issues) == 1
    assert len(gh.comments[100]) == 1
    assert "Additional Occurrence on Newer Build" in gh.comments[100][0]["body"]
    assert f"blob/{'2' * 40}/scripts/release_update.py#L92" in gh.comments[100][0]["body"]

    # 3. Another machine on v0.1.17 sees the existing comment and reconciles without posting a 2nd comment!
    other_queue = Outbox(tmp_path / "other.db", clock=lambda: now[0])
    other_queue.enqueue(scope_key(fingerprint(b2), reporting_consent), "3" * 32, b2, consent=reporting_consent)
    r3 = publish_one(other_queue, gh)
    assert r3["status"] == "reconciled"
    assert len(gh.comments[100]) == 1

    # 4. Issue #100 is closed; v0.1.18 regresses and comment POST times out AFTER landing on GitHub:
    #    First cycle marks uncertain; second cycle reconciles the landed comment without duplicating!
    gh.issues[0]["state"] = "closed"
    now[0] += 60
    b3 = _cc_event(version="0.1.18", revision="3" * 40, line=95)
    queue.enqueue(scope_key(fingerprint(b3), reporting_consent), "4" * 32, b3, consent=reporting_consent)
    gh.fail_next_comment_uncertain = True
    r4_first = publish_one(queue, gh)
    assert r4_first["status"] == "uncertain"
    assert len(gh.comments[100]) == 2
    assert "Regression Observed After Issue Closure" in gh.comments[100][1]["body"]
    now[0] += 4000
    r4_reconciled = publish_one(queue, gh)
    assert r4_reconciled["status"] == "reconciled"
    assert len(gh.comments[100]) == 2

    # 5. v0.1.19 comment POST times out BEFORE landing on GitHub: uncertain state never blindly replays!
    now[0] += 60
    b4 = _cc_event(version="0.1.19", revision="4" * 40, line=99)
    queue.enqueue(scope_key(fingerprint(b4), reporting_consent), "5" * 32, b4, consent=reporting_consent)
    gh.drop_next_comment_before_server = True
    assert publish_one(queue, gh)["status"] == "uncertain"
    now[0] += 4000
    assert publish_one(queue, gh)["status"] == "uncertain"
    assert len(gh.comments[100]) == 2


def test_operation_phase_breadcrumbs_contract_violation_and_local_fast_path(tmp_path, monkeypatch, reporting_consent, capsys):
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_ROOT", str(tmp_path))
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    script_file = script_dir / "calibrate_check.py"

    rec = configure("mindie-sim", root=tmp_path, version="0.9.4")
    code = compile(
        "def verify_regime(record_failure, root, src):\n"
        "    return record_failure(\n"
        "        'mindie-sim', 'calibrate.verify', stage='regime_audit', substage='decode_bound',\n"
        "        category='contract_violation', contract_name='wave5_decode_envelope',\n"
        "        reason_code='residual_exceeded_ceiling', soc_generation='a3',\n"
        "        version='0.9.4', revision='b' * 40, root=root, source_root=src,\n"
        "    )\n",
        str(script_file),
        "exec",
    )
    ns: dict = {}
    exec(code, ns)

    with rec.operation("calibrate.verify") as op:
        with op.phase("load_catalog"):
            pass
        with op.phase("regime_audit"):
            ref = ns["verify_regime"](record_failure, tmp_path, script_dir)
    rec.close()

    assert ref["recorded"] is True
    status = reporting_status(config=reporting_consent["config_file"])
    recent = status["local"]["recent"][0]
    assert recent["incident_id"] == ref["incident_id"]
    assert recent["category"] == "contract_violation"
    assert recent["reason_code"] == "residual_exceeded_ceiling"
    assert recent["contract_name"] == "wave5_decode_envelope"
    assert recent["substage"] == "decode_bound"
    assert recent["stack_frames"][-1]["function"] == "verify_regime"
    assert "py_version" in recent

    # CLI bundle works with omitted --root and --incident-id alias.
    assert cli.main(["bundle", "--incident-id", ref["incident_id"]]) == 0
    cli_bundle = json.loads(capsys.readouterr().out)
    assert cli_bundle["summary"]["operation_ids"] == [ref["incident_id"]]

    # Ingestion + Triage Card v2 renders phase breadcrumbs and contract violation title.
    queue = Outbox(tmp_path / "queue.db")
    assert ingest(tmp_path, queue, config=reporting_consent["config_file"])["enqueued"] == 1
    item = queue.claim()
    title, body = render_issue(item)
    assert "contract violation" in title
    assert "load_catalog" in body and "regime_audit" in body
    assert "wave5_decode_envelope" in body
    assert not scan_text(title) and not scan_text(body)


def test_typed_updater_reason_code_and_child_evidence_transport(tmp_path, reporting_consent):
    class WheelHashError(RuntimeError):
        reason_code = "wheel_hash_mismatch"

    script_file = tmp_path / "updater.py"
    code = compile("def verify_wheel():\n    raise WheelHashError('PRIVATE_PATH /Users/alice/wheel.whl')\nverify_wheel()\n",
                   str(script_file), "exec")
    try:
        exec(code, {"__name__": "__main__", "WheelHashError": WheelHashError})
    except WheelHashError as exc:
        transported = exception_evidence("mindie-agent-kimi", exc, source_root=tmp_path)

    assert transported["reason_code"] == "wheel_hash_mismatch"
    assert transported["stack_frames"][-1]["module"] == "mindie_agent_kimi.updater"

    wrapper = RuntimeError("child failed")
    wrapper.reason_code = transported["reason_code"]
    ref = record_failure(
        "mindie-agent-kimi",
        "release.update",
        stage="verify_wheel",
        category="update",
        version="0.1.17",
        revision="c" * 40,
        exception=wrapper,
        evidence=transported,
        exit_code=1,
        root=tmp_path,
    )
    assert ref["recorded"]
    queue = Outbox(tmp_path / "queue.db")
    assert ingest(tmp_path, queue, config=reporting_consent["config_file"])["enqueued"] == 1
    claimed = queue.claim()
    title, body = render_issue(claimed)
    assert "wheel_hash_mismatch" in title
    assert f"blob/{'c' * 40}/scripts/updater.py#L2" in body
    assert "alice" not in body and "PRIVATE_PATH" not in body


def test_p1_1_distinct_v2_reasons_do_not_collapse_and_legacy_receipts_stay_compatible(tmp_path, reporting_consent):
    from mindie_diagnostics.triage_card import legacy_fingerprint

    queue = Outbox(tmp_path / "queue.db")
    e1 = _cc_event(version="0.1.20", revision="1" * 40, line=84, reason_code="wheel_hash_mismatch")
    e2 = _cc_event(version="0.1.20", revision="1" * 40, line=84, reason_code="manifest_schema_invalid")
    assert fingerprint(e1) != fingerprint(e2)

    assert queue.enqueue(scope_key(fingerprint(e1), reporting_consent), "1" * 32, e1, consent=reporting_consent)
    assert queue.enqueue(scope_key(fingerprint(e2), reporting_consent), "2" * 32, e2, consent=reporting_consent)
    assert len(queue.rows()) == 2

    # Legacy 0.5.10 row with error_code (no v2 reason/contract/substage) retains legacy marker & dedups.
    legacy_bundle = {
        "events": [
            {
                "schema": 1,
                "timestamp": "2026-11-13T10:00:00Z",
                "monotonic_ns": 100,
                "component": "mindie-agent-cc",
                "severity": "ERROR",
                "event": "operation.end",
                "operation_id": "3" * 32,
                "trace_id": "3" * 32,
                "process_instance_id": "4" * 32,
                "operation": "release.update",
                "status": "error",
                "package_version": "0.1.19",
                "package_revision": "9" * 40,
                "attributes": {
                    "stage": "install",
                    "category": "internal",
                    "error_code": "update_internal_failure",
                    "error_type": "RuntimeError",
                    "stack_frames": [{"module": "mindie_agent_cc.release_update", "function": "run", "line": 12}],
                },
            }
        ]
    }
    legacy_fp = legacy_fingerprint(legacy_bundle)
    assert legacy_fp != fingerprint(legacy_bundle)
    legacy_q = Outbox(tmp_path / "legacy.db")
    assert legacy_q.enqueue(scope_key(legacy_fp, reporting_consent), "3" * 32, legacy_bundle, consent=reporting_consent)
    # Re-enqueueing under new fingerprint deduplicates onto existing legacy row.
    assert legacy_q.enqueue(scope_key(fingerprint(legacy_bundle), reporting_consent), "5" * 32, legacy_bundle, consent=reporting_consent)
    assert len(legacy_q.rows()) == 1
    assert legacy_q.rows()[0]["occurrences"] == 2
    claimed = legacy_q.claim()
    _, legacy_body = render_issue(claimed)
    assert f"<!-- mindie-incident:{legacy_fp} -->" in legacy_body


def test_p1_2_paginated_historical_search_and_incomplete_search_guard(tmp_path, reporting_consent):
    from urllib.parse import parse_qs, urlsplit

    now = [100_000.0]
    queue = Outbox(tmp_path / "queue.db", clock=lambda: now[0])
    repo = reporting_consent["repository"]

    old_bundle = _cc_event(version="0.1.10", revision="a" * 40, line=40)
    _, old_body = render_issue({"payload": old_bundle, "occurrences": 1})
    old_closed_issue = {
        "number": 12,
        "html_url": f"https://github.com/{repo}/issues/12",
        "body": old_body,
        "state": "closed",
        "created_at": "2026-06-01T00:00:00Z",
        "updated_at": "2026-06-02T00:00:00Z",
    }
    noise = [
        {
            "number": 200 + i,
            "html_url": f"https://github.com/{repo}/issues/{200 + i}",
            "body": "unrelated discussion",
            "state": "open",
            "created_at": "2026-06-01T00:00:00Z",
            "updated_at": "2026-06-01T00:00:00Z",
        }
        for i in range(35)
    ]

    class PaginatedSearchGitHub(GitHub):
        def __init__(self):
            super().__init__(repo)
            self.search_items = [*noise, old_closed_issue]
            self.incomplete = False
            self.comments = {12: []}
            self.created_issues = []

        def request(self, method, path, payload=None):
            if method == "GET":
                parsed = urlsplit(path)
                q = parse_qs(parsed.query)
                if parsed.path == "search/issues":
                    per = int(q["per_page"][0])
                    page = int(q["page"][0])
                    chunk = self.search_items[(page - 1) * per : page * per]
                    return {"items": chunk, "total_count": len(self.search_items), "incomplete_results": self.incomplete}
                if "/comments" in parsed.path:
                    return list(self.comments[12])
                return []
            assert method == "POST"
            if path.endswith("/issues"):
                self.created_issues.append(payload)
                return {"number": 999, "html_url": f"https://github.com/{repo}/issues/999"}
            cid = 8001
            c = {"id": cid, "html_url": f"https://github.com/{repo}/issues/12#issuecomment-{cid}", "body": payload["body"]}
            self.comments[12].append(c)
            return c

    gh = PaginatedSearchGitHub()
    new_bundle = _cc_event(version="0.1.20", revision="b" * 40, line=55)
    queue.enqueue(scope_key(fingerprint(new_bundle), reporting_consent), "6" * 32, new_bundle, consent=reporting_consent)
    res = publish_one(queue, gh)
    assert res["status"] == "published"
    assert res["publication_kind"] == "comment"
    assert res["comment_url"] == f"https://github.com/{repo}/issues/12#issuecomment-8001"
    assert not gh.created_issues
    assert len(gh.comments[12]) == 1

    # Incomplete search + empty since window must retry rather than authorize a duplicate Issue.
    gh2 = PaginatedSearchGitHub()
    gh2.search_items = []
    gh2.incomplete = True
    q2 = Outbox(tmp_path / "q2.db", clock=lambda: now[0])
    fresh = _cc_event(version="0.1.20", revision="c" * 40, line=60, reason_code="manifest_missing")
    q2.enqueue(scope_key(fingerprint(fresh), reporting_consent), "7" * 32, fresh, consent=reporting_consent)
    res2 = publish_one(q2, gh2)
    assert res2["status"] == "retry"
    assert res2["error"] == "github_reconciliation_window_exhausted"
    assert not gh2.created_issues


def test_p1_3_and_p2_3_comment_pagination_beyond_500_comments_and_receipt_url(tmp_path, reporting_consent):
    from urllib.parse import parse_qs, urlsplit

    now = [10_000.0]
    queue = Outbox(tmp_path / "queue.db", clock=lambda: now[0])
    repo = reporting_consent["repository"]

    root_bundle = _cc_event(version="0.1.10", revision="1" * 40, line=20)
    _, root_body = render_issue({"payload": root_bundle, "occurrences": 1})
    root_issue = {
        "number": 42,
        "html_url": f"https://github.com/{repo}/issues/42",
        "body": root_body,
        "state": "open",
        "created_at": "2026-10-01T00:00:00Z",
        "updated_at": "2026-10-01T00:00:00Z",
    }
    initial_comments = [
        {"id": idx, "html_url": f"https://github.com/{repo}/issues/42#issuecomment-{idx}", "body": f"comment {idx}"}
        for idx in range(1, 621)
    ]

    class ManyCommentsGitHub(GitHub):
        def __init__(self):
            super().__init__(repo)
            self.comments = list(initial_comments)
            self.fail_post_after_landing = True
            self.comment_page_calls: list[int] = []

        def request(self, method, path, payload=None):
            parsed = urlsplit(path)
            q = parse_qs(parsed.query)
            if method == "GET":
                if parsed.path == "search/issues":
                    return {"items": [root_issue], "total_count": 1, "incomplete_results": False}
                if parsed.path == f"repos/{repo}/issues/42":
                    return root_issue
                if parsed.path == f"repos/{repo}/issues/42/comments":
                    per = int(q["per_page"][0])
                    page = int(q["page"][0])
                    self.comment_page_calls.append(page)
                    return self.comments[(page - 1) * per : page * per]
                return []
            assert method == "POST"
            cid = self.comments[-1]["id"] + 1
            created = {
                "id": cid,
                "html_url": f"https://github.com/{repo}/issues/42#issuecomment-{cid}",
                "body": payload["body"],
            }
            self.comments.append(created)
            if self.fail_post_after_landing:
                self.fail_post_after_landing = False
                raise TransportError("github_timeout", uncertain=True)
            return created

    gh = ManyCommentsGitHub()
    b_new = _cc_event(version="0.1.21", revision="2" * 40, line=30)
    queue.enqueue(scope_key(fingerprint(b_new), reporting_consent), "8" * 32, b_new, consent=reporting_consent)

    # Cycle 1 scans pages 1..10 (250 comments), persists bounded cursor, returns retry.
    c1 = publish_one(queue, gh)
    assert c1["status"] == "retry"
    assert gh.comment_page_calls == list(range(1, 11))

    # Cycle 2 resumes from page 10 overlap -> 19 (reaches comment 475), persists cursor, returns retry.
    now[0] += 4000
    gh.comment_page_calls.clear()
    c2 = publish_one(queue, gh)
    assert c2["status"] == "retry"
    assert gh.comment_page_calls == list(range(10, 20))

    # Cycle 3 resumes from page 19 overlap -> 25 (reaches comment 620), posts #621, times out uncertain!
    now[0] += 4000
    gh.comment_page_calls.clear()
    c3 = publish_one(queue, gh)
    assert c3["status"] == "uncertain"
    assert gh.comment_page_calls == list(range(19, 26)) + [25 * k for k in range(1, 25)]
    assert len(gh.comments) == 621

    # Cycle 4 reconciles uncertain comment #621 directly on page 25 in a SINGLE GET and preserves comment_url!
    now[0] += 4000
    gh.comment_page_calls.clear()
    c4 = publish_one(queue, gh)
    assert c4["status"] == "reconciled"
    assert c4["publication_kind"] == "comment"
    assert c4["comment_url"] == f"https://github.com/{repo}/issues/42#issuecomment-621"
    assert gh.comment_page_calls == [25]
    row = queue.rows()[0]
    assert row["state"] == "published"
    assert row["issue_url"] == f"https://github.com/{repo}/issues/42"
    assert row["comment_url"] == f"https://github.com/{repo}/issues/42#issuecomment-621"
    assert row["publication_kind"] == "comment"


def test_p1_8_and_p2_1_canonical_package_source_root_and_fault_site_signature(tmp_path):
    pkg_dir = tmp_path / "mindie_knowledge"
    pkg_dir.mkdir()
    mod_file = pkg_dir / "owned_state.py"
    code = compile("def check_state():\n    raise RuntimeError('boom')\ncheck_state()\n", str(mod_file), "exec")
    try:
        exec(code, {"__name__": "__main__"})
    except RuntimeError as exc:
        ev_from_repo_root = exception_evidence("mindie-knowledge", exc, source_root=tmp_path)
        ev_from_pkg_root = exception_evidence("mindie-knowledge", exc, source_root=pkg_dir)

    assert ev_from_repo_root["stack_frames"][-1]["module"] == "mindie_knowledge.owned_state"
    assert ev_from_pkg_root["stack_frames"][-1]["module"] == "mindie_knowledge.owned_state"
    assert ev_from_pkg_root["stack_frames"][0]["function"] == "<module>"
    rel, url = component_source_permalink("mindie-knowledge", "d" * 40, ev_from_repo_root["stack_frames"][-1]["module"], 2)
    assert rel == "mindie_knowledge/owned_state.py"
    assert url == f"https://github.com/mindie-agent/knowledge/blob/{'d' * 40}/mindie_knowledge/owned_state.py#L2"

    # P2-1: Outer wrapper rename does not alter root-cause fault_signature when fault site is identical.
    b_outer_a = _cc_event(version="0.1.20", revision="1" * 40, line=84)
    b_outer_a["events"][0]["attributes"]["stack_frames"].insert(
        0, {"module": "mindie_agent_cc.cli", "function": "old_wrapper", "line": 10}
    )
    b_outer_b = _cc_event(version="0.1.21", revision="2" * 40, line=90)
    b_outer_b["events"][0]["attributes"]["stack_frames"].insert(
        0, {"module": "mindie_agent_cc.cli", "function": "renamed_wrapper", "line": 14}
    )
    assert fault_signature(b_outer_a) == fault_signature(b_outer_b)


def test_r2_p1_1_local_operation_logger_excludes_exception_message_free_text(tmp_path):
    rec = configure("npu-top", root=tmp_path, version="0.3.3")
    secret = "PrivateCustomerName confidential billing detail"
    with rec.operation("top.cli") as op:
        op.fail(stage="cli", category="internal", exception=RuntimeError(secret))
    rec.close()
    raw_files = list(tmp_path.rglob("*.jsonl"))
    assert raw_files
    joined = "\n".join(p.read_text(encoding="utf-8") for p in raw_files)
    assert secret not in joined
    assert "PrivateCustomerName" not in joined
    assert "exception_message" not in joined


def test_r2_p1_2_legacy_fingerprint_does_not_collapse_distinct_error_codes(tmp_path, reporting_consent):
    from mindie_diagnostics.triage_card import legacy_fingerprint

    def _legacy_err(code: str, op_id: str):
        return {
            "events": [
                {
                    "schema": 1,
                    "timestamp": "2026-11-13T10:00:00Z",
                    "monotonic_ns": 100,
                    "component": "mindie-agent-cc",
                    "severity": "ERROR",
                    "event": "operation.end",
                    "operation_id": op_id,
                    "trace_id": op_id,
                    "process_instance_id": "4" * 32,
                    "operation": "release.update",
                    "status": "error",
                    "package_version": "0.1.19",
                    "package_revision": "9" * 40,
                    "attributes": {
                        "stage": "install",
                        "category": "internal",
                        "error_code": code,
                        "error_type": "RuntimeError",
                        "stack_frames": [{"module": "mindie_agent_cc.release_update", "function": "run", "line": 12}],
                    },
                }
            ]
        }

    now = [1000.0]
    queue = Outbox(tmp_path / "r2p12.db", clock=lambda: now[0])
    e1 = _legacy_err("update_internal_failure", "1" * 32)
    e2 = _legacy_err("runtime_dependency_missing", "2" * 32)
    assert legacy_fingerprint(e1) == legacy_fingerprint(e2)
    assert fingerprint(e1) != fingerprint(e2)

    # Enqueue and publish e1 under its legacy key, even across compaction.
    assert queue.enqueue(scope_key(legacy_fingerprint(e1), reporting_consent), "1" * 32, e1, consent=reporting_consent)
    item1 = queue.claim()
    queue.update(item1, state="published", issue_number=10, issue_url=f"https://github.com/{reporting_consent['repository']}/issues/10")
    now[0] += 8 * 86400
    queue.maintain()

    # Enqueue e2 under its current key: must NOT alias onto published e1!
    assert queue.enqueue(scope_key(fingerprint(e2), reporting_consent), "2" * 32, e2, consent=reporting_consent)
    claimed2 = queue.claim()
    assert claimed2 is not None
    assert claimed2["payload"]["events"][0]["attributes"]["error_code"] == "runtime_dependency_missing"


def test_r2_p1_4_comment_pagination_persists_cursor_on_byte_limit_and_timeout(tmp_path, reporting_consent):
    from urllib.parse import parse_qs, urlsplit

    now = [20_000.0]
    queue = Outbox(tmp_path / "r2p14.db", clock=lambda: now[0])
    repo = reporting_consent["repository"]
    root_bundle = _cc_event(version="0.1.10", revision="1" * 40, line=20)
    _, root_body = render_issue({"payload": root_bundle, "occurrences": 1})
    root_issue = {
        "number": 55,
        "html_url": f"https://github.com/{repo}/issues/55",
        "body": root_body,
        "state": "open",
        "created_at": "2026-10-01T00:00:00Z",
        "updated_at": "2026-10-01T00:00:00Z",
    }
    big_comments = [
        {"id": idx, "html_url": f"https://github.com/{repo}/issues/55#issuecomment-{idx}", "body": "c" * 60000}
        for idx in range(1, 101)
    ]

    class ByteBoundedGitHub(GitHub):
        def __init__(self):
            super().__init__(repo)
            self.comments = list(big_comments)
            self.calls: list[int] = []
            self.used_in_cycle = 0

        def begin_cycle(self):
            super().begin_cycle()
            self.used_in_cycle = 0

        def request(self, method, path, payload=None):
            parsed = urlsplit(path)
            q = parse_qs(parsed.query)
            if method == "GET":
                if parsed.path == "search/issues":
                    return {"items": [root_issue], "total_count": 1, "incomplete_results": False}
                if parsed.path == f"repos/{repo}/issues/55":
                    return root_issue
                if parsed.path == f"repos/{repo}/issues/55/comments":
                    per = int(q["per_page"][0])
                    page = int(q["page"][0])
                    self.calls.append(page)
                    chunk = self.comments[(page - 1) * per : page * per]
                    sz = sum(len(r["body"]) for r in chunk)
                    if self.used_in_cycle + sz > 3_500_000:
                        raise TransportError("github_output_limit")
                    self.used_in_cycle += sz
                    return chunk
                return []
            assert method == "POST"
            cid = 9001
            created = {"id": cid, "html_url": f"https://github.com/{repo}/issues/55#issuecomment-{cid}", "body": payload["body"]}
            self.comments.append(created)
            return created

    gh = ByteBoundedGitHub()
    b_new = _cc_event(version="0.1.21", revision="2" * 40, line=30)
    queue.enqueue(scope_key(fingerprint(b_new), reporting_consent), "9" * 32, b_new, consent=reporting_consent)

    # Each 25-comment page is ~1.5 MB; 2 pages fit per 3.5 MB budget, page 3 hits github_output_limit.
    r1 = publish_one(queue, gh)
    assert r1["status"] == "retry"
    assert gh.calls == [1, 2, 3]

    now[0] += 4000
    gh.calls.clear()
    r2 = publish_one(queue, gh)
    assert r2["status"] == "retry"
    assert gh.calls == [2, 3, 4]

    now[0] += 4000
    gh.calls.clear()
    r3 = publish_one(queue, gh)
    assert r3["status"] == "published"
    assert r3["publication_kind"] == "comment"
    assert gh.calls == [3, 4, 5, 25, 50, 75, 100]


def test_r2_p2_3_and_r2_p2_4_nested_package_init_permalinks_and_deep_trace_tail_retention(tmp_path):
    pkg = tmp_path / "mindie_knowledge" / "loop"
    pkg.mkdir(parents=True)
    init_file = pkg / "__init__.py"
    code = compile("def boot_loop():\n    raise RuntimeError('init fail')\nboot_loop()\n", str(init_file), "exec")
    try:
        exec(code, {"__name__": "__main__"})
    except RuntimeError as exc:
        ev = exception_evidence("mindie-knowledge", exc, source_root=tmp_path / "mindie_knowledge")

    inner = ev["stack_frames"][-1]
    assert inner["module"] == "mindie_knowledge.loop"
    assert inner.get("init") is True
    rel, url = component_source_permalink(
        "mindie-knowledge", "e" * 40, inner["module"], inner["line"], is_init=bool(inner.get("init"))
    )
    assert rel == "mindie_knowledge/loop/__init__.py"
    assert url == f"https://github.com/mindie-agent/knowledge/blob/{'e' * 40}/mindie_knowledge/loop/__init__.py#L2"

    # 30 recursive frames ending in true_fault() must retain true_fault() at the tail and set stack_truncated=True.
    deep_file = tmp_path / "mindie_knowledge" / "deep.py"
    deep_src = (
        "def true_fault():\n"
        "    raise RuntimeError('leaf')\n"
        "def recurse(n):\n"
        "    if n <= 0:\n"
        "        return true_fault()\n"
        "    return recurse(n - 1)\n"
        "recurse(30)\n"
    )
    try:
        exec(compile(deep_src, str(deep_file), "exec"), {"__name__": "__main__"})
    except RuntimeError as exc:
        deep_ev = exception_evidence("mindie-knowledge", exc, source_root=tmp_path / "mindie_knowledge")

    assert len(deep_ev["stack_frames"]) == 24
    assert deep_ev["stack_frames"][-1]["function"] == "true_fault"
    assert deep_ev.get("stack_truncated") is True
