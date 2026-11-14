"""Tier 0/1 regression tests for Diagnostics 2.0 Round 3 edge cases (R3-4, R3-5, R3-7)."""
from __future__ import annotations

import json
import sys
import types
from urllib.parse import parse_qs, urlsplit

from mindie_diagnostics.evidence import exception_evidence
from mindie_diagnostics.fallback import update_failure_policy
from mindie_diagnostics.outbox import Outbox
from mindie_diagnostics.reporter import GitHub, TransportError, fingerprint, publish_one, render_issue
from mindie_diagnostics.reporting import scope_key
from mindie_diagnostics.triage_card import FAULT_MARKER, MARKER, fault_signature


def _sample_event(*, version: str = "0.1.21", revision: str = "2" * 40, line: int = 30) -> dict:
    return {
        "events": [
            {
                "schema": 1,
                "timestamp": "2026-11-13T10:00:00Z",
                "component": "mindie-agent-cc",
                "severity": "ERROR",
                "event": "operation.end",
                "operation": "release.update",
                "status": "error",
                "package_version": version,
                "package_revision": revision,
                "attributes": {
                    "stage": "install",
                    "category": "internal",
                    "error_type": "RuntimeError",
                    "error_code": "update_internal_failure",
                    "stack_frames": [{"module": "mindie_agent_cc.release_update", "function": "run", "line": line}],
                },
            }
        ]
    }


def test_r3_4_handoff_starvation_and_single_page_deletion_overlap(tmp_path, reporting_consent):
    repo = reporting_consent["repository"]
    ev = _sample_event()
    fbody = f"{FAULT_MARKER}{fault_signature(ev)} -->"
    ebody = f"{MARKER}{fingerprint(ev)} -->"

    class SimGitHub(GitHub):
        def __init__(self, issues, comments, *, search=None, limit=8_000_000):
            super().__init__(repo, max_response_bytes=limit)
            self.issues = list(issues)
            self.comments = list(comments)
            self.search = list(search or [])
            self.calls: list[tuple[str, int, int]] = []
            self.posts = 0

        def request(self, method, path, payload=None):
            u = urlsplit(path)
            q = parse_qs(u.query)
            per = int(q.get("per_page", [1])[0])
            page = int(q.get("page", [1])[0])
            tag = "S" if u.path == "search/issues" else ("C" if "/comments" in u.path else ("I" if q else "PIN"))
            self.calls.append((tag, per, page))
            if method == "POST":
                self.posts += 1
                num = 999
                return {"id": num, "number": num, "html_url": f"https://github.com/{repo}/issues/121#issuecomment-{num}", "body": payload["body"]}
            if u.path == "search/issues":
                data = {"items": self.search[(page - 1) * per : page * per], "total_count": len(self.search), "incomplete_results": False}
            elif "/comments" in u.path:
                data = self.comments[(page - 1) * per : page * per]
            elif q:
                data = self.issues[(page - 1) * per : page * per]
            else:
                num = int(u.path.rsplit("/", 1)[-1])
                data = next(r for r in self.issues if r["number"] == num)
            sz = len(json.dumps(data).encode())
            if self._bytes + sz > self.max_response_bytes:
                raise TransportError("github_output_limit")
            self._bytes += sz
            return data

        def begin_cycle(self):
            super().begin_cycle()
            self._bytes = 0

    def _issue(num: int, body: str) -> dict:
        return {"number": num, "html_url": f"https://github.com/{repo}/issues/{num}", "body": body, "state": "open", "created_at": "2026-10-01T00:00:00Z", "updated_at": "2026-10-01T00:00:00Z"}

    def _comment(cid: int, body: str) -> dict:
        return {"id": cid, "html_url": f"https://github.com/{repo}/issues/121#issuecomment-{cid}", "body": body}

    # 1. Handoff starvation: historical page 5 finds issue 121 right before comment page 1 hits byte limit.
    now = [10_000.0]
    q1 = Outbox(tmp_path / "q1.db", clock=lambda: now[0])
    q1.enqueue(scope_key(fingerprint(ev), reporting_consent), "1" * 32, ev, consent=reporting_consent)
    g1 = SimGitHub([_issue(i, "x" * 60000) for i in range(1, 121)] + [_issue(121, fbody)], [_comment(i, "x" * 60000) for i in range(1, 51)], limit=7_800_000)
    r1 = publish_one(q1, g1)
    assert r1["status"] == "retry"
    now[0] += 4000
    g1.calls.clear()
    r2 = publish_one(q1, g1)
    assert r2["status"] == "published"
    assert g1.calls[0][0] == "PIN"
    assert g1.posts == 1

    # 2. Single-page comment interruption + early deletion: comment 26 shifts to 25 and is still matched without duplicate POST.
    q2 = Outbox(tmp_path / "q2.db", clock=lambda: now[0])
    q2.enqueue(scope_key(fingerprint(ev), reporting_consent), "2" * 32, ev, consent=reporting_consent)
    comments = [_comment(i, ebody if i == 26 else "noise") for i in range(1, 51)]
    c_budget = len(json.dumps({"items": [_issue(121, fbody)], "total_count": 1, "incomplete_results": False}).encode()) + len(json.dumps(comments[:25]).encode()) + 20
    g2 = SimGitHub([_issue(121, fbody)], comments, search=[_issue(121, fbody)], limit=c_budget)
    assert publish_one(q2, g2)["status"] == "retry"
    g2.comments.pop(0)
    now[0] += 4000
    c_res = publish_one(q2, g2)
    assert c_res["status"] == "reconciled"
    assert g2.posts == 0

    # 3. Single-page issue interruption + early deletion: issue 31 shifts to 30 and is still matched without duplicate POST.
    q3 = Outbox(tmp_path / "q3.db", clock=lambda: now[0])
    q3.enqueue(scope_key(fingerprint(ev), reporting_consent), "3" * 32, ev, consent=reporting_consent)
    issues = [_issue(i, ebody if i == 31 else "noise") for i in range(1, 51)]
    i_budget = len(json.dumps({"items": [], "total_count": 0, "incomplete_results": False}).encode()) + len(json.dumps(issues[:30]).encode()) + 20
    g3 = SimGitHub(issues, [], limit=i_budget)
    assert publish_one(q3, g3)["status"] == "retry"
    g3.issues.pop(0)
    now[0] += 4000
    i_res = publish_one(q3, g3)
    assert i_res["status"] == "reconciled"
    assert g3.posts == 0


def test_r3_5_update_failure_policy_context_and_over_limit_chain():
    try:
        try:
            raise PermissionError("denied")
        except PermissionError:
            raise RuntimeError("wrapped without from")
    except RuntimeError as ctx_exc:
        assert update_failure_policy(ctx_exc) == ("prerequisite", False, "update_prerequisite_unavailable")

    cur: BaseException = PermissionError("root")
    for idx in range(40):
        nxt = RuntimeError(f"wrap_{idx}")
        nxt.__cause__ = cur
        cur = nxt
    assert update_failure_policy(cur) == ("prerequisite", False, "update_prerequisite_unavailable")


def test_r3_7_deep_300_recursion_reaches_true_fault_and_incomplete_stack_omits_fault_tag(tmp_path):
    deep_file = tmp_path / "mindie_knowledge" / "deep300.py"
    deep_file.parent.mkdir(parents=True, exist_ok=True)
    src = (
        "def true_fault():\n"
        "    raise RuntimeError('leaf')\n"
        "def recurse(n):\n"
        "    return recurse(n - 1) if n > 0 else true_fault()\n"
    )
    ns = {"__name__": "__main__"}
    exec(compile(src, str(deep_file), "exec"), ns)
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 1200))
    try:
        ns["recurse"](300)
    except RuntimeError as exc:
        ev = exception_evidence("mindie-knowledge", exc, source_root=tmp_path / "mindie_knowledge")
    finally:
        sys.setrecursionlimit(old_limit)

    assert len(ev["stack_frames"]) == 24
    assert ev["stack_frames"][-1]["function"] == "true_fault"
    assert ev.get("stack_truncated") is True
    assert not ev.get("stack_incomplete")

    incomplete_bundle = _sample_event()
    incomplete_bundle["events"][0]["attributes"]["stack_incomplete"] = True
    _, body = render_issue({"payload": incomplete_bundle, "occurrences": 1})
    assert "**(fault site)**" not in body
    assert fault_signature(incomplete_bundle) != fault_signature(_sample_event())


def test_round4_f3_intra_cycle_deletion_continuity_issues_and_comments(tmp_path, reporting_consent):
    """F3: Deleting issue 1 or comment 1 between page 1 and page 2 MUST NOT authorize duplicates."""
    from urllib.parse import parse_qs, urlsplit
    from mindie_diagnostics.reporter import GitHub, Outbox, fingerprint, publish_one, render_issue, render_occurrence_comment
    from mindie_diagnostics.reporting import scope_key

    repo = reporting_consent["repository"]
    now = [50_000.0]

    # Case 1: Issues 1..50, exact incident marker on issue 31; delete issue 1 (or 1 & 2) after page 1!
    for delete_count in (1, 2):
        ev = _sample_event(line=100 + delete_count)
        q_issue = Outbox(tmp_path / f"f3_issue_{delete_count}.db", clock=lambda: now[0])
        q_issue.enqueue(scope_key(fingerprint(ev), reporting_consent), f"{delete_count}" * 32, ev, consent=reporting_consent)
        _, exact_body = render_issue({"payload": ev, "occurrences": 1})

        class DeletingIssuesGitHub(GitHub):
            def __init__(self):
                super().__init__(repo)
                self.issues = [
                    {
                        "number": i,
                        "html_url": f"https://github.com/{repo}/issues/{i}",
                        "body": exact_body if i == 31 else f"unrelated {i}",
                        "state": "open",
                        "created_at": f"2026-01-{(i % 28) + 1:02d}T00:00:00Z",
                        "updated_at": f"2026-01-{(i % 28) + 1:02d}T00:00:00Z",
                    }
                    for i in range(1, 51)
                ]
                self.deleted_once = False
                self.created_posts = 0

            def request(self, method, path, payload=None):
                u = urlsplit(path)
                q = parse_qs(u.query)
                if method == "GET":
                    if u.path == "search/issues":
                        return {"items": [], "total_count": 0, "incomplete_results": False}
                    per, pg = int(q["per_page"][0]), int(q["page"][0])
                    chunk = self.issues[(pg - 1) * per : pg * per]
                    if per == 30 and pg == 1 and not self.deleted_once:
                        self.deleted_once = True
                        del self.issues[:delete_count]
                    return chunk
                self.created_posts += 1
                return {"number": 999, "html_url": f"https://github.com/{repo}/issues/999"}

        gh_i = DeletingIssuesGitHub()
        res_i = publish_one(q_issue, gh_i)
        assert res_i["status"] == "reconciled", res_i
        assert res_i["issue_url"] == f"https://github.com/{repo}/issues/31"
        assert gh_i.created_posts == 0

    # Case 2: Comments 1..40, exact incident marker on comment 26; delete comment 1 (or 1 & 2) after page 1!
    for delete_count in (1, 2):
        root_ev = _sample_event(line=200 + delete_count, revision="1" * 40)
        _, root_body = render_issue({"payload": root_ev, "occurrences": 1})
        occ_ev = _sample_event(line=200 + delete_count, revision="2" * 40)
        q_c = Outbox(tmp_path / f"f3_comment_{delete_count}.db", clock=lambda: now[0])
        q_c.enqueue(scope_key(fingerprint(occ_ev), reporting_consent), f"{delete_count + 3}" * 32, occ_ev, consent=reporting_consent)
        exact_comment = render_occurrence_comment({"payload": occ_ev, "occurrences": 1}, issue_state="open")
        root_issue = {
            "number": 77,
            "html_url": f"https://github.com/{repo}/issues/77",
            "body": root_body,
            "state": "open",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }

        class DeletingCommentsGitHub(GitHub):
            def __init__(self):
                super().__init__(repo)
                self.comments = [
                    {
                        "id": idx,
                        "html_url": f"https://github.com/{repo}/issues/77#issuecomment-{idx}",
                        "body": exact_comment if idx == 26 else f"c {idx}",
                    }
                    for idx in range(1, 41)
                ]
                self.deleted_once = False
                self.created_comments = 0

            def request(self, method, path, payload=None):
                u = urlsplit(path)
                q = parse_qs(u.query)
                if method == "GET":
                    if u.path == "search/issues":
                        return {"items": [root_issue], "total_count": 1, "incomplete_results": False}
                    if u.path == f"repos/{repo}/issues/77":
                        return root_issue
                    per, pg = int(q["per_page"][0]), int(q["page"][0])
                    chunk = self.comments[(pg - 1) * per : pg * per]
                    if per == 25 and pg == 1 and not self.deleted_once:
                        self.deleted_once = True
                        del self.comments[:delete_count]
                    return chunk
                self.created_comments += 1
                return {"id": 999, "html_url": f"https://github.com/{repo}/issues/77#issuecomment-999", "body": payload["body"]}

        gh_c = DeletingCommentsGitHub()
        res_c = publish_one(q_c, gh_c)
        assert res_c["status"] == "reconciled", res_c
        assert res_c["comment_url"] == f"https://github.com/{repo}/issues/77#issuecomment-26"
        assert gh_c.created_comments == 0


def test_round4_f6_deep_causal_chain_retains_leaf_and_marks_over_64_incomplete(tmp_path, monkeypatch):
    """F6: 9+ chained wrappers retain true causal leaf; >64 chained links set stack_incomplete=True."""
    from mindie_diagnostics.evidence import exception_evidence
    from mindie_diagnostics.reporter import fault_signature, render_issue

    pkg = tmp_path / "mindie_coordinator"
    pkg.mkdir(parents=True, exist_ok=True)
    mod_file = pkg / "deep_chain.py"
    mod_file.write_text(
        "def leaf_alpha():\n"
        "    raise KeyError('a')\n\n"
        "def leaf_beta():\n"
        "    raise KeyError('b')\n\n"
        "def wrap_chain(fn, depth):\n"
        "    try:\n"
        "        if depth <= 0:\n"
        "            return fn()\n"
        "        return wrap_chain(fn, depth - 1)\n"
        "    except Exception as exc:\n"
        "        raise RuntimeError('wrap') from exc\n",
        encoding="utf-8",
    )
    ns: dict = {"__name__": "mindie_coordinator.deep_chain"}
    exec(compile(mod_file.read_text(encoding="utf-8"), str(mod_file), "exec"), ns)

    exc_a = exc_b = None
    try:
        ns["wrap_chain"](ns["leaf_alpha"], 10)
    except Exception as e:
        exc_a = e
    try:
        ns["wrap_chain"](ns["leaf_beta"], 10)
    except Exception as e:
        exc_b = e

    ev_a = exception_evidence("mindie-coordinator", exc_a, source_root=pkg)
    ev_b = exception_evidence("mindie-coordinator", exc_b, source_root=pkg)
    assert ev_a["stack_frames"][-1]["function"] == "leaf_alpha"
    assert ev_b["stack_frames"][-1]["function"] == "leaf_beta"
    assert ev_a["exception_chain"][-1] == "KeyError"
    assert "stack_incomplete" not in ev_a

    b_a = _sample_event()
    b_a["events"][-1]["component"] = "mindie-coordinator"
    b_a["events"][-1]["attributes"].update(ev_a)
    b_b = _sample_event()
    b_b["events"][-1]["component"] = "mindie-coordinator"
    b_b["events"][-1]["attributes"].update(ev_b)
    assert fault_signature(b_a) != fault_signature(b_b)

    # Beyond 64 causal links, stack_incomplete MUST be True and wrapper MUST NOT be labeled **(fault site)**.
    exc_70 = None
    try:
        ns["wrap_chain"](ns["leaf_alpha"], 70)
    except Exception as e:
        exc_70 = e
    ev_70 = exception_evidence("mindie-coordinator", exc_70, source_root=pkg)
    assert ev_70.get("stack_incomplete") is True
    b_70 = _sample_event()
    b_70["events"][-1]["component"] = "mindie-coordinator"
    b_70["events"][-1]["attributes"].update(ev_70)
    _, body_70 = render_issue({"payload": b_70, "occurrences": 1})
    assert "**(fault site)**" not in body_70
    assert "innermost fault site was not reached" in body_70


def test_round4_invocation_token_stamped_locally_and_stripped_on_export(tmp_path, monkeypatch, reporting_consent):
    """Invocation provenance token stays strictly local and never leaks into bundles or Issues."""
    from mindie_diagnostics.bundle import collect_bundle
    from mindie_diagnostics.ingestion import ingest
    from mindie_diagnostics.integration import record_failure
    from mindie_diagnostics.outbox import Outbox

    diag_root = tmp_path / "diag"
    diag_root.mkdir()
    tok = "a1b2c3d4" * 4
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_ROOT", str(diag_root))
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_CONFIG", reporting_consent["config_file"])
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_INVOCATION_TOKEN", tok)

    res = record_failure("mindie-coordinator", "mindie.hosts", stage="cli_dispatch", category="internal_exception", exception=KeyError("k"))
    assert res["recorded"] is True
    jsonl_files = list((diag_root / "events" / "mindie-coordinator").glob("*.jsonl"))
    assert len(jsonl_files) == 1
    raw_obj = json.loads(jsonl_files[0].read_text(encoding="utf-8").splitlines()[-1])
    assert raw_obj["invocation_token"] == tok

    bundle = collect_bundle(diag_root, operation_id=res["incident_id"])
    assert tok not in json.dumps(bundle)
    assert all(not ev.get("attributes", {}).get("attributes_omitted") for ev in bundle["events"])

    queue = Outbox(tmp_path / "q.db")
    assert ingest(diag_root, queue, config=reporting_consent["config_file"])["enqueued"] == 1
    claimed = queue.claim()
    assert tok not in json.dumps(claimed["payload"])


def test_round5_cross_cycle_deletion_continuity_across_transport_and_budget(tmp_path, reporting_consent):
    """Deleting item 1 before page 2 + interrupting before exhaustion (transport or max_pages) MUST NOT post duplicates."""
    from mindie_diagnostics.reporter import render_occurrence_comment

    repo = reporting_consent["repository"]
    now = [80_000.0]

    for kind in ("issues", "comments"):
        for interrupt in ("none", "transport", "budget"):
            ev = _sample_event(line=300 + (10 if kind == "comments" else 0) + len(interrupt), revision="2" * 40)
            q = Outbox(tmp_path / f"r5_{kind}_{interrupt}.db", clock=lambda: now[0])
            q.enqueue(scope_key(fingerprint(ev), reporting_consent), f"{len(interrupt)}" * 32, ev, consent=reporting_consent)

            if kind == "issues":
                total = 50 if interrupt == "none" else (105 if interrupt == "transport" else 190)
                _, marker_body = render_issue({"payload": ev, "occurrences": 1})
                root_issue = None
            else:
                total = 40 if interrupt == "none" else (90 if interrupt == "transport" else 285)
                root_ev = _sample_event(line=ev["events"][0]["attributes"]["stack_frames"][0]["line"], revision="1" * 40)
                _, root_body = render_issue({"payload": root_ev, "occurrences": 1})
                marker_body = render_occurrence_comment({"payload": ev, "occurrences": 1}, issue_state="open")
                root_issue = {
                    "number": 1,
                    "html_url": f"https://github.com/{repo}/issues/1",
                    "body": root_body,
                    "state": "open",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                }

            class InterruptingDeletingGitHub(GitHub):
                def __init__(self):
                    super().__init__(repo)
                    if kind == "issues":
                        self.rows = [
                            {
                                "number": i,
                                "html_url": f"https://github.com/{repo}/issues/{i}",
                                "body": marker_body if i == 31 else f"i {i}",
                                "state": "open",
                                "created_at": "2026-01-01T00:00:00Z",
                                "updated_at": "2026-01-01T00:00:00Z",
                            }
                            for i in range(1, total + 1)
                        ]
                    else:
                        self.rows = [
                            {
                                "id": i,
                                "html_url": f"https://github.com/{repo}/issues/1#issuecomment-{i}",
                                "body": marker_body if i == 26 else f"c {i}",
                            }
                            for i in range(1, total + 1)
                        ]
                    self.deleted_once = False
                    self.failed_once = False
                    self.posts = 0

                def request(self, method, path, payload=None):
                    u = urlsplit(path)
                    qs = parse_qs(u.query)
                    if method == "POST":
                        self.posts += 1
                        return {"id": 9999, "number": 9999, "html_url": f"https://github.com/{repo}/issues/9999", "body": payload["body"]}
                    if u.path == "search/issues":
                        items = [root_issue] if root_issue else []
                        return {"items": items, "total_count": len(items), "incomplete_results": False}
                    if root_issue and u.path == f"repos/{repo}/issues/1":
                        return root_issue
                    per, pg = int(qs["per_page"][0]), int(qs["page"][0])
                    if interrupt == "transport" and per > 1 and pg == 3 and not self.failed_once:
                        self.failed_once = True
                        raise TransportError("github_http_502")
                    chunk = self.rows[(pg - 1) * per : pg * per]
                    if per in (25, 30) and pg == 1 and not self.deleted_once:
                        self.deleted_once = True
                        self.rows.pop(0)
                    return chunk

            gh = InterruptingDeletingGitHub()
            statuses: list[str] = []
            for _ in range(4):
                res = publish_one(q, gh)
                statuses.append(res["status"])
                if res["status"] != "retry":
                    break
                now[0] += 4000
            assert statuses[-1] == "reconciled", (kind, interrupt, statuses)
            assert gh.posts == 0, (kind, interrupt, statuses)


def test_round5_explicit_build_meta_suppresses_parent_enrichment(tmp_path, monkeypatch):
    """R5-2: Explicit partial child build metadata or infer_build_meta=False never mixes parent revision/version."""
    import mindie_diagnostics.integration as md_int

    root = tmp_path / "diag"
    root.mkdir()
    monkeypatch.setattr(md_int, "_component_meta", lambda _comp: ("9.0", "b" * 40))

    r_partial = md_int.record_failure(
        "mindie-profiler", "cli.execute", stage="cli", category="subprocess",
        version="1.0", revision=None, root=root,
    )
    assert r_partial["recorded"] is True
    ev1 = json.loads(next((root / "events" / "mindie-profiler").glob("*.jsonl")).read_text(encoding="utf-8").splitlines()[-1])
    assert ev1["package_version"] == "1.0"
    assert "package_revision" not in ev1

    r_no_infer = md_int.record_failure(
        "mindie-profiler", "cli.execute", stage="cli", category="subprocess",
        version=None, revision=None, infer_build_meta=False, root=root,
    )
    assert r_no_infer["recorded"] is True
    ev2 = json.loads(next((root / "events" / "mindie-profiler").glob("*.jsonl")).read_text(encoding="utf-8").splitlines()[-1])
    assert "package_version" not in ev2
    assert "package_revision" not in ev2


def test_round7_operation_logging_64_link_chain_and_incomplete_marker(tmp_path):
    """R7-3: Operation logging walks up to 64 causal links and marks >64 links stack_incomplete."""
    import pytest
    from mindie_diagnostics.logging import configure

    ns: dict = {"__name__": "mindie_coordinator.worker"}
    exec(
        "def leaf_one():\n    raise KeyError('one')\n"
        "def leaf_two():\n    raise ValueError('two')\n"
        "def wrap_n(fn, n):\n"
        "    if n <= 0:\n        return fn()\n"
        "    try:\n        return wrap_n(fn, n - 1)\n"
        "    except Exception as err:\n        raise RuntimeError('wrap') from err\n",
        ns,
    )
    rec = configure("mindie-coordinator", root=tmp_path, version="0.7.9")
    try:
        for fn in (ns["leaf_one"], ns["leaf_two"]):
            with pytest.raises(RuntimeError):
                with rec.operation("mindie.run", reportable=True):
                    ns["wrap_n"](fn, 9)
        with pytest.raises(RuntimeError):
            with rec.operation("mindie.run", reportable=True):
                ns["wrap_n"](ns["leaf_one"], 70)
    finally:
        rec.close()

    ev_lines = [
        json.loads(line)
        for p in sorted((tmp_path / "events" / "mindie-coordinator").glob("*.jsonl"))
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ends = [e for e in ev_lines if e.get("event") == "operation.end"]
    assert len(ends) == 3
    a1, a2, a70 = ends[0]["attributes"], ends[1]["attributes"], ends[2]["attributes"]
    assert a1["exception_chain"][-1] == "KeyError"
    assert a2["exception_chain"][-1] == "ValueError"
    assert "stack_incomplete" not in a1 and "stack_incomplete" not in a2
    assert fault_signature({"events": [ends[0]]}) != fault_signature({"events": [ends[1]]})
    assert a70.get("stack_incomplete") is True
    _, body70 = render_issue({"payload": {"events": [ends[2]]}, "occurrences": 1})
    assert "**(fault site)**" not in body70
