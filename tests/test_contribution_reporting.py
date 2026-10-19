import json
import uuid

import pytest

from mindie_diagnostics import fallback as policy
from mindie_diagnostics.ingestion import ingest
from mindie_diagnostics.integration import configure_reporting, record_failure
from mindie_diagnostics.outbox import Outbox
from mindie_diagnostics.reporting import ConsentUnavailable, require_consent


def authority(path, choice="contribute", revision=None):
    revision = revision or uuid.uuid4().hex
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": "mindie-consent/1", "choice": choice,
        "choice_revision": revision,
    }))
    return revision


def test_two_profiles_ingest_only_their_own_events_and_withdraw_independently(tmp_path, monkeypatch):
    root = tmp_path / "logs"
    profiles = []
    for name in ("codex", "kimi"):
        consent, config = tmp_path / name / "consent.json", tmp_path / name / "diagnostics.json"
        authority(consent)
        result = configure_reporting(True, config=config, consent_config=consent, roots=[str(root)])
        assert result["status"] == "configured"
        monkeypatch.setenv("MINDIE_DIAGNOSTICS_CONFIG", str(config))
        event = record_failure("remote-dev", "probe." + name, stage="tool_call",
                               category="internal_exception", exception=AssertionError(),
                               root=root, version="0.10.0", revision="a" * 40)
        assert event["recorded"]
        profiles.append((consent, config, event, Outbox(policy.state_path(config) / "reporter.sqlite3")))
    for _, config, event, queue in profiles:
        assert ingest(root, queue, config=config)["enqueued"] == 1
        assert queue.rows()[0]["incident_ids"] == [event["incident_id"]]
    authority(profiles[0][0], "disabled")
    assert profiles[0][3].withdraw_unconsented() == 1
    assert profiles[1][3].withdraw_unconsented() == 0
    assert policy.current_consent(profiles[0][1]) is None
    assert policy.current_consent(profiles[1][1]) is not None
    assert policy.service_directory(profiles[0][1]) != policy.service_directory(profiles[1][1])


def test_live_authority_gates_post_without_policy_reconfiguration(tmp_path):
    consent, config = tmp_path / "consent.json", tmp_path / "diagnostics.json"
    revision = authority(consent)
    configure_reporting(True, config=config, consent_config=consent)
    reference = policy.current_consent(config)
    consent.write_text("{")
    with pytest.raises(ConsentUnavailable):
        require_consent(reference)
    authority(consent, revision=revision)
    require_consent(reference)
    authority(consent, "disabled")
    assert policy.consent_status(reference) == "withdrawn"
    authority(consent)
    assert policy.consent_status(reference) == "withdrawn"
    configure_reporting(True, config=config, consent_config=consent)
    assert policy.current_consent(config)["revision"] != reference["revision"]


def test_legacy_choice_does_not_authorize_new_reporting(tmp_path):
    consent, config = tmp_path / "consent.json", tmp_path / "diagnostics.json"
    consent.write_text(json.dumps({"schema": "mindie-consent/1", "choice": "contribute",
                                   "reporting": "disabled"}))
    assert configure_reporting(True, config=config, consent_config=consent)["category"] == "contribution_not_enabled"
    assert policy.current_consent(config) is None
