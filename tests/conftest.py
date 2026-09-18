import json
import uuid

import pytest


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    # Tests must not use a developer's real contribution or GitHub settings.
    # Setting (rather than deleting) also restores env changes made directly by
    # runner-under-test code, even when the original variable did not exist.
    for key in ('GH_TOKEN', 'GITHUB_TOKEN', 'MINDIE_COMMUNITY_POLICY'):
        monkeypatch.setenv(key, '')
    monkeypatch.setenv('MINDIE_DIAGNOSTICS_ROOT', str(tmp_path.resolve() / 'isolated-diagnostics'))
    from mindie_diagnostics import reporter
    def unexpected_network(*args, **kwargs):
        pytest.fail('unit tests must explicitly mock the HTTPS transport')
    monkeypatch.setattr(reporter, 'build_opener', unexpected_network)


@pytest.fixture
def community_consent(tmp_path, monkeypatch, isolated_runtime):
    """Explicit isolated opt-in for tests that exercise automatic publication."""
    path = tmp_path.resolve() / 'community.json'
    path.write_text(json.dumps({'schema': 'mindie.community.v1', 'workspace_id': uuid.uuid4().hex,
                                'revision': uuid.uuid4().hex, 'decision': 'enabled'}), encoding='utf-8')
    monkeypatch.setenv('MINDIE_COMMUNITY_POLICY', str(path))
    from mindie_diagnostics.community import current_consent
    return current_consent()
