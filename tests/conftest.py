import pytest


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    # Tests must not use a developer's real contribution or GitHub settings.
    # Setting (rather than deleting) also restores env changes made directly by
    # runner-under-test code, even when the original variable did not exist.
    for key in ('GH_TOKEN', 'GITHUB_TOKEN', 'MINDIE_COMMUNITY_POLICY'):
        monkeypatch.setenv(key, '')
    home = tmp_path.resolve() / 'home'
    for key, value in (
        ('HOME', home),
        ('USERPROFILE', home),
        ('XDG_CONFIG_HOME', tmp_path.resolve() / 'xdg-config'),
        ('XDG_DATA_HOME', tmp_path.resolve() / 'xdg-data'),
        ('XDG_STATE_HOME', tmp_path.resolve() / 'xdg-state'),
        ('LOCALAPPDATA', tmp_path.resolve() / 'localappdata'),
        ('MINDIE_DIAGNOSTICS_SERVICE_DIR', tmp_path.resolve() / 'service'),
    ):
        monkeypatch.setenv(key, str(value))
    monkeypatch.setenv('MINDIE_DIAGNOSTICS_ROOT', str(tmp_path.resolve() / 'isolated-diagnostics'))
    monkeypatch.setenv('MINDIE_DIAGNOSTICS_CONFIG', str(tmp_path.resolve() / 'reporting.json'))
    from mindie_diagnostics import reporter
    def unexpected_network(*args, **kwargs):
        pytest.fail('unit tests must explicitly mock the HTTPS transport')
    monkeypatch.setattr(reporter.GitHub, 'request', unexpected_network)


@pytest.fixture
def reporting_consent(tmp_path, monkeypatch, isolated_runtime):
    """Independent explicit public fault-reporting consent, no community scope."""
    from mindie_diagnostics.integration import configure_reporting
    from mindie_diagnostics.reporting import current_consent
    configure_reporting(True, roots=[str(tmp_path.resolve())])
    return current_consent()
