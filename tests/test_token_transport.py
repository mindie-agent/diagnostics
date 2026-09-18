import io
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from mindie_diagnostics.community import ConsentWithdrawn, guard_consent
from mindie_diagnostics.reporter import GitHub, TransportError, _NoRedirect


@pytest.mark.parametrize('variable', ['GH_TOKEN', 'GITHUB_TOKEN'])
def test_token_only_auth_uses_fixed_https_host_without_subprocess(monkeypatch, variable):
    from mindie_diagnostics import reporter
    monkeypatch.delenv('GH_TOKEN', raising=False)
    monkeypatch.delenv('GITHUB_TOKEN', raising=False)
    monkeypatch.setenv(variable, 'test-private-placeholder')
    monkeypatch.setattr(reporter.shutil, 'which', lambda _: None)
    calls = []

    def open_request(request, **kwargs):
        calls.append(request)
        assert request.full_url == 'https://api.github.com/repos/example/project/issues'
        assert request.get_header('Authorization') == 'Bearer test-private-placeholder'
        assert kwargs['timeout'] == 30
        return io.BytesIO(b'{"number":1,"html_url":"https://github.com/example/project/issues/1"}')

    monkeypatch.setattr(reporter, 'build_opener', lambda handler: SimpleNamespace(open=open_request))
    monkeypatch.setattr(reporter.subprocess, 'run', lambda *a, **k: pytest.fail('gh subprocess unexpectedly used'))
    assert GitHub('example/project').create_issue('title', 'body')['number'] == 1
    assert len(calls) == 1


@pytest.mark.parametrize('method,code,uncertain,delay', [('POST', 403, False, 3600),
    ('POST', 429, False, 3600), ('POST', 502, True, 60), ('GET', 502, False, 60), ('POST', None, True, 60)])
def test_token_failures_preserve_submission_state_without_credentials(monkeypatch, method, code, uncertain, delay):
    from mindie_diagnostics import reporter
    monkeypatch.setenv('GH_TOKEN', 'test-private-placeholder')
    monkeypatch.setattr(reporter.shutil, 'which', lambda _: None)

    def open_request(*args, **kwargs):
        if code:
            raise HTTPError('https://api.github.com', code, 'private response secret', {}, None)
        raise URLError('private response secret')

    monkeypatch.setattr(reporter, 'build_opener', lambda handler: SimpleNamespace(open=open_request))
    with pytest.raises(TransportError) as error:
        GitHub().request(method, 'repos/example/project/issues')
    assert error.value.uncertain == uncertain
    assert error.value.retry_after == delay
    assert 'secret' not in str(error.value) and 'placeholder' not in str(error.value)
    assert error.value.__suppress_context__


def test_no_token_transport_can_bypass_revoked_policy(monkeypatch, community_consent):
    from mindie_diagnostics import reporter
    monkeypatch.setenv('GH_TOKEN', 'test-private-placeholder')
    monkeypatch.setattr(reporter, 'build_opener', lambda *a: pytest.fail('revoked request opened'))
    from pathlib import Path
    Path(community_consent['policy_file']).unlink()
    with pytest.raises(ConsentWithdrawn), guard_consent(community_consent):
        GitHub().create_issue('title', 'body')


def test_token_transport_never_follows_redirects():
    assert _NoRedirect().redirect_request(None, None, 302, '', {}, 'https://elsewhere.invalid') is None
