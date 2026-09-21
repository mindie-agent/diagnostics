"""Single gh transport: no token HTTP fallback and no raw stderr classification."""
import sys
import threading
import time

import pytest

from mindie_diagnostics import reporter
from mindie_diagnostics.reporter import GitHub, TransportError, _parse_response, _run_gh

# Retain only the real method before the suite's default network guard replaces it.
_REQUEST = GitHub.request


@pytest.mark.parametrize('status,headers,method,permanent,uncertain,delay', [
    (401, '', 'POST', True, False, 60), (403, '', 'POST', True, False, 60),
    (403, 'X-RateLimit-Remaining: 0\nRetry-After: 42\n', 'POST', False, False, 42),
    (429, 'Retry-After: 999999\n', 'POST', False, False, 3600),
    (503, '', 'POST', False, True, 60), (503, '', 'GET', False, False, 60),
])
def test_reliable_header_classification(status, headers, method, permanent, uncertain, delay):
    with pytest.raises(TransportError) as caught:
        _parse_response(f'HTTP/2 {status} Response\n{headers}\n{{}}'.encode(), 1, method)
    error = caught.value
    assert (error.permanent, error.uncertain, error.retry_after) == (permanent, uncertain, delay)


def test_final_header_block_and_malformed_post():
    assert _parse_response(b'HTTP/1.1 100 Continue\r\n\r\nHTTP/2.0 200 OK\r\n\r\n[]', 0, 'GET') == []
    with pytest.raises(TransportError) as caught:
        _parse_response(b'HTTP/2 201 Created\n\nnot_json', 0, 'POST')
    assert caught.value.uncertain


@pytest.mark.parametrize('variable', ['GH_TOKEN', 'GITHUB_TOKEN'])
def test_missing_gh_does_not_fallback_to_token_http(tmp_path, monkeypatch, variable):
    monkeypatch.setenv(variable, 'private_token_sentinel')
    github = GitHub(executable=str(tmp_path / 'missing-gh'))
    with pytest.raises(TransportError) as caught:
        _REQUEST(github, 'POST', f'repos/{github.repository}/issues', {'title': 'title', 'body': 'body'})
    assert caught.value.permanent and not caught.value.uncertain
    assert str(caught.value) == 'github_client_missing'


def test_request_uses_include_and_stdin_without_credentials(monkeypatch):
    monkeypatch.setenv('GH_TOKEN', 'private_token_sentinel')
    def owned(command, data, **options):
        assert '--include' in command and '--input' in command
        assert 'private_token_sentinel' not in repr(command)
        assert data == b'{"title": "title", "body": "body"}'
        return 0, b'HTTP/2 201 Created\n\n{}', 33
    monkeypatch.setattr(reporter, '_run_gh', owned)
    github = GitHub()
    assert _REQUEST(github, 'POST', f'repos/{github.repository}/issues', {'title': 'title', 'body': 'body'}) == {}


def test_real_owned_process_stderr_limit():
    with pytest.raises(TransportError) as caught:
        _run_gh([sys.executable, '-c', 'import os;os.write(2,b"private_stderr_sentinel"*10000)'],
                None, deadline=time.monotonic() + 3, max_bytes=1024)
    assert str(caught.value) == 'github_output_limit'


def test_prelaunch_cancel_starts_nothing():
    stop = threading.Event()
    stop.set()
    with pytest.raises(TransportError) as caught:
        _run_gh(['unused'], None, deadline=time.monotonic() + 3, max_bytes=1024, cancel=stop)
    assert caught.value.code == 'github_cancelled' and not caught.value.started
