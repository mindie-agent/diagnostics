import json

from mindie_diagnostics.integration import exception_evidence, record_failure
from mindie_diagnostics.evidence import validate_evidence
from mindie_diagnostics.bundle import export_issue_event


def test_script_evidence_reaches_issue(tmp_path, monkeypatch):
    monkeypatch.setenv('MINDIE_DIAGNOSTICS_CONFIG', str(tmp_path / 'policy'))
    namespace = {'__name__': 'release_activation'}
    code = compile('def activate():\n raise OSError(13, "PRIVATE_MESSAGE", "/private/secret")\nactivate()\n',
                   str(tmp_path / 'release_activation.py'), 'exec')
    try:
        exec(code, namespace)
    except OSError as exc:
        evidence = exception_evidence('mindie-agent-codex', exc, source_root=tmp_path)
    assert evidence['errno'] == 13
    assert evidence['exception_chain'] == ['PermissionError']
    assert evidence['stack_frames'][-1]['module'] == 'mindie_agent_codex.release_activation'
    root = tmp_path / 'logs'
    assert record_failure('mindie-agent-codex', 'activate', stage='end', category='internal',
                          root=root, evidence=json.loads(json.dumps(evidence)))['recorded']
    event = json.loads(next(root.glob('events/*/*.jsonl')).read_text().splitlines()[-1])
    assert event['reporting'] is None
    projected = export_issue_event(event)
    serialized = json.dumps(projected)
    assert 'mindie_agent_codex.release_activation' in serialized
    assert 'PRIVATE_MESSAGE' not in serialized and str(tmp_path) not in serialized
    assert '/private/secret' not in serialized


def test_untrusted_transport_is_projected():
    safe = validate_evidence('mindie-agent-codex', {
        'message': 'secret', 'errno': True, 'stack_fingerprint': 'forged',
        'exception_chain': ['ValueError', '/private/path', 'x' * 101],
        'stack_frames': [{'module': 'user_code', 'function': 'run', 'line': 1},
                         {'module': 'mindie_agent_codex.run', 'function': 'run', 'line': True}]})
    assert safe == {'exception_chain': ['ValueError']}


def test_exception_chain_is_bounded_and_cycle_safe():
    current = ValueError('secret')
    first = current
    for _ in range(20):
        following = RuntimeError('private')
        current.__cause__ = following
        current = following
    current.__cause__ = first
    evidence = exception_evidence('mindie-agent-codex', first)
    assert len(evidence['exception_chain']) == 8
    first.__cause__ = first
    assert exception_evidence('mindie-agent-codex', first)['exception_chain'] == ['ValueError']


def test_source_mapping_is_lexically_confined(tmp_path):
    root = tmp_path / 'owned'
    source = compile('raise ValueError("private")', str(tmp_path / 'owned-other' / 'secret.py'), 'exec')
    try:
        exec(source, {'__name__': '__main__'})
    except ValueError as exc:
        assert 'stack_frames' not in exception_evidence('mindie-agent-codex', exc, source_root=root)
        assert 'stack_frames' not in exception_evidence('mindie-agent-codex', exc, source_root='relative')


def test_different_drive_frame_does_not_discard_owned_evidence(tmp_path, monkeypatch):
    import mindie_diagnostics.evidence as evidence_module
    original = evidence_module.os.path.commonpath
    root = str(tmp_path)

    def commonpath(paths):
        if paths[1] != str(tmp_path / 'release_activation.py'):
            raise ValueError('Paths are on different drives')
        return original(paths)

    monkeypatch.setattr(evidence_module.os.path, 'commonpath', commonpath)
    source = compile('raise ValueError("private")', str(tmp_path / 'release_activation.py'), 'exec')
    try:
        exec(source, {'__name__': '__main__'})
    except ValueError as exc:
        result = exception_evidence('mindie-agent-codex', exc, source_root=root)
    assert result['exception_chain'] == ['ValueError']
    assert result['stack_frames'] == [dict(module='mindie_agent_codex.release_activation',
                                          function='<module>', line=1)]


def test_child_evidence_takes_precedence_over_caught_wrapper(tmp_path, monkeypatch):
    monkeypatch.setenv('MINDIE_DIAGNOSTICS_CONFIG', str(tmp_path / 'policy'))
    child_code = compile('def prepare():\n raise KeyError("private child")\nprepare()\n',
                         str(tmp_path / 'release_activation.py'), 'exec')
    try:
        exec(child_code, {'__name__': '__main__'})
    except KeyError as child:
        transported = json.loads(json.dumps(exception_evidence(
            'mindie-agent-codex', child, source_root=tmp_path)))
    wrapper_code = compile('def activate():\n raise RuntimeError("private wrapper")\nactivate()\n',
                           str(tmp_path / 'wrapper.py'), 'exec')
    root = tmp_path / 'logs'
    try:
        exec(wrapper_code, {'__name__': 'mindie_agent_codex.wrapper'})
    except RuntimeError as wrapper:
        assert record_failure('mindie-agent-codex', 'activate', stage='end', category='internal',
                              exception=wrapper, evidence=transported, root=root)['recorded']
    event = json.loads(next(root.glob('events/*/*.jsonl')).read_text().splitlines()[-1])
    projected = export_issue_event(event)
    attrs = projected['attributes']
    assert attrs['exception_chain'] == ['KeyError']
    assert attrs['stack_frames'][-1]['module'] == 'mindie_agent_codex.release_activation'
    assert attrs['stack_frames'][-1]['function'] == 'prepare'
    assert attrs['stack_fingerprint'] == transported['stack_fingerprint']
    assert 'private child' not in json.dumps(projected)
    assert 'private wrapper' not in json.dumps(projected)
