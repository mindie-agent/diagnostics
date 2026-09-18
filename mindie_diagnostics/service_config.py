"""Read current owned worker arguments without shell interpretation or secrets."""
from __future__ import annotations

import re


def worker_options(argv, environment_file):
    from .service import ServiceError, _absolute, _since
    if not isinstance(argv, list) or not argv or argv[0] != 'worker':
        raise ServiceError('invalid_reporter_configuration')
    if '--central-bot' in argv:
        raise ServiceError('central_bot_is_not_a_local_reporter')
    names = {'--state': 'state', '--repository': 'repository', '--gh': 'gh',
             '--since': 'since', '--interval': 'interval', '--grok': 'grok',
             '--grok-home': 'grok_home', '--grok-work': 'grok_work'}
    result = {'roots': [], 'environment_file': environment_file}
    for index in range(1, len(argv), 2):
        if index + 1 >= len(argv) or not isinstance(argv[index + 1], str):
            raise ServiceError('invalid_reporter_configuration')
        flag, value = argv[index:index + 2]
        if flag == '--root':
            result['roots'].append(str(_absolute(value)))
        elif flag in names and names[flag] not in result:
            result[names[flag]] = value
        else:
            raise ServiceError('invalid_reporter_configuration')
    if not {'state', 'repository', 'gh', 'since', 'interval'} <= result.keys() or not result['roots']:
        raise ServiceError('invalid_reporter_configuration')
    result['state'] = str(_absolute(result['state']))
    result['since'] = _since(result['since'])
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', result['repository']):
        raise ServiceError('invalid_reporter_configuration')
    try:
        interval = float(result['interval'])
    except ValueError:
        raise ServiceError('invalid_reporter_configuration') from None
    if not 5 <= interval <= 86400:
        raise ServiceError('invalid_reporter_configuration')
    for key in ('gh', 'grok', 'grok_home', 'grok_work'):
        if result.get(key):
            result[key] = 'gh' if key == 'gh' and result[key] == 'gh' else str(_absolute(result[key]))
        else:
            result[key] = None
    if (bool(result['grok']) != bool(result['grok_home'] and result['grok_work'])
            or (result['grok_home'] or result['grok_work']) and not result['grok']):
        raise ServiceError('invalid_reporter_configuration')
    return result


def merged_reporter_options(argv, environment_file, roots, repository):
    from .service import ServiceError, _absolute
    result = worker_options(argv, environment_file)
    if result['repository'] != repository:
        raise ServiceError('reporter_repository_mismatch')
    result['roots'] = list(dict.fromkeys([*result['roots'], *(str(_absolute(root)) for root in roots)]))
    if len(result['roots']) > 32:
        raise ServiceError('diagnostic_roots_required')
    return result


def linux_worker_configuration(text):
    """Accept exactly the literal quoting emitted by the current installer."""
    from .service import ServiceError, _quoted
    commands = [line[len('ExecStart='):] for line in text.splitlines() if line.startswith('ExecStart=')]
    environments = [line[len('EnvironmentFile='):] for line in text.splitlines() if line.startswith('EnvironmentFile=')]
    if len(commands) != 1 or len(environments) > 1:
        raise ServiceError('invalid_reporter_configuration')
    encoded = commands[0]
    tokens = re.findall(r'"((?:[^"\\]|\\.)*)"', encoded)
    argv = [re.sub(r'\\([\\"])', r'\1', token).replace('%%', '%').replace('$$', '$') for token in tokens]
    if ' '.join(map(_quoted, argv)) != encoded or argv[1:5] != ['-I', '-m', 'mindie_diagnostics.cli', 'worker']:
        raise ServiceError('invalid_reporter_configuration')
    return argv[4:], environments[0].replace('%%', '%') if environments else None
