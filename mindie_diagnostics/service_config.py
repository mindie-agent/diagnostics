"""Read current owned worker arguments without shell interpretation or secrets."""
from __future__ import annotations

import re

_LEGACY_FLAGS = ('--central-bot', '--grok', '--grok-home', '--grok-work')


def worker_options(argv, environment_file):
    """Parse exactly the owned pure local reporter's arguments.

    A preexisting legacy paid-model worker is refused explicitly, never
    silently adopted; an independent reporting policy file is required.
    """
    from .service import ServiceError, _absolute, _since
    if not isinstance(argv, list) or not argv or argv[0] != 'worker':
        raise ServiceError('invalid_reporter_configuration')
    if any(flag in argv for flag in _LEGACY_FLAGS):
        raise ServiceError('legacy_worker_requires_explicit_removal')
    names = {'--state': 'state', '--repository': 'repository', '--gh': 'gh',
             '--since': 'since', '--interval': 'interval',
             '--reporting-config': 'reporting_config'}
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
    if not {'state', 'repository', 'gh', 'since', 'interval', 'reporting_config'} <= result.keys() or not result['roots']:
        raise ServiceError('invalid_reporter_configuration')
    result['state'] = str(_absolute(result['state']))
    result['reporting_config'] = str(_absolute(result['reporting_config']))
    result['since'] = _since(result['since'])
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', result['repository']):
        raise ServiceError('invalid_reporter_configuration')
    try:
        interval = float(result['interval'])
    except ValueError:
        raise ServiceError('invalid_reporter_configuration') from None
    if not 5 <= interval <= 86400:
        raise ServiceError('invalid_reporter_configuration')
    result['gh'] = 'gh' if result['gh'] == 'gh' else str(_absolute(result['gh']))
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
