"""Pure, bounded exception evidence for child-process JSON transport.

[Role]
Capture owned exception frames and validate evidence received by record_failure.

[Invariants]
Only canonical component frames, exception class identifiers, and integer errno
are retained. Never serialize messages, source paths, source text, or locals.
Transport is untrusted: unknown fields are dropped and fingerprints recomputed.

[Timeouts & Locks]
No locks; walks stop after 64 exceptions and 4096 traceback links per exception.
Output is bounded to 24 frames and eight exception class names.

[Side-Effect Policy]
No file or network I/O, writes, reporting, or consent changes. source_root is a
caller-declared trusted absolute source directory used only for lexical mapping.
"""
from collections import deque
import hashlib
import json
import os
import re

_IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,99}\Z')
_MODULE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z')
_REASON_CODE = re.compile(r'[a-z][a-z0-9_]{0,63}\Z')
_LABEL = re.compile(r'[A-Za-z][A-Za-z0-9_.:-]{0,119}\Z')
_SOC_GENERATIONS = frozenset({'a2', 'a3', 'a5'})
_SPECIAL_FUNCTIONS = frozenset({'<module>', '<lambda>', '<listcomp>', '<dictcomp>', '<setcomp>', '<genexpr>'})


def component_module_name(component) -> str:
    """Map canonical component distribution/plugin name to Python module root."""
    if type(component) is not str or not component:
        return ''
    if component == 'mindie-knowledge':
        return 'mindie_knowledge'
    if component == 'remote-dev':
        return 'remote_dev'
    if component == 'npu-top':
        return 'npu_top'
    return component.replace('-', '_')


def validate_evidence(component, evidence):
    """Project untrusted JSON transport; fingerprints are always recomputed."""
    from .redact import scan_text
    module = component_module_name(component)
    if not module or not _MODULE.fullmatch(module) or type(evidence) is not dict:
        return {}
    result, frames = {}, []
    truncated = evidence.get('stack_truncated') is True
    incomplete = evidence.get('stack_incomplete') is True
    values = evidence.get('stack_frames')
    if type(values) is list:
        if len(values) > 256:
            truncated = True
        for frame in values[-256:]:
            if type(frame) is not dict:
                continue
            name, function, line = (frame.get(k) for k in ('module', 'function', 'line'))
            if type(name) is not str or len(name) > 160 or not _MODULE.fullmatch(name):
                continue
            if name != module and not name.startswith(module + '.'):
                continue
            if type(function) is not str or not (_IDENT.fullmatch(function) or function in _SPECIAL_FUNCTIONS):
                continue
            if type(line) is not int or not 0 < line < 10**7 or scan_text(name + ' ' + function):
                continue
            item = dict(module=name, function=function, line=line)
            if frame.get('init') is True:
                item['init'] = True
            frames.append(item)
    if len(frames) > 24:
        truncated = True
        frames = frames[-24:]
    if frames:
        result['stack_frames'] = frames
        canonical = json.dumps(frames, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
        result['stack_fingerprint'] = hashlib.sha256(canonical.encode()).hexdigest()
        if truncated or incomplete:
            result['stack_truncated'] = True
        if incomplete:
            result['stack_incomplete'] = True
    chain = evidence.get('exception_chain')
    if type(chain) is list:
        names = [x for x in chain[:8] if type(x) is str and _IDENT.fullmatch(x) and not scan_text(x)]
        if names:
            result['exception_chain'] = names
    number = evidence.get('errno')
    if type(number) is int and -(2**31) <= number < 2**31:
        result['errno'] = number
    exit_code = evidence.get('exit_code')
    if type(exit_code) is int and -(2**31) <= exit_code < 2**31:
        result['exit_code'] = exit_code
    reason_code = evidence.get('reason_code')
    if type(reason_code) is str and _REASON_CODE.fullmatch(reason_code) and not scan_text(reason_code):
        result['reason_code'] = reason_code
    for key in ('substage', 'contract_name'):
        val = evidence.get(key)
        if type(val) is str and _LABEL.fullmatch(val) and not scan_text(val):
            result[key] = val
    soc = evidence.get('soc_generation')
    if type(soc) is str and soc in _SOC_GENERATIONS:
        result['soc_generation'] = soc
    return result


def exception_evidence(component, exception, *, source_root=None):
    """Capture owned traceback frames using lexical absolute paths, without I/O.

    source_root must be a trusted caller-declared owned source directory. It is
    used only for mapping script filenames to canonical component module names.
    No source contents, locals, messages, or paths enter the returned dictionary.
    """
    try:
        module = component_module_name(component)
        root = os.path.normpath(os.fspath(source_root)) if source_root is not None else None
        if root is not None and not os.path.isabs(root):
            root = None
        frames, chain, seen = [], [], set()
        truncated, incomplete = False, False
        current = exception
        number = None
        reason_code = None
        for _ in range(64):
            if not isinstance(current, BaseException) or id(current) in seen:
                break
            seen.add(id(current))
            chain.append(type(current).__name__)
            if number is None and isinstance(current, OSError) and not isinstance(current, TimeoutError) and type(current.errno) is int:
                number = current.errno
            if reason_code is None:
                for attr in ('reason_code', 'code'):
                    candidate = getattr(current, attr, None)
                    if type(candidate) is str and _REASON_CODE.fullmatch(candidate):
                        reason_code = candidate
                        break
            tb = current.__traceback__
            tail_tbs = deque(maxlen=256)
            walked = 0
            while tb is not None and walked < 4096:
                tail_tbs.append((tb.tb_frame, tb.tb_lineno))
                tb = tb.tb_next
                walked += 1
            if walked > 256:
                truncated = True
            if tb is not None:
                truncated = True
                incomplete = True
            for frame, lineno in tail_tbs:
                name = frame.f_globals.get('__name__')
                filename = frame.f_code.co_filename
                is_init = isinstance(filename, str) and os.path.basename(filename.replace('\\', '/')) == '__init__.py'
                if root and isinstance(filename, str) and os.path.isabs(filename):
                    path = os.path.normpath(filename)
                    try:
                        owned = os.path.commonpath((root, path)) == root
                    except ValueError:
                        # Windows paths on different drives have no common root.
                        owned = False
                    if owned:
                        relative = os.path.relpath(path, root)
                        if relative.endswith('.py'):
                            parts = [p for p in relative[:-3].replace('\\', '/').split('/') if p and p != '.']
                            if parts and parts[-1] == '__init__':
                                parts.pop()
                                is_init = True
                            if parts and parts[0] == module:
                                name = '.'.join(parts)
                            else:
                                name = '.'.join([module] + parts)
                entry = dict(module=name, function=frame.f_code.co_name, line=lineno)
                if is_init:
                    entry['init'] = True
                valid = validate_evidence(component, {'stack_frames': [entry]}).get('stack_frames', [])
                if valid:
                    frames.extend(valid)
            current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
        if isinstance(current, BaseException) and id(current) not in seen:
            truncated = True
            incomplete = True
        if len(chain) > 8:
            truncated = True
            chain = [chain[0], *chain[-7:]]
        payload = dict(stack_frames=frames, exception_chain=chain, errno=number)
        if truncated or len(frames) > 24:
            payload['stack_truncated'] = True
        if incomplete:
            payload['stack_incomplete'] = True
        if reason_code is not None:
            payload['reason_code'] = reason_code
        return validate_evidence(component, payload)
    except Exception:
        return {}
