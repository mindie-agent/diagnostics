"""Pure, bounded exception evidence for child-process JSON transport.

[Role]
Capture owned exception frames and validate evidence received by record_failure.

[Invariants]
Only canonical component frames, exception class identifiers, and integer errno
are retained. Never serialize messages, source paths, source text, or locals.
Transport is untrusted: unknown fields are dropped and fingerprints recomputed.

[Timeouts & Locks]
No locks; walks stop after eight exceptions and 128 traceback links per exception.
Output is bounded to 24 frames and eight exception class names.

[Side-Effect Policy]
No file or network I/O, writes, reporting, or consent changes. source_root is a
caller-declared trusted absolute source directory used only for lexical mapping.
"""
import hashlib
import json
import os
import re

_IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,99}\Z')
_MODULE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z')


def validate_evidence(component, evidence):
    """Project untrusted JSON transport; fingerprints are always recomputed."""
    from .redact import scan_text
    module = component.replace('-', '_') if type(component) is str else ''
    if not module or not _MODULE.fullmatch(module) or type(evidence) is not dict:
        return {}
    result, frames = {}, []
    values = evidence.get('stack_frames')
    if type(values) is list:
        for frame in values[:24]:
            if type(frame) is not dict:
                continue
            name, function, line = (frame.get(k) for k in ('module', 'function', 'line'))
            if type(name) is not str or len(name) > 160 or not _MODULE.fullmatch(name):
                continue
            if name != module and not name.startswith(module + '.'):
                continue
            if type(function) is not str or not (_IDENT.fullmatch(function) or function in ('<module>', '<lambda>')):
                continue
            if type(line) is not int or not 0 < line < 10**7 or scan_text(name + ' ' + function):
                continue
            frames.append(dict(module=name, function=function, line=line))
    if frames:
        result['stack_frames'] = frames
        canonical = json.dumps(frames, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
        result['stack_fingerprint'] = hashlib.sha256(canonical.encode()).hexdigest()
    chain = evidence.get('exception_chain')
    if type(chain) is list:
        names = [x for x in chain[:8] if type(x) is str and _IDENT.fullmatch(x) and not scan_text(x)]
        if names:
            result['exception_chain'] = names
    number = evidence.get('errno')
    if type(number) is int and -(2**31) <= number < 2**31:
        result['errno'] = number
    return result


def exception_evidence(component, exception, *, source_root=None):
    """Capture owned traceback frames using lexical absolute paths, without I/O.

    source_root must be a trusted caller-declared owned source directory. It is
    used only for mapping script filenames to canonical component module names.
    No source contents, locals, messages, or paths enter the returned dictionary.
    """
    try:
        module = component.replace('-', '_')
        root = os.path.normpath(os.fspath(source_root)) if source_root is not None else None
        if root is not None and not os.path.isabs(root):
            root = None
        frames, chain, seen = [], [], set()
        current = exception
        number = None
        for _ in range(8):
            if not isinstance(current, BaseException) or id(current) in seen:
                break
            seen.add(id(current))
            chain.append(type(current).__name__)
            if number is None and isinstance(current, OSError) and not isinstance(current, TimeoutError) and type(current.errno) is int:
                number = current.errno
            tb = current.__traceback__
            for _ in range(128):
                if tb is None:
                    break
                frame = tb.tb_frame
                name = frame.f_globals.get('__name__')
                filename = frame.f_code.co_filename
                if root and os.path.isabs(filename):
                    path = os.path.normpath(filename)
                    try:
                        owned = os.path.commonpath((root, path)) == root
                    except ValueError:
                        # Windows paths on different drives have no common root.
                        owned = False
                    if owned:
                        relative = os.path.relpath(path, root)
                        if relative.endswith('.py'):
                            parts = relative[:-3].split(os.sep)
                            if parts[-1] == '__init__':
                                parts.pop()
                            name = '.'.join([module] + parts)
                frames.append(dict(module=name, function=frame.f_code.co_name, line=tb.tb_lineno))
                # Validate before bounding, so unrelated outer frames cannot
                # crowd out owned frames deeper in the traceback.
                frames = validate_evidence(component, {'stack_frames': frames}).get('stack_frames', [])
                tb = tb.tb_next
            current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
        return validate_evidence(component, dict(stack_frames=frames, exception_chain=chain, errno=number))
    except Exception:
        return {}
