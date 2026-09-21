"""Bounded read-only local faults, independent of reporting consent."""
import json
import os
import re
import stat
import time
from . import fallback as f

_FILE = re.compile(r'\d+-[0-9a-f]{32}\.jsonl(?:\.[1-3])?\Z')
_LABEL = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:+-]{0,119}\Z')
_STAMP = re.compile(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,6})?Z\Z')


def _tail(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not f._owned(info) or info.st_nlink != 1:
            return None, False
        start = max(0, info.st_size - 16384)
        os.lseek(fd, start, os.SEEK_SET)
        data = os.read(fd, 16384)
        return (data.partition(b'\n')[2] if start else data), bool(start)
    finally:
        os.close(fd)


def recent_failures(roots):
    """Inspect <=256 entries, eight 16KiB tails, and 150ms; never create files."""
    deadline = time.monotonic() + .15
    result = {'recent': [], 'truncated': False, 'scanned_files': 0}
    files, visited = [], 0
    try:
        for raw in roots[:32]:
            root = f._as_local_absolute(raw)
            if root is None or not f._ancestors_are_real_dirs(root / 'events' / '_'):
                continue
            events = root / 'events'
            if not events.exists():
                continue
            with os.scandir(events) as folders:
                for folder in folders:
                    visited += 1
                    if visited > 256 or time.monotonic() > deadline:
                        result['truncated'] = True
                        break
                    if not folder.is_dir(follow_symlinks=False) or not _LABEL.fullmatch(folder.name):
                        continue
                    with os.scandir(folder.path) as entries:
                        for entry in entries:
                            visited += 1
                            if visited > 256 or time.monotonic() > deadline:
                                result['truncated'] = True
                                break
                            if not _FILE.fullmatch(entry.name) or not entry.is_file(follow_symlinks=False):
                                continue
                            info = entry.stat(follow_symlinks=False)
                            if f._owned(info):
                                files.append((info.st_mtime_ns, entry.path))
                    if result['truncated']:
                        break
            if result['truncated']:
                break
        result['truncated'] |= len(files) > 8
        records = []
        for _, name in sorted(files, reverse=True)[:8]:
            if time.monotonic() > deadline:
                result['truncated'] = True
                break
            try:
                data, partial = _tail(name)
            except OSError:
                result['truncated'] = True
                continue
            result['truncated'] |= partial
            if data is None:
                continue
            result['scanned_files'] += 1
            for line in data.splitlines():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or row.get('event') != 'operation.end' or row.get('status') != 'error':
                        continue
                    iid = row.get('operation_id')
                    if not f._valid_incident(iid):
                        continue
                    item = {'incident_id': iid, 'record_ref': name}
                    for key in ('component', 'operation', 'package_version', 'package_revision'):
                        value = row.get(key)
                        if isinstance(value, str) and _LABEL.fullmatch(value):
                            item[key] = value
                    stamp = row.get('timestamp')
                    if isinstance(stamp, str) and _STAMP.fullmatch(stamp):
                        item['timestamp'] = stamp
                    attrs = row.get('attributes')
                    if isinstance(attrs, dict):
                        for key in ('stage', 'category', 'error_type'):
                            value = attrs.get(key)
                            if isinstance(value, str) and _LABEL.fullmatch(value):
                                item[key] = value
                    records.append(item)
                except (ValueError, TypeError):
                    continue
        records.sort(key=lambda row: row.get('timestamp', ''), reverse=True)
        result['recent'] = records[:5]
        result['truncated'] |= len(records) > 5
    except (OSError, ValueError, TypeError):
        result['truncated'] = True
    return result
