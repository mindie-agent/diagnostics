"""Windows path identity and SQLite filenames without changing I/O locations.

[Role] Compare local path spellings and encode read-only/read-write SQLite URIs.
[Invariants] Extended paths retain long-path support and virtualenv identity.
[Timeouts & Locks] Pure path arithmetic; no filesystem reads, locks or waits.
[Side-Effect Policy] Returns strings only; never changes files or configuration.
"""
from __future__ import annotations

import ntpath
import os
from pathlib import Path
from urllib.parse import quote


def path_identity(value):
    """Compare drive paths without resolving a virtualenv into its base Python."""
    text = os.fspath(value)
    if os.name == "nt":
        logical = text[4:] if text.startswith("\\\\?\\") else text
        drive, tail = ntpath.splitdrive(logical)
        if drive and tail.startswith(("\\", "/")):
            return ntpath.normcase(ntpath.normpath(logical))
    return text


def sqlite_file_uri(value):
    path = Path(value).absolute()
    # Path.as_uri() treats the extended namespace's '?' as a URI authority.
    if os.name == "nt" and str(path).startswith("\\\\?\\"):
        return "file:" + quote(str(path), safe=":")
    return path.as_uri()
