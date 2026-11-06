"""[Role] Validate the executable used by a Windows scheduled worker.

[Invariants] Only PE GUI subsystem 2 is accepted, regardless of filename.
[Timeouts & Locks] Fixed-size header reads; no subprocesses or locks.
[Side-Effect Policy] Read-only; never replaces an installed interpreter.
"""
from pathlib import Path
import struct


def require_windowless(python):
    """PE32 and PE32+ store Subsystem at optional-header offset 68."""
    from .service import ServiceError
    try:
        with Path(python).open("rb") as stream:
            dos = stream.read(64)
            if len(dos) != 64 or dos[:2] != b"MZ":
                raise ValueError()
            offset = struct.unpack_from("<I", dos, 60)[0]
            if offset < 64:
                raise ValueError()
            stream.seek(offset)
            pe = stream.read(24)
            optional = stream.read(70)
            if (len(pe) != 24 or pe[:4] != b"PE\0\0"
                    or struct.unpack_from("<H", pe, 20)[0] < 70
                    or len(optional) != 70
                    or struct.unpack_from("<H", optional)[0] not in {0x10B, 0x20B}
                    or struct.unpack_from("<H", optional, 68)[0] != 2):
                raise ValueError()
    except TimeoutError:
        raise
    except (OSError, ValueError):
        raise ServiceError("windowless_python_required") from None
