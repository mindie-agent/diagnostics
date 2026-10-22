"""Read-only Windows process liveness and creation-time probe.

[Role]
Queries Windows `kernel32` (`OpenProcess`, `WaitForSingleObject`,
`GetProcessTimes`) to return `(alive, creation_filetime)` so `_worker_view` can
detect PID reuse without signals or subprocesses.

[Invariants]
- Opens processes with `SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION` only;
  cannot terminate or mutate the target process.
- Inaccessible processes return `(None, None)` (`unverified`), never `False`.

[Timeouts & Locks]
- Zero-wait `WaitForSingleObject(handle, 0)`; no locks or blocking waits.

[Side-Effect Policy]
- Strictly read-only OS handle inspection closed in `finally`.
"""
from __future__ import annotations


def windows_process(pid):
    """Return (alive, creation FILETIME), preserving inaccessible as unknown."""
    if type(pid) is not int or not 0 < pid <= 0xFFFFFFFF:
        return None, None
    import ctypes
    from ctypes import wintypes

    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        # Synchronize and query-limited rights cannot terminate or alter a process.
        handle = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
        if not handle:
            return (False, None) if ctypes.get_last_error() == 87 else (None, None)
        try:
            wait = kernel.WaitForSingleObject(handle, 0)
            if wait == 0:
                return False, None
            if wait != 258:
                return None, None
            created, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time)):
                return None, None
            identity = (created.dwHighDateTime << 32) | created.dwLowDateTime
            return True, identity
        finally:
            kernel.CloseHandle(handle)
    except TimeoutError:
        raise
    except (OSError, AttributeError):
        return None, None
