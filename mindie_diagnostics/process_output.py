"""Capture service-process FDs for its own lifetime, never the MCP parent's."""
from contextlib import contextmanager
import os
import re
import sys
import threading

from .context import wrap_context

MAX_LINE_BYTES = 16_384
_LEVEL = re.compile(r"(?:^|\s)(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)(?:\s|:|\])")
_CAPTURE_LOCK = threading.Lock()
_CAPTURE_PID = os.getpid()
_STORAGE_WARNING = "mindie-diagnostics: WARNING diagnostic storage unavailable; business outcome unchanged"


def drain(descriptor, operation, stream, storage_warning=None):
    pending = b""
    discarded = 0
    try:
        while True:
            chunk = os.read(descriptor, 8192)
            if not chunk:
                break
            pieces = chunk.split(b"\n")
            for index, piece in enumerate(pieces):
                if discarded or len(pending) + len(piece) > MAX_LINE_BYTES:
                    discarded += len(pending) + len(piece)
                    pending = b""
                else:
                    pending += piece
                if index < len(pieces) - 1:
                    if discarded:
                        operation.event("WARNING", "process.output.gap", stream=stream,
                                        omitted_bytes=discarded, reason="oversized_line")
                    elif pending:
                        text = pending.decode("utf-8", errors="replace")
                        if (text.strip() == _STORAGE_WARNING
                                and getattr(getattr(operation, "recorder", None), "logging_failed", False)
                                and storage_warning is not None):
                            # Do not recursively send the logger's own fallback
                            # through a failed logger. Re-emit after FD restoration.
                            storage_warning.set()
                        else:
                            match = _LEVEL.search(text[:100])
                            operation.event(match.group(1) if match else "INFO", "process.output",
                                            stream=stream, preview=text)
                    pending, discarded = b"", 0
        if pending or discarded:
            # A final fragment is not proven to contain a complete credential.
            operation.event("WARNING", "process.output.gap", stream=stream,
                            omitted_bytes=len(pending) + discarded, reason="incomplete_line")
    except OSError as exc:
        operation.event("WARNING", "process.output.gap", stream=stream, error_type=type(exc).__name__)
    finally:
        os.close(descriptor)


@contextmanager
def capture_output(operation):
    """Capture this process once; no spawning, ownership, service, or network.

    Enter in the long-lived process itself, never around a detached child's
    lifetime in a short-lived launcher. Concurrent/nested captures reuse the
    existing FD sink and explicitly report that their scope was not installed.
    """
    global _CAPTURE_LOCK, _CAPTURE_PID
    if _CAPTURE_PID != os.getpid():
        _CAPTURE_LOCK, _CAPTURE_PID = threading.Lock(), os.getpid()
    lock = _CAPTURE_LOCK
    if not lock.acquire(blocking=False):
        operation.event("WARNING", "process.output.gap", reason="capture_already_active")
        yield
        return
    saved, readers = [], []
    storage_warning = threading.Event()
    try:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()  # Keep bytes written before this scope outside it.
            except Exception:
                pass
        try:
            for descriptor, stream in ((1, "stdout"), (2, "stderr")):
                read_fd, write_fd = os.pipe()
                backup = None
                try:
                    backup = os.dup(descriptor)
                    os.dup2(write_fd, descriptor)
                    thread = threading.Thread(target=wrap_context(drain),
                                              args=(read_fd, operation, stream, storage_warning), daemon=True,
                                              name="mindie-output-" + stream)
                    thread.start()
                except BaseException:
                    if backup is not None:
                        os.dup2(backup, descriptor)
                        os.close(backup)
                    os.close(read_fd)
                    raise
                finally:
                    os.close(write_fd)
                saved.append((descriptor, backup))
                readers.append(thread)
        except Exception as exc:
            # Keep successfully captured streams. Unmodified FDs retain their
            # caller-selected fallback (detached launchers use DEVNULL).
            operation.event("WARNING", "process.output.gap", error_type=type(exc).__name__, reason="capture_unavailable")
        yield
    finally:
        try:
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except Exception:
                    pass
            for descriptor, backup in saved:
                try:
                    os.dup2(backup, descriptor)
                except OSError as exc:
                    operation.event("WARNING", "process.output.gap", error_type=type(exc).__name__, reason="restore_unavailable")
                finally:
                    try:
                        os.close(backup)
                    except OSError:
                        pass
            for thread in readers:
                thread.join(timeout=2)
                if thread.is_alive():
                    operation.event("WARNING", "process.output.gap", reason="drain_incomplete")
            if storage_warning.is_set():
                try:
                    sys.stderr.write(_STORAGE_WARNING + "\n")
                    sys.stderr.flush()
                except Exception:
                    pass
        finally:
            lock.release()
