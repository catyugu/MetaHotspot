"""Process-wide native logging configuration."""

from __future__ import annotations

import ctypes
from collections.abc import Callable

from metahotspot._error import check
from metahotspot._lib import get_dll
from metahotspot.enums import LogLevel

_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.c_char_p, ctypes.c_void_p)
_callback_handle: ctypes._CFuncPtr | None = None


def set_log_level(level: LogLevel) -> None:
    """Set the runtime minimum logging level, overriding MHS_LOG_LEVEL."""
    check(get_dll().mhs_log_set_level(int(level)), "set_log_level")


def get_log_level() -> LogLevel:
    """Return the current native logging level."""
    return LogLevel(get_dll().mhs_log_get_level())


def set_log_callback(callback: Callable[[LogLevel, str], None] | None) -> None:
    """Install or remove a process-wide synchronous native logging callback."""
    global _callback_handle
    dll = get_dll()
    if callback is None:
        dll.mhs_log_set_callback(None, None)
        _callback_handle = None
        return

    def forward(level: int, message: bytes, _context: int) -> None:
        callback(LogLevel(level), message.decode("utf-8", errors="replace"))

    handle = _CALLBACK(forward)
    dll.mhs_log_set_callback(ctypes.cast(handle, ctypes.c_void_p), None)
    _callback_handle = handle
