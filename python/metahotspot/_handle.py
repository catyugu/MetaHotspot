"""Thin base class for C API handle wrappers with ownership semantics."""

from __future__ import annotations

import ctypes

from metahotspot._error import check


class OwnedHandle:
    """Thin base for C API handle wrappers with automatic cleanup.

    Handles the common pattern of _dll, _handle, __del__, close.
    Supports context manager protocol.
    """

    def __init__(self, dll, handle, destroy_fn):
        self._dll = dll
        self._handle: ctypes.Structure | None = handle
        self._destroy_fn = destroy_fn

    def _call(self, name: str, *args, ctx: str = ""):
        """Invoke a status-returning C API function taking ``self._handle`` first.

        Raises ``MetaHotspotError`` (via ``check``) on non-zero status.  ``ctx``
        is the error-context label; it defaults to the function name.
        """
        check(getattr(self._dll, name)(self._handle, *args), ctx or name)

    def __del__(self) -> None:
        if hasattr(self, "_handle"):
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._destroy_fn(self._handle)
            self._handle = None
