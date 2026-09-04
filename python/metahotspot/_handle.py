"""Thin base class for C API handle wrappers with ownership semantics."""

from __future__ import annotations

import ctypes


class OwnedHandle:
    """Thin base for C API handle wrappers with automatic cleanup.

    Handles the common pattern of _dll, _handle, __del__, close.
    Supports context manager protocol.
    """

    def __init__(self, dll, handle, destroy_fn):
        self._dll = dll
        self._handle: ctypes.Structure | None = handle
        self._destroy_fn = destroy_fn

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
