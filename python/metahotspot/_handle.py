"""Thin base class for C API handle wrappers with ownership semantics."""

from __future__ import annotations

import ctypes


class OwnedHandle:
    """Thin base for C API handle wrappers with automatic cleanup.

    Handles the common pattern of _dll, _handle, _owned, __del__, close.
    Subclasses call init() in __init__ and/or assign from a factory
    classmethod.
    """

    def __init__(self, destroy_fn, dll=None):
        """
        Parameters
        ----------
        destroy_fn : callable
            A single-argument callable (typically a ctypes function on *dll*)
            that destroys the wrapped handle, e.g. ``dll.mhs_model_destroy``.
        dll : ctypes.CDLL or None
            The loaded shared library.
        """
        self._dll = dll
        self._handle: ctypes.Structure | None = None
        self._destroy_fn = destroy_fn
        self._owned = True

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self._owned and self._handle is not None:
            self._destroy_fn(self._handle)
            self._handle = None
