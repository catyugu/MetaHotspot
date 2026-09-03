"""Internal error-checking helper."""

from __future__ import annotations


class MetaHotspotError(RuntimeError):
    """Raised when a C API function fails."""

    def __init__(self, context: str = "", msg: str = ""):
        self.context = context
        full = "MetaHotspot API error"
        if context:
            full += f" in {context}"
        if msg:
            full += f": {msg}"
        super().__init__(full)


def check(status: int, ctx: str = "") -> None:
    """Raise MetaHotspotError if the C API reports failure."""
    if status != 0:
        # Avoid circular import — we import _get_dll lazily in the error
        # path only.
        from metahotspot._lib import get_dll

        dll = get_dll()
        err_ptr = dll.mhs_last_error()
        msg = ""
        if err_ptr:
            raw = err_ptr
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            msg = raw
        raise MetaHotspotError(context=ctx, msg=msg)
