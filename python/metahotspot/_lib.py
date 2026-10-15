"""Internal helper to locate and load the MetaHotspot C shared library.

Module-level singleton::

    from metahotspot._lib import get_dll
    dll = get_dll()  # first call loads & configures; subsequent calls cached
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
#  Cache for the loaded CDLL
# ---------------------------------------------------------------------------
_dll: ctypes.CDLL | None = None


def get_dll() -> ctypes.CDLL:
    """Return the loaded (and configured) core CDLL — cached singleton."""
    global _dll
    if _dll is None:
        from metahotspot._dll_interface import configure_dll

        dll = load_library()
        configure_dll(dll)
        _dll = dll
    return _dll


def load_library() -> ctypes.CDLL:
    """Find and load the MetaHotspot C shared library.

    Returns the ctypes.CDLL instance.
    An explicit MHS_CAPI_PATH selects a file or directory. Load errors propagate
    immediately; automatic discovery is used only when no path was selected.
    Raises RuntimeError when automatic discovery finds no library.
    """
    explicit = os.environ.get("MHS_CAPI_PATH")
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_dir():
            path /= _lib_name()
        return ctypes.CDLL(str(path.resolve()))

    candidates = _probe_lib_paths()
    for path in candidates:
        if path.is_file():
            return ctypes.CDLL(str(path))
    tried = "\n".join(f"  {path}" for path in candidates)
    raise RuntimeError(
        f"Cannot locate MetaHotspot C API library. Tried:\n{tried}\n"
        "Set MHS_CAPI_PATH to the shared library or its directory."
    )


def _probe_lib_paths() -> list[Path]:
    """Installed package, editable build, then the working directory's build."""
    package = Path(__file__).resolve().parent
    name = _lib_name()
    candidates = (
        package / name,
        package.parent.parent / "build" / "src" / "api" / name,
        Path.cwd() / "build" / "src" / "api" / name,
    )
    return list(dict.fromkeys(candidates))


def _lib_name() -> str:
    if sys.platform.startswith("linux"):
        return "libmhs_c_api.so"
    elif sys.platform == "darwin":
        return "libmhs_c_api.dylib"
    elif sys.platform == "win32":
        return "mhs_c_api.dll"
    else:
        raise OSError(f"Unsupported platform: {sys.platform}")
