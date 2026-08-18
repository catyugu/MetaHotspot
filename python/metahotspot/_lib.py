"""Internal helper to locate and load the MetaHotspot C shared library.

Module-level singleton::

    from metahotspot._lib import get_dll
    dll = get_dll()  # first call loads & configures; subsequent calls cached
"""

from __future__ import annotations

import ctypes
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
        _dll = load_library()
        from metahotspot._dll_interface import configure_dll

        configure_dll(_dll)
    return _dll


def load_library() -> ctypes.CDLL:
    """Find and load the MetaHotspot C shared library.

    Returns the ctypes.CDLL instance.
    Raises RuntimeError if the library cannot be found.
    """
    errors: list[str] = []
    for cand in _probe_lib_paths():
        if cand.is_file():
            try:
                return ctypes.CDLL(str(cand))
            except OSError as e:
                errors.append(f"{cand!s}: {e}")
                continue

    msg = "Cannot locate MetaHotspot C API library. Tried:\n"
    for cand in _probe_lib_paths():
        exists = " (found)" if cand.is_file() else " (missing)"
        msg += f"  {cand!s}{exists}\n"
    if errors:
        msg += "Load errors:\n" + "\n".join(f"  {e}" for e in errors)
    msg += (
        "\n\nSet MHS_CAPI_PATH to the full path of the shared library, "
        "or ensure the build tree is at the expected location."
    )
    raise RuntimeError(msg)


def _probe_lib_paths() -> list[Path]:
    """Return a list of candidate paths, most-preferred first."""
    candidates: list[Path] = []

    env = Path.cwd() / "build" / "src" / "api"

    env_var = _env_path()
    if env_var is not None:
        candidates.append(env_var)

    # wheel sibling layout
    this_dir = Path(__file__).resolve().parent
    candidates.append(this_dir / _lib_name())

    # build tree (editable install / in-tree usage)
    repo_root = this_dir.parent.parent
    for build_dir in [repo_root / "build"]:
        api_dir = build_dir / "src" / "api"
        if api_dir.is_dir():
            candidates.append(api_dir / _lib_name())

    candidates.append(env / _lib_name())
    return candidates


def _lib_name() -> str:
    if sys.platform.startswith("linux"):
        return "libmhs_c_api.so"
    elif sys.platform == "darwin":
        return "libmhs_c_api.dylib"
    elif sys.platform == "win32":
        return "mhs_c_api.dll"
    else:
        raise OSError(f"Unsupported platform: {sys.platform}")


def _env_path() -> Path | None:
    import os

    val = os.environ.get("MHS_CAPI_PATH")
    if val:
        p = Path(val)
        if p.is_file():
            return p
        candidate = p / _lib_name()
        if candidate.is_file():
            return candidate
    return None
