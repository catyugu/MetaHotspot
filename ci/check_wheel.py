"""Check wheel contents and Windows runtime closure after platform repair."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path


def check_version(actual, maximum, binary):
    def version(value):
        return tuple(int(part) for part in value.split(".")) + (0,) * (
            4 - len(value.split("."))
        )

    if version(actual) > version(maximum):
        raise ValueError(f"{binary}: requires {actual}, baseline is {maximum}")


def check_pe(binary, root, *, sdk=False):
    import pefile

    with pefile.PE(str(binary)) as pe:
        names = [
            entry.dll.decode("ascii")
            for kind in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT")
            for entry in getattr(pe, kind, [])
        ]
    bundled = {p.name.lower(): p for p in root.rglob("*.dll")}
    for name in names:
        lower = name.lower()
        if lower in bundled:
            if not sdk and lower.startswith(("vcruntime", "msvcp")):
                with pefile.PE(str(bundled[lower])) as runtime:
                    version = runtime.VS_FIXEDFILEINFO[0].FileVersionMS
                header = pe.OPTIONAL_HEADER
                check_version(
                    f"{header.MajorLinkerVersion}.{header.MinorLinkerVersion}",
                    f"{version >> 16}.{version & 0xFFFF}",
                    f"{binary}: {name} toolset",
                )
            continue
        if lower.startswith(("api-ms-", "ext-ms-")):
            continue
        crt = lower.startswith(("vcruntime", "msvcp", "ucrtbase"))
        if sdk and (crt or re.fullmatch(r"vcomp[0-9]+d\.dll", lower)):
            # SDKs use the installed toolchain/VC Redistributable. Debug CRT
            # and Debug OpenMP belong to the development toolchain.
            continue
        runtime = crt or lower.startswith(("vcomp", "libomp", "libiomp", "tbb"))
        system = Path(os.environ["SYSTEMROOT"]) / "System32" / name
        if runtime or not system.is_file():
            raise ValueError(f"{binary}: missing DLL {name}")


def inspect(root, *, sdk=False):
    binaries = []
    for binary in root.rglob("*"):
        if not binary.is_file():
            continue
        with binary.open("rb") as stream:
            magic = stream.read(4)
        if magic[:2] == b"MZ":
            check_pe(binary, root, sdk=sdk)
        else:
            continue
        binaries.append(binary)
    if not binaries:
        raise ValueError(f"No native libraries in {root}")
    print(f"Verified {len(binaries)} native binaries in {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--sdk", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp).resolve()
        with zipfile.ZipFile(args.artifact) as archive:
            if not args.sdk:
                wheel = next(
                    n for n in archive.namelist() if n.endswith(".dist-info/WHEEL")
                )
                tags = archive.read(wheel).decode()
                platform = {
                    "linux": "manylinux_2_28_x86_64",
                    "darwin": "macosx_11_0_arm64",
                    "win32": "win_amd64",
                }[sys.platform]
                if f"Tag: py3-none-{platform}" not in tags:
                    raise ValueError(f"Unexpected wheel tags: {tags}")
                library = {
                    "linux": "libmhs_c_api.so",
                    "darwin": "libmhs_c_api.dylib",
                    "win32": "mhs_c_api.dll",
                }[sys.platform]
                if f"metahotspot/{library}" not in archive.namelist():
                    raise ValueError(f"Missing ctypes library: {library}")
            if args.sdk or sys.platform == "win32":
                archive.extractall(root)
                inspect(root, sdk=args.sdk)


if __name__ == "__main__":
    main()
