"""Inspect the final wheel's native dependency closure and platform baseline."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Libraries permitted by the manylinux policy; OpenMP and TBB are never system
# dependencies, regardless of where they lived on the build machine.
ELF_SYSTEM = re.compile(
    r"^(lib(c|m|dl|rt|pthread|gcc_s|stdc\+\+|util|resolv)\.so[.\d]*|ld-linux.*)$"
)


def output(*args):
    return subprocess.check_output(list(map(str, args)), text=True)


def check_version(actual, maximum, binary):
    def version(value):
        return tuple(int(part) for part in value.split(".")) + (0,) * (
            4 - len(value.split("."))
        )

    if version(actual) > version(maximum):
        raise ValueError(f"{binary}: requires {actual}, baseline is {maximum}")


def check_dependency(name, binary, root, search):
    if Path(name).is_absolute():
        raise ValueError(f"{binary}: absolute dependency outside wheel: {name}")
    for directory in search:
        candidate = (directory / name).resolve()
        if candidate.is_relative_to(root.resolve()) and candidate.is_file():
            return
    raise ValueError(f"{binary}: unresolved dependency {name}")


def check_elf(binary, root):
    dynamic = output("readelf", "-d", binary)
    paths = re.findall(r"\((?:RUNPATH|RPATH)\).*\[(.*?)\]", dynamic)
    search = []
    for path in ":".join(paths).split(":"):
        if not path:
            continue
        path = path.replace("${ORIGIN}", str(binary.parent)).replace(
            "$ORIGIN", str(binary.parent)
        )
        directory = Path(path).resolve()
        if not directory.is_relative_to(root):
            raise ValueError(f"{binary}: external runtime search path {path}")
        search.append(directory)
    for name in re.findall(r"\(NEEDED\).*\[(.*?)\]", dynamic):
        if not ELF_SYSTEM.fullmatch(name):
            check_dependency(name, binary, root, search)
    versions = output("readelf", "--version-info", binary)
    for family, maximum in (
        ("GLIBC", "2.28"),
        ("GLIBCXX", "3.4.25"),
        ("CXXABI", "1.3.11"),
    ):
        for actual in re.findall(rf"\b{family}_([0-9.]+)", versions):
            check_version(actual, maximum, binary)


def check_macho(binary, root, baseline):
    if output("lipo", "-archs", binary).strip() != "arm64":
        raise ValueError(f"{binary}: expected arm64")
    commands = output("otool", "-l", binary)
    for actual in re.findall(r"\bminos\s+([0-9.]+)", commands):
        check_version(actual, baseline, binary)
    for block in commands.split("Load command "):
        if "LC_VERSION_MIN_MACOSX" in block:
            actual = re.search(r"\bversion\s+([0-9.]+)", block).group(1)
            check_version(actual, baseline, binary)
    # -D identifies the dylib's own install name, which -L also prints.
    own_ids = output("otool", "-D", binary).splitlines()[1:]
    rpaths = re.findall(r"cmd LC_RPATH\s+cmdsize \d+\s+path (.*?) \(offset", commands)
    search = [Path(p.replace("@loader_path", str(binary.parent))) for p in rpaths]
    for line in output("otool", "-L", binary).splitlines()[1:]:
        name = line.strip().split(" (", 1)[0]
        if name in own_ids or name.startswith(("/usr/lib/", "/System/Library/")):
            continue
        if name.startswith("@loader_path/"):
            check_dependency(
                name[len("@loader_path/") :], binary, root, [binary.parent]
            )
        elif name.startswith("@rpath/"):
            check_dependency(name[len("@rpath/") :], binary, root, search)
        else:
            check_dependency(name, binary, root, [])


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


def inspect(root, *, baseline="11.0", sdk=False):
    binaries = []
    for binary in root.rglob("*"):
        if not binary.is_file():
            continue
        with binary.open("rb") as stream:
            magic = stream.read(4)
        if magic == b"\x7fELF":
            check_elf(binary, root)
        elif magic in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe"):
            check_macho(binary, root, baseline)
        elif magic[:2] == b"MZ":
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
            archive.extractall(root)
        inspect(root, sdk=args.sdk)


if __name__ == "__main__":
    main()
