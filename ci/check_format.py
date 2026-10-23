"""Fail when a tracked C++ source does not match .clang-format."""

import subprocess
import sys
from pathlib import Path

SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp"}


def tracked_sources(root):
    listing = subprocess.run(
        ["git", "ls-files", "-z", "."],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [
        name for name in listing.split("\0") if Path(name).suffix in SOURCE_SUFFIXES
    ]


def main():
    root = Path(__file__).resolve().parents[1]
    sources = tracked_sources(root)
    result = subprocess.run(
        ["clang-format", "--dry-run", "--Werror", *sources],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        return 1
    print(f"{len(sources)} sources match .clang-format")
    return 0


if __name__ == "__main__":
    sys.exit(main())
