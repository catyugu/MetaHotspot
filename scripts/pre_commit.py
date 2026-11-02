"""Format and lint the sources a commit is about to carry.

Run it without arguments to rewrite and re-check every tracked source in scope;
the pre-commit hook passes --staged so only the content being committed is
inspected. clang-format owns src, include and tests; ruff owns python and ci.
"""

import argparse
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp"}
CLANG_PATHS = ("src", "include", "tests")
RUFF_PATHS = ("python", "ci")


def git(*args, stdin=None):
    result = subprocess.run(("git", *args), cwd=ROOT, input=stdin, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(result.stderr.decode("utf-8", "replace").strip())
    return result.stdout


@cache
def tool(name):
    """Resolve a tool, preferring the project virtualenv over PATH."""
    for candidate in (
        ROOT / ".venv" / "Scripts" / f"{name}.exe",
        ROOT / ".venv" / "bin" / name,
    ):
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if found is None:
        raise SystemExit(f"{name} is not installed; see AGENTS.md for the pinned tools")
    return found


def run(command, source):
    return subprocess.run(command, cwd=ROOT, input=source, capture_output=True)


def failed(result, path):
    """Report why a tool refused the source and stop before the commit."""
    sys.stderr.write(result.stderr.decode("utf-8", "replace"))
    raise SystemExit(f"{Path(result.args[0]).stem} rejected {path}")


def process(path, source):
    """Return the rewritten source and whatever ruff still has to report."""
    location = Path(path)
    if location.parts[0] in CLANG_PATHS and location.suffix in CPP_SUFFIXES:
        result = run((tool("clang-format"), f"--assume-filename={path}"), source)
        if result.returncode != 0:
            failed(result, path)
        return result.stdout, ""
    if location.parts[0] not in RUFF_PATHS or location.suffix != ".py":
        return source, ""
    fixed = run((tool("ruff"), "check", "--fix", "--stdin-filename", path, "-"), source)
    if fixed.returncode not in (0, 1):
        failed(fixed, path)
    formatted = run(
        (tool("ruff"), "format", "--stdin-filename", path, "-"), fixed.stdout
    )
    if formatted.returncode != 0:
        failed(formatted, path)
    report = fixed.stderr.decode("utf-8", "replace") if fixed.returncode else ""
    return formatted.stdout, report


def index_entries():
    """Map every file in the index to its mode and blob id."""
    entries = {}
    for record in git("ls-files", "-s", "-z").split(b"\0"):
        if record:
            info, name = record.split(b"\t", 1)
            mode, blob, _stage = info.decode().split()
            entries[name.decode()] = (mode, blob)
    return entries


def staged_paths():
    listing = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    return [name.decode() for name in listing.split(b"\0") if name]


def tracked_paths():
    return [name.decode() for name in git("ls-files", "-z").split(b"\0") if name]


def store_blob(path, mode, content):
    blob = git("hash-object", "-w", "--stdin", stdin=content).decode().strip()
    git("update-index", "--cacheinfo", f"{mode},{blob},{path}")


def sync_worktree(path, staged, rewritten):
    """Keep the working copy in step with the rewritten staged blob.

    A working copy that still matches what was staged is rewritten; one that
    differs only by CRLF endings is rewritten with those endings kept; a working
    copy that carries unstaged edits is left alone.
    """
    file = ROOT / path
    if not file.is_file():
        return
    content = file.read_bytes()
    if content == staged:
        file.write_bytes(rewritten)
    elif content.replace(b"\r\n", b"\n") == staged:
        file.write_bytes(rewritten.replace(b"\n", b"\r\n"))
    else:
        print(f"  {path}: unstaged changes kept, only the staged copy is rewritten")


def process_staged():
    entries = index_entries()
    changes, findings = [], []
    for path in staged_paths():
        if path not in entries:
            continue
        mode, blob = entries[path]
        source = git("cat-file", "blob", blob)
        content, report = process(path, source)
        if report:
            findings.append((path, report))
        if content != source:
            store_blob(path, mode, content)
            sync_worktree(path, source, content)
            changes.append(path)
    return changes, findings


def process_tracked():
    changes, findings = [], []
    for path in tracked_paths():
        file = ROOT / path
        if not file.is_file():
            continue
        source = file.read_bytes()
        content, report = process(path, source)
        if report:
            findings.append((path, report))
        if content != source:
            file.write_bytes(content)
            changes.append(path)
    return changes, findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged",
        action="store_true",
        help="check only the files staged for commit",
    )
    args = parser.parse_args()
    changes, findings = process_staged() if args.staged else process_tracked()
    for path in changes:
        print(f"rewrote {path}")
    for path, report in findings:
        sys.stderr.write(f"\n{path}\n{report}")
    if findings:
        print(f"{len(findings)} file(s) still fail ruff check", file=sys.stderr)
        return 1
    print(f"{len(changes)} file(s) rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
