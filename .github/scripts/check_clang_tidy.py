"""Run the pinned clang-tidy on project translation units (Python 3.11+)."""

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache
from pathlib import Path

import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build-lint")
    parser.add_argument("-j", "--jobs", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    build = Path(args.build_dir).resolve()
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = config["tool"]["metahotspot"]["clang-tidy"]["version"]

    binary = shutil.which("clang-tidy")
    if not binary or tool_version(binary) != version:
        directory = build / "clang-tools"
        directory.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "clang-tools",
                "install",
                "clang-tidy",
                "--version",
                version,
                "--directory",
                str(directory),
                "--overwrite",
                "--no-progress-bar",
            ],
            check=True,
        )
        binary = str(
            directory / ("clang-tidy.exe" if sys.platform == "win32" else "clang-tidy")
        )
    if tool_version(binary) != version:
        raise RuntimeError(f"Expected clang-tidy {version}: {binary}")
    subprocess.run([binary, "--verify-config"], cwd=root, check=True)

    database = json.loads((build / "compile_commands.json").read_text(encoding="utf-8"))
    directories = [root / name for name in ("src", "apps", "tests")]
    commands = {
        (Path(entry["directory"]) / entry["file"]).resolve(): entry
        for entry in database
    }
    sources = sorted(
        source
        for source in commands
        if any(source.is_relative_to(directory) for directory in directories)
    )
    if not sources:
        raise RuntimeError("No project translation units in compile_commands.json")

    @cache
    def compiler_headers(compiler):
        # OpenMP headers can live in the build compiler's resource directory.
        # Search it after tidy's own headers when their LLVM versions differ.
        result = subprocess.run(
            [compiler, "-print-file-name=include"], capture_output=True, text=True
        )
        directory = Path(result.stdout.strip())
        if result.returncode == 0 and directory.is_absolute() and directory.is_dir():
            return ["--extra-arg=-idirafter", f"--extra-arg={directory}"]
        return []

    def check(source):
        entry = commands[source]
        command = entry.get("arguments") or shlex.split(entry["command"])
        result = subprocess.run(
            [binary, "-p", str(build), *compiler_headers(command[0]), str(source)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        return source, result

    failed = False
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(check, source) for source in sources]
        for future in as_completed(futures):
            source, result = future.result()
            print(f"clang-tidy: {source.relative_to(root)}", flush=True)
            if result.returncode:
                failed = True
            # Successful runs still report diagnostics that explain suppressions.
            print(result.stdout + result.stderr, end="", flush=True)
    print(f"Checked {len(sources)} translation units with clang-tidy {version}.")
    return int(failed)


def tool_version(binary):
    output = subprocess.check_output([binary, "--version"], text=True)
    match = re.search(r"version (\d+\.\d+\.\d+)", output)
    return match[1] if match else None


if __name__ == "__main__":
    sys.exit(main())
