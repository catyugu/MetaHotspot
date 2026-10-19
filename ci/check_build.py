"""Build, test, install, and package MetaHotspot."""

import argparse
from pathlib import Path
import subprocess


def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--config", default="Release")
    parser.add_argument("--generator", default="Ninja")
    parser.add_argument("--cmake-arg", action="append", default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = Path(args.build_dir).resolve()
    prefix = build / "package"
    flags = ["-G", args.generator, f"-DCMAKE_BUILD_TYPE={args.config}", *args.cmake_arg]
    run("cmake", "-S", root, "-B", build, *flags)
    run("cmake", "--build", build, "--config", args.config, "--parallel", "4")
    run("ctest", "--test-dir", build, "-C", args.config, "--output-on-failure")
    run("cmake", "--install", build, "--config", args.config, "--prefix", prefix)
    run("cpack", "--config", build / "CPackConfig.cmake", "-C", args.config, "-B", build)


if __name__ == "__main__":
    main()
