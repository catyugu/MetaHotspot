"""Build, test, install and package MetaHotspot."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def check_sdk(prefix, build, config, generator):
    for pattern in ("muParser*.h", "tinyxml2.h"):
        for leaked in (prefix / "include").glob(pattern):
            raise RuntimeError(f"Private dependency header installed at SDK root: {leaked}")
    headers = prefix / "include/metahotspot"
    for header in headers.rglob("*.hpp"):
        if "Eigen" in header.read_text(encoding="utf-8"):
            raise RuntimeError(f"Eigen leaked into public header: {header}")
    linear_headers = {p.name for p in (headers / "linear").glob("*.hpp")}
    if linear_headers != {"solver.hpp"}:
        raise RuntimeError(
            f"Unexpected public linear backend headers: {linear_headers}"
        )
    for package in prefix.rglob("*MetaHotspot*.cmake"):
        if "Eigen" in package.read_text(encoding="utf-8"):
            raise RuntimeError(f"Eigen leaked into installed package: {package}")
    consumer = build / "sdk-consumer"
    consumer.mkdir(exist_ok=True)
    (consumer / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\n"
        "project(SDKConsumer LANGUAGES CXX)\n"
        "find_package(MetaHotspot CONFIG REQUIRED)\n"
        "add_executable(consumer main.cpp)\n"
        "target_link_libraries(consumer PRIVATE mhs::metahotspot)\n",
        encoding="utf-8",
    )
    (consumer / "main.cpp").write_text(
        '#include "solver/solver.hpp"\n'
        '#include "core/matrix.hpp"\n'
        '#include "linear/solver.hpp"\n'
        "int main() {\n"
        "  mhs::core::CSRMatrix<double> a(1, 1, {0, 1}, {0}, {2.0});\n"
        "  mhs::core::Matrix<double> dense(1, 1);\n"
        "  dense(0, 0) = a.coeff(0, 0);\n"
        "  mhs::core::Vector<double> b{6.0}, x{0.0};\n"
        "  auto solver = mhs::linear::create_solver();\n"
        "  solver->compute(a.view());\n"
        "  return solver->solve(b, x).success ? 0 : 1;\n"
        "}\n",
        encoding="utf-8",
    )
    run(
        "cmake",
        "-S",
        consumer,
        "-B",
        consumer / "build",
        "-G",
        generator,
        f"-DCMAKE_BUILD_TYPE={config}",
        f"-DCMAKE_PREFIX_PATH={prefix}",
        "-DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=TRUE",
    )
    run("cmake", "--build", consumer / "build", "--config", config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--config", default="Release")
    parser.add_argument("--generator", default="Ninja")
    parser.add_argument("--cmake-arg", action="append", default=[])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    build = Path(args.build_dir).resolve()
    prefix = build / "package"
    flags = ["-G", args.generator, f"-DCMAKE_BUILD_TYPE={args.config}", *args.cmake_arg]
    run("cmake", "-S", root, "-B", build, *flags)
    run("cmake", "--build", build, "--config", args.config, "--parallel", "4")
    run("ctest", "--test-dir", build, "-C", args.config, "--output-on-failure")
    if args.config == "Release":
        executable = build / "apps/metahotspot/metahotspot"
        if sys.platform == "win32":
            executable = executable.with_suffix(".exe")
        os.environ["MHS_EXECUTABLE"] = str(executable)
        os.environ["PYTHONIOENCODING"] = "utf-8"
        run(sys.executable, root / "run_cases.py")
        run("cmake", "--install", build, "--config", args.config, "--prefix", prefix)
        check_sdk(prefix, build, args.config, args.generator)
        run(
            "cpack",
            "--config",
            build / "CPackConfig.cmake",
            "-C",
            args.config,
            "-B",
            build,
        )


if __name__ == "__main__":
    main()
