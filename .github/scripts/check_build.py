"""Build, test, install and package MetaHotspot."""

import argparse
import subprocess
from pathlib import Path


def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def check_sdk(prefix, build, config, generator):
    headers = prefix / "include/metahotspot"
    for header in headers.rglob("*.hpp"):
        if "Eigen" in header.read_text(encoding="utf-8"):
            raise RuntimeError(f"Eigen leaked into public header: {header}")
    linear_headers = {p.name for p in (headers / "numerics/linear").glob("*.hpp")}
    if linear_headers != {"linear_solver.hpp"}:
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
        '#include "core/solver.hpp"\n'
        '#include "core/matrix.hpp"\n'
        '#include "numerics/linear/linear_solver.hpp"\n'
        "int main() {\n"
        "  mhs::core::CSRMatrix<double> a(1, 1, {0, 1}, {0}, {2.0});\n"
        "  mhs::core::Matrix<double> dense(1, 1);\n"
        "  dense(0, 0) = a.coeff(0, 0);\n"
        "  mhs::core::Vector<double> b{6.0}, x{0.0};\n"
        "  auto solver = mhs::numerics::create_solver();\n"
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
