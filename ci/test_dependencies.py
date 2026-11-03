"""Check dependency options without fetching or compiling third-party code."""

import subprocess
from pathlib import Path


def test_muparser_disables_its_openmp_backend(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "CMakeLists.txt"
    source.write_text(
        "cmake_minimum_required(VERSION 3.26)\n"
        "project(DependencyOptions LANGUAGES NONE)\n"
        "add_library(Eigen3::Eigen INTERFACE IMPORTED)\n"
        "add_library(spdlog::spdlog INTERFACE IMPORTED)\n"
        "add_library(tinyxml2::tinyxml2 INTERFACE IMPORTED)\n"
        "add_library(TBB::tbb INTERFACE IMPORTED)\n"
        "add_library(amgcl::amgcl INTERFACE IMPORTED)\n"
        "function(find_package)\nendfunction()\n"
        "function(CPMAddPackage)\nendfunction()\n"
        "function(CPMFindPackage)\n"
        '  if(NOT "ENABLE_OPENMP OFF" IN_LIST ARGN)\n'
        '    message(FATAL_ERROR "muparser must disable OpenMP")\n'
        "  endif()\n"
        "endfunction()\n"
        f'include("{root.as_posix()}/cmake/Dependencies.cmake")\n'
    )
    subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "build")], check=True
    )
