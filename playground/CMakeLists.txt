# Standalone build for the reproduction experiments. The library is embedded with
# add_subdirectory, which its root CMakeLists supports: as a non-top-level project
# it defaults tests/tools/install off and exposes the single mhs::metahotspot
# target, so nothing in the main tree needs to know about this directory.
#
#   cmake -S playground -B build-playground -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-playground --parallel
cmake_minimum_required(VERSION 3.26)
project(mhs_playground LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_BINARY_DIR}/metahotspot)
