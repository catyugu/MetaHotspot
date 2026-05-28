cmake_minimum_required(VERSION 3.16)
project(MetaHotspot VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


option(VERBOSE "Enable DEBUG level logging" OFF)

include(cmake/CompilerOptions.cmake)
include(cmake/Dependencies.cmake)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
add_subdirectory(bin)
