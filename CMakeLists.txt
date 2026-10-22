cmake_minimum_required(VERSION 3.26)

# The same source added twice (two subprojects each add_subdirectory this repo)
# would define the targets twice; leave the existing ones in place instead.
if(TARGET mhs::metahotspot)
    return()
endif()

# The public header owns the version; the project version, the installed CMake
# package report and the Python wheel all derive from these three macros.
foreach(_part MAJOR MINOR PATCH)
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/metahotspot/metahotspot_c_api.h" _mhs_version_line
        REGEX "^#define MHS_VERSION_${_part} ([0-9]+)$")
    string(REGEX REPLACE ".* ([0-9]+)$" "\\1" _mhs_version_${_part} "${_mhs_version_line}")
endforeach()
project(MetaHotspot VERSION ${_mhs_version_MAJOR}.${_mhs_version_MINOR}.${_mhs_version_PATCH} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
    endif()
endif()

# Standalone builds default to everything on; as a subproject only the library.
option(MHS_BUILD_TESTS "Build unit tests" ${PROJECT_IS_TOP_LEVEL})
option(MHS_BUILD_APPS "Build the command-line applications" ${PROJECT_IS_TOP_LEVEL})
option(MHS_BUILD_C_API "Build the shared C/Python API" ON)
option(MHS_INSTALL "Install libraries, headers and CMake package" ${PROJECT_IS_TOP_LEVEL})
option(MHS_WERROR "Treat project warnings as errors" ${PROJECT_IS_TOP_LEVEL})
option(MHS_VERBOSE "Enable DEBUG level logging" OFF)
option(USE_MKL "Enable optional Intel MKL-backed Pardiso solver" OFF)

include(GNUInstallDirs)
# Dependencies come from installed packages or the pinned CPM versions only.
set(CMAKE_FIND_USE_PACKAGE_REGISTRY OFF)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(cmake/MhsOptions.cmake)
include(cmake/Dependencies.cmake)
include(cmake/RuntimeDlls.cmake)

add_subdirectory(src)

add_library(mhs_metahotspot INTERFACE)
add_library(mhs::metahotspot ALIAS mhs_metahotspot)
set_target_properties(mhs_metahotspot PROPERTIES EXPORT_NAME metahotspot)
target_link_libraries(mhs_metahotspot INTERFACE mhs_compiler mhs_solver mhs_io mhs_logging)
target_compile_features(mhs_metahotspot INTERFACE cxx_std_20)

if(MHS_BUILD_TESTS)
    if(PROJECT_IS_TOP_LEVEL)
        enable_testing()
    endif()
    add_subdirectory(tests)
endif()
if(MHS_BUILD_APPS)
    add_subdirectory(apps/metahotspot)
endif()
if(MHS_INSTALL)
    include(cmake/Install.cmake)
endif()
