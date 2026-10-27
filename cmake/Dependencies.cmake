# Dependencies.cmake
if(PROJECT_IS_TOP_LEVEL AND NOT DEFINED CPM_SOURCE_CACHE AND NOT DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_SOURCE_CACHE "${PROJECT_SOURCE_DIR}/.cache/cpm"
        CACHE PATH "CPM dependency source cache")
endif()
if(NOT COMMAND CPMAddPackage)
    include("${CMAKE_CURRENT_LIST_DIR}/CPM.cmake")
endif()

# Versions are declared here and nowhere else: the fetch below and the installed
# package config both read them.
set(MHS_EIGEN_VERSION 5.0.0)
set(MHS_SPDLOG_VERSION 1.17.0)
set(MHS_TINYXML2_VERSION 11.0.0)
set(MHS_MUPARSER_VERSION 2.3.5)
set(MHS_TBB_VERSION 2023.0.0)
set(MHS_AMGCL_VERSION 1.5.0)
set(MHS_GTEST_VERSION 1.18.0)

find_package(OpenMP REQUIRED COMPONENTS CXX)

# ----------------------------------------------------------------------------
# Eigen - header-only linear algebra. SYSTEM keeps a dependency's headers out of
# this project's warning policy. The CMake package is only generated when it is
# going to be installed next to us. Sourced from the GitHub mirror: anonymous
# clones of gitlab.com fail from CI runners while GitLab is under load.
# ----------------------------------------------------------------------------
if(NOT TARGET Eigen3::Eigen)
    CPMFindPackage(
        NAME Eigen3
        VERSION ${MHS_EIGEN_VERSION}
        FIND_PACKAGE_ARGUMENTS CONFIG
        GITHUB_REPOSITORY eigen-mirror/eigen
        GIT_TAG ${MHS_EIGEN_VERSION}
        SYSTEM YES
        OPTIONS "EIGEN_BUILD_CMAKE_PACKAGE ${MHS_INSTALL}"
    )
endif()

# ----------------------------------------------------------------------------
# spdlog - logging. Installed next to us because logger.hpp is a public header.
# ----------------------------------------------------------------------------
if(NOT TARGET spdlog::spdlog)
    CPMFindPackage(
        NAME spdlog
        VERSION ${MHS_SPDLOG_VERSION}
        FIND_PACKAGE_ARGUMENTS CONFIG
        GITHUB_REPOSITORY gabime/spdlog
        GIT_TAG "v${MHS_SPDLOG_VERSION}"
        SYSTEM YES
        OPTIONS "SPDLOG_INSTALL ${MHS_INSTALL}"
    )
endif()

# ----------------------------------------------------------------------------
# tinyxml2 - XML model reader/writer.
# ----------------------------------------------------------------------------
if(NOT TARGET tinyxml2::tinyxml2)
    CPMFindPackage(
        NAME tinyxml2
        VERSION ${MHS_TINYXML2_VERSION}
        FIND_PACKAGE_ARGUMENTS CONFIG
        GITHUB_REPOSITORY leethomason/tinyxml2
        GIT_TAG ${MHS_TINYXML2_VERSION}
        SYSTEM YES
    )
endif()

# ----------------------------------------------------------------------------
# muparser - mathematical expression parser, built static.
# ----------------------------------------------------------------------------
if(NOT TARGET muparser::muparser)
    CPMFindPackage(
        NAME muparser
        VERSION ${MHS_MUPARSER_VERSION}
        FIND_PACKAGE_ARGUMENTS CONFIG
        GITHUB_REPOSITORY beltoforion/muparser
        GIT_TAG "v${MHS_MUPARSER_VERSION}"
        SYSTEM YES
        OPTIONS
        "ENABLE_SAMPLES OFF"
        "BUILD_TESTING OFF"
        "BUILD_SHARED_LIBS OFF"
    )
endif()

# ----------------------------------------------------------------------------
# oneTBB - parallel assembly.
# ----------------------------------------------------------------------------
if(NOT TARGET TBB::tbb)
    CPMFindPackage(
        NAME TBB
        VERSION ${MHS_TBB_VERSION}
        FIND_PACKAGE_ARGUMENTS CONFIG
        GITHUB_REPOSITORY oneapi-src/oneTBB
        GIT_TAG "v${MHS_TBB_VERSION}"
        SYSTEM YES
        OPTIONS
        "BUILD_SHARED_LIBS ON"
        "TBB_STRICT OFF"
        "TBBMALLOC_BUILD OFF"
        "TBB_TEST OFF"
    )
endif()

# TBB 2023.0.0 needs two MinGW compatibility settings in Debug builds.
if(MINGW AND TARGET tbb)
    target_compile_definitions(tbb
        PUBLIC TBB_USE_PROFILING_TOOLS=0
        PRIVATE _WIN32_WINNT=0x0A00 WINVER=0x0A00
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        get_target_property(_tbb_compile_options tbb COMPILE_OPTIONS)
        if(_tbb_compile_options)
            list(REMOVE_ITEM _tbb_compile_options "-U__STRICT_ANSI__")
            set_property(TARGET tbb PROPERTY COMPILE_OPTIONS "${_tbb_compile_options}")
        endif()
    endif()
endif()

# ----------------------------------------------------------------------------
# AMGCL's builtin backend only needs headers. Do not run its upstream build,
# probe Boost, or inherit compiler flags; mhs_linear links OpenMP explicitly.
# ----------------------------------------------------------------------------
if(NOT TARGET amgcl::amgcl)
    CPMAddPackage(
        NAME amgcl
        GITHUB_REPOSITORY ddemidov/amgcl
        GIT_TAG ${MHS_AMGCL_VERSION}
        DOWNLOAD_ONLY YES
    )
    add_library(mhs_amgcl INTERFACE)
    add_library(amgcl::amgcl ALIAS mhs_amgcl)
    target_include_directories(mhs_amgcl SYSTEM INTERFACE "${amgcl_SOURCE_DIR}")
endif()

# ----------------------------------------------------------------------------
# googletest (tests only, never exported)
# ----------------------------------------------------------------------------
if(MHS_BUILD_TESTS AND NOT TARGET GTest::gtest_main)
    CPMFindPackage(
        NAME GTest
        VERSION ${MHS_GTEST_VERSION}
        FIND_PACKAGE_ARGUMENTS CONFIG
        GITHUB_REPOSITORY google/googletest
        GIT_TAG "v${MHS_GTEST_VERSION}"
        SYSTEM YES
        OPTIONS
        "BUILD_GMOCK OFF"
        "INSTALL_GTEST OFF"
    )
endif()

# ----------------------------------------------------------------------------
# Intel oneMKL (optional, provides MKL::MKL). Pardiso is the direct solver; the
# AMG-preconditioned CG (AmgCg) default does not depend on MKL.
# ----------------------------------------------------------------------------
set(MHS_ENABLE_PARDISO OFF)

if(MHS_USE_MKL)
    # Consumed by oneMKL's own MKLConfig.cmake (MKL_LINK etc.); `sdl` is a
    # valid link mode when MPI is off (the default here).
    set(MKL_LINK "sdl" CACHE STRING "MKL link type (sdl|static|dynamic)")
    set(MKL_THREADING "intel_thread" CACHE STRING "MKL threading runtime")
    set(MKL_INTERFACE "lp64" CACHE STRING "MKL index interface (lp64|ilp64)")
    find_package(MKL CONFIG REQUIRED)
    set(MHS_ENABLE_PARDISO ON)
endif()
