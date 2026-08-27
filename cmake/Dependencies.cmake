# Dependencies.cmake
# Centralized dependency management (CPM / FetchContent), modern-cpp-template
# style. All third-party packages live here; nothing else in the build graph
# brings in external code.

include(CPM)

# ----------------------------------------------------------------------------
# Eigen - header-only from GitLab mirror. EIGEN_BUILD_DOC / _PKGCONFIG / _TESTING
# already default OFF for a non-top-level subproject, so no OPTIONS are needed.
# ----------------------------------------------------------------------------
CPMAddPackage(
    GITLAB_REPOSITORY libeigen/eigen
    GIT_TAG 5.0.0
)

# ----------------------------------------------------------------------------
# spdlog. Tests are gated on SPDLOG_BUILD_TESTS (default OFF), not BUILD_TESTING,
# so no OPTIONS are needed.
# ----------------------------------------------------------------------------
CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY gabime/spdlog
    GIT_TAG v1.17.0
)

# ----------------------------------------------------------------------------
# tinyxml2. Tests are gated on tinyxml2_BUILD_TESTING (defaults to the global
# BUILD_TESTING, which is OFF here), so no OPTIONS are needed.
# ----------------------------------------------------------------------------
CPMAddPackage(
    NAME tinyxml2
    GITHUB_REPOSITORY leethomason/tinyxml2
    GIT_TAG 11.0.0
)

# ----------------------------------------------------------------------------
# muparser - mathematical expression parser.  Build it static.
# ----------------------------------------------------------------------------
CPMAddPackage(
    NAME muparser
    GITHUB_REPOSITORY beltoforion/muparser
    GIT_TAG v2.3.5
    OPTIONS
    "ENABLE_SAMPLES OFF"
    "BUILD_TESTING OFF"
    "BUILD_SHARED_LIBS OFF"
)
# ----------------------------------------------------------------------------
# googletest (tests only)
# -----------------------------------------------------------------------------
CPMAddPackage(
    NAME googletest
    GITHUB_REPOSITORY google/googletest
    GIT_TAG "d72f9c8"
    OPTIONS
    "BUILD_GMOCK OFF"
    "INSTALL_GTEST OFF"
)

# ----------------------------------------------------------------------------
# oneTBB - parallel assembly.  Build it static.
# ----------------------------------------------------------------------------
CPMAddPackage(
    NAME TBB
    GITHUB_REPOSITORY "oneapi-src/oneTBB"
    GIT_TAG "v2023.0.0"
    OPTIONS
    "BUILD_SHARED_LIBS OFF"
    "TBB_STRICT OFF"
    "TBBMALLOC_BUILD OFF"
    "TBB_TEST OFF"
)


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
# amgcl - header-only algebraic multigrid library (builtin backend). Powers
# the AMG-preconditioned CG solver (thermal steady/transient default).
# ----------------------------------------------------------------------------
CPMAddPackage(
    NAME amgcl
    GITHUB_REPOSITORY ddemidov/amgcl
    GIT_TAG 1.5.0
    OPTIONS
    "AMGCL_BUILD_TESTS OFF"
    "AMGCL_BUILD_EXAMPLES OFF"
)

# ----------------------------------------------------------------------------
# Intel oneMKL (optional, provides MKL::MKL). Pardiso is the direct solver;
# the AMG-preconditioned CG (AmgCg) default does not depend on MKL.
# ----------------------------------------------------------------------------
set(MHS_ENABLE_PARDISO FALSE CACHE INTERNAL "" FORCE)

if(USE_MKL)
    # Consumed by oneMKL's own MKLConfig.cmake (MKL_LINK etc.); `sdl` is a
    # valid link mode when MPI is off (the default here).
    set(MKL_LINK "sdl" CACHE STRING "MKL link type (sdl|static|dynamic)")
    set(MKL_THREADING "intel_thread" CACHE STRING "MKL threading runtime")
    set(MKL_INTERFACE "lp64" CACHE STRING "MKL index interface (lp64|ilp64)")
    find_package(MKL QUIET)

    if(TARGET MKL::MKL)
        set(MHS_ENABLE_PARDISO TRUE CACHE INTERNAL "" FORCE)

        # SDL packages expose MKL::mkl_rt; use it only for optional DLL copying.
        # Only Windows executables need runtime DLLs copied next to them.
        if(WIN32 AND TARGET MKL::mkl_rt)
            get_target_property(_mkl_rt_loc MKL::mkl_rt LOCATION)
            get_filename_component(MKL_BIN_DIR "${_mkl_rt_loc}" DIRECTORY)
        elseif(WIN32)
            message(WARNING
                "oneMKL was found without MKL::mkl_rt; runtime DLLs will not "
                "be copied automatically")
        endif()
    else()
        message(WARNING
            "USE_MKL=ON but oneMKL was not found; Pardiso is disabled "
            "(the AmgCg default still works)")
    endif()
else()
    message(STATUS "USE_MKL=OFF; Pardiso support is disabled")
endif()
