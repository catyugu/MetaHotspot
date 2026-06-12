include(${CMAKE_CURRENT_LIST_DIR}/CPM.cmake)

# Eigen - header-only from GitHub mirror
CPMAddPackage(
    GITLAB_REPOSITORY libeigen/eigen
    GIT_TAG 5.0.0
    OPTIONS
    "EIGEN_BUILD_DOC OFF"
    "EIGEN_BUILD_TESTING OFF"
    "EIGEN_BUILD_PKGCONFIG OFF"
)

# spdlog
CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY gabime/spdlog
    GIT_TAG v1.17.0
    OPTIONS
    "BUILD_TESTING OFF"
)

# exprtk - header-only, local copy in external/exprtk/exprtk.hpp (no CPM needed)

# tinyxml2
CPMAddPackage(
    NAME tinyxml2
    GITHUB_REPOSITORY leethomason/tinyxml2
    GIT_TAG 11.0.0
    OPTIONS
    "BUILD_TESTING OFF"
)

CPMAddPackage(
    NAME googletest
    GITHUB_REPOSITORY google/googletest
    GIT_TAG "d72f9c8"
    OPTIONS
    "BUILD_GMOCK OFF"
    "INSTALL_GTEST OFF"
)

# oneTBB - parallel assembly
CPMAddPackage(
    NAME TBB
    GITHUB_REPOSITORY "oneapi-src/oneTBB"
    GIT_TAG "v2023.0.0"
    OPTIONS
    "BUILD_SHARED_LIBS ON"
    "TBB_STRICT OFF"
    "TBBMALLOC_BUILD OFF"
    "TBB_TEST OFF"
    "TBB_EXAMPLES OFF"
)

# oneTBB 2023.0.0 needs two MinGW compatibility settings in Debug builds.
# Remove this block when the pinned upstream package no longer needs them.
if(MINGW AND TARGET tbb)
    # MinGW headers expose LOAD_LIBRARY_SAFE_CURRENT_DIRS only for Windows 10.
    # Profiling is disabled because this oneTBB release mixes narrow and wide
    # synchronization names on MinGW.
    target_compile_definitions(tbb
        PUBLIC TBB_USE_PROFILING_TOOLS=0
        PRIVATE _WIN32_WINNT=0x0A00 WINVER=0x0A00
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # Clang 20 with MSYS2 libstdc++ 15 rejects oneTBB's explicit undefine.
        get_target_property(_tbb_compile_options tbb COMPILE_OPTIONS)
        if(_tbb_compile_options)
            list(REMOVE_ITEM _tbb_compile_options "-U__STRICT_ANSI__")
            set_property(TARGET tbb PROPERTY COMPILE_OPTIONS "${_tbb_compile_options}")
        endif()
    endif()
endif()


# Intel oneMKL provides MKL::MKL. Pardiso is optional: when oneMKL is absent,
# the existing solver factory falls back to Eigen SparseLU.
set(MHS_ENABLE_PARDISO FALSE CACHE INTERNAL "" FORCE)

if(USE_MKL)
    set(MKL_LINK "sdl" CACHE STRING "MKL link type (sdl|static|dynamic)")
    set(MKL_THREADING "intel_thread" CACHE STRING "MKL threading runtime")
    set(MKL_INTERFACE "lp64" CACHE STRING "MKL index interface (lp64|ilp64)")
    find_package(MKL QUIET)

    if(TARGET MKL::MKL)
        set(MHS_ENABLE_PARDISO TRUE CACHE INTERNAL "" FORCE)

        # SDL packages expose MKL::mkl_rt; use it only for optional DLL copying.
        if(TARGET MKL::mkl_rt)
            get_target_property(_mkl_rt_loc MKL::mkl_rt LOCATION)
            get_filename_component(MKL_BIN_DIR "${_mkl_rt_loc}" DIRECTORY)
        else()
            message(WARNING
                "oneMKL was found without MKL::mkl_rt; runtime DLLs will not "
                "be copied automatically")
        endif()
    else()
        message(WARNING
            "USE_MKL=ON but oneMKL was not found; disabling Pardiso and "
            "falling back to Eigen SparseLU")
    endif()
else()
    message(STATUS "USE_MKL=OFF; Pardiso support is disabled")
endif()
