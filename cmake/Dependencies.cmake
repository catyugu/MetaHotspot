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

# muparser - mathematical expression parser (https://github.com/beltoforion/muparser)
CPMAddPackage(
    NAME muparser
    GITHUB_REPOSITORY beltoforion/muparser
    GIT_TAG v2.3.5
    OPTIONS
    "ENABLE_SAMPLES OFF"
    "ENABLE_OPENMP OFF"
    "BUILD_SHARED_LIBS OFF"
    "BUILD_TESTING OFF"
)
# muparser's GLOB_RECURSE picks up its own muParserTest.cpp which fails under
# our -Werror / -Wunused-parameter. Relax warnings for that target only.
if(TARGET muparser)
    target_compile_options(muparser PRIVATE -Wno-unused-parameter -Wno-error)
endif()

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
    "TBB_TEST OFF"
    "TBB_EXAMPLES OFF"
)

# Intel oneMKL — provides MKL::MKL INTERFACE IMPORTED target with include
# paths, libraries, and runtime DLLs. Required only when USE_MKL=ON.
if(USE_MKL)
    set(MKL_LINK "sdl" CACHE STRING "MKL link type (sdl|static|dynamic)")
    set(MKL_THREADING "intel_thread" CACHE STRING "MKL threading runtime")
    set(MKL_INTERFACE "lp64" CACHE STRING "MKL index interface (lp64|ilp64)")
    find_package(MKL REQUIRED)
    set(MHS_ENABLE_PARDISO TRUE CACHE INTERNAL "" FORCE)
    # Cache the bin dir so subdirectories can replicate MKL runtime DLLs.
    get_target_property(_mkl_rt_loc MKL::mkl_rt LOCATION)
    get_filename_component(MKL_BIN_DIR "${_mkl_rt_loc}" DIRECTORY)
else()
    message(WARNING "MKL not found, setting MHS_ENABLE_PARDISO to FALSE")
    set(MHS_ENABLE_PARDISO FALSE CACHE INTERNAL "" FORCE)
endif()
