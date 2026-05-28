include(${CMAKE_CURRENT_LIST_DIR}/CPM.cmake)

# Eigen - header-only from GitHub mirror
CPMAddPackage(
    NAME Eigen
    GIT_TAG 3.4
    GITHUB_REPOSITORY PX4/eigen
    OPTIONS "EIGEN_BUILD_TESTING OFF"
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
