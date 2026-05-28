include(${CMAKE_CURRENT_LIST_DIR}/CPM.cmake)

# Eigen - header-only from GitHub mirror
CPMAddPackage(
    GITLAB_REPOSITORY libeigen/eigen
    GIT_TAG 5.0.0
    OPTIONS
    "EIGEN_BUILD_DOC OFF"
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
