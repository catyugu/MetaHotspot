mhs_add_library(common_lib
    SOURCES logger.cpp
    PUBLIC_LINKS spdlog
    PRIVATE_LINKS Eigen3::Eigen
)
