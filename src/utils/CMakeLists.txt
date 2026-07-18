mhs_add_library(utils_lib
    SOURCES sampling.cpp
    PUBLIC_LINKS mhs_model_lib
    PRIVATE_LINKS Eigen3::Eigen
)
