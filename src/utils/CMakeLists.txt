mhs_add_library(utils_lib
    SOURCES sampling.cpp face_key.cpp
    PRIVATE_LINKS Eigen3::Eigen
)
