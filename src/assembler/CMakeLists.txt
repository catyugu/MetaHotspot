mhs_add_library(assembler_lib
    SOURCES assembler.cpp
    PRIVATE_LINKS fluid_lib TBB::tbb utils_lib Eigen3::Eigen
)
