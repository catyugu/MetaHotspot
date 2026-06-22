mhs_add_library(assembler_lib
    SOURCES assembler.cpp
    PRIVATE_LINKS TBB::tbb common_lib Eigen3::Eigen
)
