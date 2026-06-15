mhs_add_library(assembler_lib
    SOURCES assembler.cpp
    PUBLIC_LINKS Eigen3::Eigen
    PRIVATE_LINKS TBB::tbb common_lib
)
