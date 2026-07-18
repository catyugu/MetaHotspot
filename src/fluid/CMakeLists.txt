mhs_add_library(fluid_lib
    SOURCES fluid_preprocessor.cpp fluid_assembler.cpp
    PUBLIC_LINKS
    mhs_model_lib
    Eigen3::Eigen
    PRIVATE_LINKS
    linear_solver_lib
    logger_lib
    utils_lib
    TBB::tbb
)
