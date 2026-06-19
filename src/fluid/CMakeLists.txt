mhs_add_library(fluid_lib
    SOURCES fluid_preprocessor.cpp
    PRIVATE_LINKS linear_solver_lib common_lib
)
