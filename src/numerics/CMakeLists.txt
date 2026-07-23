mhs_add_library(mhs_expression
    SOURCES expression/expr.cpp
    PRIVATE_LINKS TBB::tbb muparser
)

set(_linear_sources
    linear/linear_solver.cpp
    linear/eigen_bicgstab_solver.cpp
    linear/eigen_sparse_lu_solver.cpp
)

if(MHS_ENABLE_PARDISO)
    list(APPEND _linear_sources linear/pardiso_lu_solver.cpp)
endif()

mhs_add_library(mhs_linear
    SOURCES ${_linear_sources}
    PUBLIC_LINKS Eigen3::Eigen
)

if(MHS_ENABLE_PARDISO)
    target_link_libraries(mhs_linear PUBLIC MKL::MKL)
    target_compile_definitions(mhs_linear PUBLIC MHS_ENABLE_PARDISO)
endif()
