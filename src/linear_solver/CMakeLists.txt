set(_linear_solver_sources
    linear_solver.cpp
    bicgstab_solver.cpp
    sparse_lu_solver.cpp
)

if(MHS_ENABLE_PARDISO)
    list(APPEND _linear_solver_sources pardiso_solver.cpp)
endif()

mhs_add_library(linear_solver_lib
    SOURCES ${_linear_solver_sources}
    PUBLIC_LINKS Eigen3::Eigen
)

if(MHS_ENABLE_PARDISO)
    target_link_libraries(linear_solver_lib PUBLIC MKL::MKL)
    target_compile_definitions(linear_solver_lib PUBLIC MHS_ENABLE_PARDISO)
endif()
