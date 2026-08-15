add_library(mhs_expression STATIC expression/expr.cpp)
target_include_directories(mhs_expression PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mhs_expression PRIVATE
    mhs_options
    TBB::tbb
    muparser
)

set(_linear_sources
    linear/linear_solver.cpp
    linear/eigen_bicgstab_solver.cpp
    linear/eigen_sparse_lu_solver.cpp
)

if(MHS_ENABLE_PARDISO)
    list(APPEND _linear_sources linear/pardiso_lu_solver.cpp)
endif()

add_library(mhs_linear STATIC ${_linear_sources})
target_include_directories(mhs_linear PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mhs_linear PUBLIC Eigen3::Eigen)
target_link_libraries(mhs_linear PRIVATE mhs_options)

if(MHS_ENABLE_PARDISO)
    target_link_libraries(mhs_linear PUBLIC MKL::MKL)
    target_compile_definitions(mhs_linear PUBLIC MHS_ENABLE_PARDISO)
endif()
