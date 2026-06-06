add_library(linear_solver_lib STATIC
    linear_solver.cpp
    bicgstab_solver.cpp
    sparse_lu_solver.cpp
)

if(MHS_ENABLE_PARDISO)
    target_sources(linear_solver_lib PRIVATE pardiso_solver.cpp)
    target_link_libraries(linear_solver_lib PUBLIC MKL::MKL)
    target_compile_definitions(linear_solver_lib PUBLIC MHS_ENABLE_PARDISO)
endif()

target_link_libraries(linear_solver_lib PUBLIC Eigen3::Eigen)
target_include_directories(linear_solver_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external>
)
