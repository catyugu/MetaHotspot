add_library(solver_lib STATIC
    solver.cpp
    bicgstab_solver.cpp
    sparse_lu_solver.cpp
)

if(MHS_HAS_PARDISO)
    target_sources(solver_lib PRIVATE pardiso_solver.cpp)
    target_link_libraries(solver_lib PUBLIC MKL::MKL)
    target_compile_definitions(solver_lib PUBLIC MHS_ENABLE_PARDISO)
endif()

target_link_libraries(solver_lib PUBLIC Eigen3::Eigen)
target_include_directories(solver_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external>
)
