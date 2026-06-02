add_library(solver_lib STATIC solver.cpp)
target_link_libraries(solver_lib PUBLIC Eigen3::Eigen)
target_include_directories(solver_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external>
)
