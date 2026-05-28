add_subdirectory(expr)

add_library(mhs_lib
    preprocessor/preprocessor.cpp
    assembler/assembler.cpp
    scheduler/scheduler.cpp
    solver/solver.cpp
    postprocessor/postprocessor.cpp
)

target_link_libraries(mhs_lib PUBLIC Eigen3::Eigen expr_lib)

target_include_directories(mhs_lib
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external>
    $<INSTALL_INTERFACE:include>
)
