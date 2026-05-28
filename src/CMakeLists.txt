add_subdirectory(io)
add_subdirectory(expr)
add_subdirectory(logger)

add_library(mhs_lib
    io/io.cpp
    preprocessor/preprocessor.cpp
    assembler/assembler.cpp
    scheduler/scheduler.cpp
    solver/solver.cpp
    postprocessor/postprocessor.cpp
)

target_link_libraries(mhs_lib PUBLIC Eigen3::Eigen expr_lib logger_lib tinyxml2)

target_include_directories(mhs_lib
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/_deps/tinyxml2-src>
    $<INSTALL_INTERFACE:include>
)
