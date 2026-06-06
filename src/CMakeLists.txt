add_subdirectory(io)
add_subdirectory(expr)
add_subdirectory(assembler)
add_subdirectory(common)
add_subdirectory(linear_solver)
add_subdirectory(preprocessor)

add_library(mhs_lib
    io/io.cpp
    nonlinear/nonlinear_solver.cpp
    scheduler/scheduler.cpp
    postprocessor/postprocessor.cpp
)

target_link_libraries(mhs_lib PUBLIC preprocessor_lib expr_lib common_lib io_lib linear_solver_lib assembler_lib)

if(VERBOSE)
    target_compile_definitions(mhs_lib PUBLIC VERBOSE)
endif()

target_include_directories(mhs_lib
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)
