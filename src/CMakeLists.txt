add_subdirectory(io)
add_subdirectory(expr)
add_subdirectory(common)
add_subdirectory(assembler)
add_subdirectory(linear_solver)
add_subdirectory(preprocessor)

mhs_add_library(mhs_lib
    SOURCES
    nonlinear/nonlinear_solver.cpp
    scheduler/scheduler.cpp
    scheduler/probe_recorder.cpp
    postprocessor/postprocessor.cpp
    postprocessor/sample_point.cpp
    time_scheme/time_scheme.cpp
    PUBLIC_LINKS
    preprocessor_lib
    expr_lib
    common_lib
    io_lib
    linear_solver_lib
    assembler_lib
)

if(VERBOSE)
    target_compile_definitions(mhs_lib PUBLIC VERBOSE)
endif()
