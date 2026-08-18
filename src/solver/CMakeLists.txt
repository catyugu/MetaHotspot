add_library(mhs_solver STATIC
    assembler.cpp
    fluid_assembler.cpp
    interpolation.cpp
    nonlinear_solver.cpp
    postprocessor.cpp
    probe_recorder.cpp
    solve.cpp
    time_integration.cpp
)
target_include_directories(mhs_solver PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mhs_solver
    PUBLIC
    mhs_common
    mhs_linear
    PRIVATE
    mhs_logging
    TBB::tbb
    mhs_options
)
