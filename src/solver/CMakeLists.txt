set(mhs_solver_sources
    assembler.cpp
    fluid_assembler.cpp
    interpolation.cpp
    nonlinear_solver.cpp
    postprocessor.cpp
    probe_recorder.cpp
    solve.cpp
    time_integration.cpp
)

add_library(mhs_solver STATIC ${mhs_solver_sources})
target_include_directories(mhs_solver PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(mhs_solver
    PUBLIC
    mhs_core
    mhs_linear
    PRIVATE
    mhs_logging
    TBB::tbb
    $<BUILD_INTERFACE:mhs_options>
)
mhs_install_headers(solver postprocessor.hpp)
