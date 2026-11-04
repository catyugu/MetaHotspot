add_library(mhs_solver STATIC
    assembler.cpp
    nonlinear_solver.cpp
    postprocessor.cpp
    probe_recorder.cpp
    solve.cpp
    thermal_reconstruction.cpp
)
target_link_libraries(mhs_solver
    PUBLIC mhs_model
    PRIVATE
    mhs_expression
    mhs_linear
    $<BUILD_INTERFACE:Eigen3::Eigen>
    mhs_logging
    TBB::tbb
    $<BUILD_INTERFACE:mhs_options>
)


# Header-only implementation dependency; keep it out of the installed SDK graph.
target_include_directories(mhs_solver SYSTEM PRIVATE "${AdaptiveTimeStepper_SOURCE_DIR}/include")
