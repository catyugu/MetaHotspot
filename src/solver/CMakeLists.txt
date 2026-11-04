add_library(mhs_solver STATIC
    assembler.cpp
    nonlinear_solver.cpp
    postprocessor.cpp
    probe_recorder.cpp
    solve.cpp
    thermal_reconstruction.cpp
    time_integration.cpp
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
