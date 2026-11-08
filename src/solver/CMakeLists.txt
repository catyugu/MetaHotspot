add_library(mhs_solver STATIC
    assembler.cpp
    connection_flux.cpp
    frozen_flow.cpp
    stokes_flow.cpp
    nonlinear_solver.cpp
    solve.cpp
    thermal_reconstruction.cpp
)
target_link_libraries(mhs_solver
    PUBLIC mhs_model
    PRIVATE
    mhs_expression
    mhs_linear
    $<BUILD_INTERFACE:Eigen3::Eigen>
    $<BUILD_INTERFACE:AdaptiveTimeStepper::AdaptiveTimeStepper>
    mhs_logging
    TBB::tbb
    $<BUILD_INTERFACE:mhs_options>
)
