add_library(mhs_solver STATIC
    assembler.cpp
    project_output.cpp
    project_assembler.cpp
    nonlinear_solver.cpp
    solve.cpp
    thermal_reconstruction.cpp
    thermal_physics.cpp
)
target_link_libraries(mhs_solver
    PUBLIC mhs_model
    PRIVATE
    mhs_compiler
    mhs_expression
    mhs_linear
    $<BUILD_INTERFACE:Eigen3::Eigen>
    $<BUILD_INTERFACE:AdaptiveTimeStepper::AdaptiveTimeStepper>
    mhs_logging
    $<BUILD_INTERFACE:mhs_options>
)
