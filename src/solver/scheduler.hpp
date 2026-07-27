#pragma once

#include "runtime/model.hpp"
#include "runtime/solution.hpp"
#include "solver/assembler.hpp"
#include "solver/nonlinear_solver.hpp"
#include "solver/time_integration.hpp"

#include <functional>
#include <span>

namespace mhs::sim {

    struct SolverOpts {
        // Time integration
        time_scheme::IntegratorKind integrator = time_scheme::IntegratorKind::Bdf1;
        time_scheme::StepStrategy step_strategy = time_scheme::StepStrategy::AdaptiveFree;

        // Error control
        double error_abs_tol = 1e-4;
        double error_safety = 0.9;

        // Step bounds
        double min_dt = 1e-12;
        double max_dt = 1.0;
        double fixed_dt = 1.0;

        // Linear solver
        SolverSpec solver;

        // Non-linear solver
        NonLinearConfig nonlinear;
    };

    /// Output callback invoked at each flush/output time during solve_system.
    using OutputCallback = std::function<void(double time, std::span<const double> state)>;

    /// Solve a generic system described by an Assemble callback.
    /// The state vector may carry pure-thermal or combined (extra-DoF) variables.
    mhs::core::SolveResult solve_system(Assemble provider, std::span<const double> initial_state,
        mhs::core::StudyType study_type, double transient_duration, double transient_time_step,
        const SolverOpts& opts = {}, OutputCallback on_output = nullptr);

    /// Thermal convenience wrapper around solve_system.
    /// Returns temperature field + probe traces.
    mhs::core::ThermalSolution solve_thermal(
        const mhs::core::Model& model, const SolverOpts& opts = {}, std::span<const double> initial_state = {});

} // namespace mhs::sim
