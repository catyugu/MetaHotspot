#pragma once

#include "runtime/model.hpp"
#include "runtime/solution.hpp"
#include "solver/nonlinear_solver.hpp"
#include "solver/time_integration.hpp"

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

    /// Result of a single transient step.
    struct StepResult {
        bool accepted = false;
        double error_ratio = 0.0;
        double suggested_dt_factor = 1.0;
        int nonlinear_iterations = 0;
        bool nonlinear_converged = true;
    };

    /// Execute a single transient step from *current_time* with step *dt*.
    StepResult take_step(Assembler& assembler, LinearSolver& solver, mhs::core::SolutionHistory& history,
        std::vector<double>& state, double current_time, double dt, const SolverOpts& opts);

    /// Solve a steady or transient thermal model.
    mhs::core::Solution solve(
        const mhs::core::Model& model, const SolverOpts& opts = {}, std::span<const double> initial_state = {});

} // namespace mhs::sim
