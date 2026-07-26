#pragma once

#include "runtime/model.hpp"
#include "runtime/solution.hpp"
#include "solver/nonlinear_solver.hpp"
#include "solver/time_integration.hpp"

namespace mhs::sim {

    struct SolveOptions {
        SolverSpec solver;
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
    ///
    /// Reuses the supplied *assembler* and *solver* (avoids re-allocating internal
    /// data across repeated calls).  The *history* ring buffer must already hold
    /// the accepted solution at *current_time* before the first call.
    ///
    /// On return *history* is updated if the step was accepted.
    StepResult take_step(Assembler& assembler, LinearSolver& solver, mhs::core::SolutionHistory& history,
        std::vector<double>& state, double current_time, double dt, const NonLinearConfig& nonlinear_cfg = {},
        time_scheme::IntegratorKind integrator = time_scheme::IntegratorKind::Bdf1);

    /// Solve a steady or transient thermal model.
    mhs::core::Solution solve(const mhs::core::Model& model, const SolveOptions& options = {});

} // namespace mhs::sim
