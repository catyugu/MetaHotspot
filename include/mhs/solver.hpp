#pragma once

#include "mhs/model.hpp"
#include "mhs/solution.hpp"
#include <span>

namespace mhs::sim {

    /// Flattened user-facing solver configuration.
    /// Defaults are defined here — C API and Python convert from this struct.
    struct SolveOptions {
        // Linear solver
        enum class LinearSolverType : int { Pardiso, EigenSparseLU, EigenBiCGSTAB };
        LinearSolverType linear_solver = LinearSolverType::Pardiso;
        double linear_tolerance = 1e-8;
        int linear_max_iterations = 1000;

        // Non-linear solver
        double underrelaxation = 1.0;
        int nonlinear_max_iterations = 200;
        double nonlinear_relative_tolerance = 1e-6;
        double nonlinear_absolute_tolerance = 1e-12;

        // Time integration
        enum class Integrator : int { Bdf1, Bdf2 };
        Integrator integrator = Integrator::Bdf1;

        enum class StepStrategy : int { Adaptive, Fixed };
        StepStrategy step_strategy = StepStrategy::Adaptive;

        double error_abs_tol = 1e-4;
        double error_safety = 0.9;
        double min_dt = 1e-12;
        double max_dt = 1.0;
        double fixed_dt = 1.0;
    };

    /// Standard thermal solve entry point.
    ///
    /// Creates uniform initial state when initial_state is empty.
    /// Validates initial_state size when provided.
    /// Internally assembles the thermal system, runs nonlinear iteration
    /// and time integration, and records probes.
    mhs::core::Solution solve(
        const mhs::core::Model& model, std::span<const double> initial_state = {}, const SolveOptions& options = {});

} // namespace mhs::sim
