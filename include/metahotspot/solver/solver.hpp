#pragma once

#include "model/model.hpp"
#include "solver/solution.hpp"
#include <optional>
#include <span>

namespace mhs::solver {

    /// Flattened user-facing solver configuration.
    /// Defaults are defined here — C API and Python convert from this struct.
    struct SolveOptions {
        // Linear solver
        enum class LinearSolverType : int { Pardiso, Amg, SparseLu };
        LinearSolverType linear_solver = LinearSolverType::Amg;
        double linear_tolerance = 1e-8;
        int linear_max_iterations = 1000;

        // Non-linear solver
        double underrelaxation = 1.0;
        int nonlinear_max_iterations = 200;
        double nonlinear_relative_tolerance = 1e-6;
        double nonlinear_absolute_tolerance = 1e-12;

        // Adaptive BDF1/2 time integration. Output sampling is set by Study.
        double time_tolerance = 1e-4;
        std::optional<double> max_time_step; // seconds; unset selects automatic
    };

    /// Standard thermal solve entry point.
    ///
    /// Creates uniform initial state when initial_state is empty.
    /// Validates initial_state size when provided.
    /// Internally assembles the thermal system, runs nonlinear iteration
    /// and time integration, and records probes.
    mhs::solver::Solution solve(const mhs::model::CompiledModel& model, const mhs::model::Study& study,
        std::span<const double> initial_state = {}, const SolveOptions& options = {});

    inline Solution solve(const mhs::model::CompiledModel& model) { return solve(model, {}); }

} // namespace mhs::solver
