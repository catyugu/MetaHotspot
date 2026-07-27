#pragma once

#include "numerics/linear/linear_solver.hpp"
#include "solver/linear_system.hpp"

#include <functional>
#include <span>
#include <vector>

namespace mhs::sim {

    struct NonLinearResult {
        bool converged = false;
        int iterations = 0;
    };

    struct NonLinearConfig {
        double underrelaxation = 1.0;
        int max_iterations = 200;
        double relative_tolerance = 1e-6;
        double absolute_tolerance = 1e-12;
    };

    /// Per-iteration linear-system factory evaluated at the current state (read-only).
    using LinearSystemProvider = std::function<LinearSystem(std::span<const double>)>;

    NonLinearResult nonlinear_solve(LinearSystemProvider ls_provider, std::vector<double>& state, LinearSolver& solver,
        const NonLinearConfig& cfg = {});

} // namespace mhs::sim
