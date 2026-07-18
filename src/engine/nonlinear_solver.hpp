#pragma once

#include "engine/linear_system.hpp"
#include "numerics/linear/linear_solver.hpp"

#include <functional>

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

    /// Per-iteration linear-system factory. Receives the current temperature
    /// iterate as a writable Eigen::Ref so callers can pass a Map alias of an
    /// existing std::vector without copying.
    using LinearSystemProvider = std::function<LinearSystem(Eigen::Ref<const Eigen::VectorXd>)>;

    NonLinearResult nonlinear_solve(LinearSystemProvider ls_provider, Eigen::Ref<Eigen::VectorXd> T,
        LinearSolver& solver, const NonLinearConfig& cfg = {});

} // namespace mhs::sim
