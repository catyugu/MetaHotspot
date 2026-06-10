#pragma once

#include "common/internal_model.hpp"
#include "linear_solver/linear_solver.hpp"

namespace mhs::sim {

    struct NonLinearResult {
        bool converged = false;
        int iterations = 0;
    };

    struct NonLinearConfig {
        double underrelaxation = 1.0;
        int max_iterations = 50;
        double relative_tolerance = 1e-6;
        double absolute_tolerance = 1e-12;
    };

    // Anderson-accelerated fixed-point iteration over `LinearSolver`.
    // Renamed from `solve` to `nonlinear_solve` so that `mhs::sim::nonlinear_solve`
    // is unambiguous in the flat sim domain.
    NonLinearResult nonlinear_solve(const core::InternalModel& model, core::GlobalState& state, LinearSolver& solver);

} // namespace mhs::sim