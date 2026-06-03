#pragma once

#include "common/internal_model.hpp"
#include "solver/solver.hpp"

namespace mhs::nonlinear {

    struct NonLinearResult {
        bool converged = false;
        int iterations = 0;
    };

    struct NonLinearConfig {
        double underrelaxation = 1.0;
        int max_iterations = 50;
        double tolerance = 1e-6;
    };

    NonLinearResult solve(const InternalModel& model,
        GlobalState& state,
        Solver& solver,
        const NonLinearConfig& cfg);

} // namespace mhs::nonlinear