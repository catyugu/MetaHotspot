#pragma once

#include "common/internal_model.hpp"
#include "solver/solver.hpp"

namespace mhs::nonlinear {

    struct NonLinearResult {
        bool converged = false;
        int iterations = 0;
    };

    NonLinearResult solve(const InternalModel& model,
        GlobalState& state,
        Solver& solver,
        double underrelaxation,
        int max_iterations,
        double tolerance);

} // namespace mhs::nonlinear