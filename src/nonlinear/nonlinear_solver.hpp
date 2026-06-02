#pragma once

#include "model/internal_model.hpp"
#include "solver/solver.hpp"

namespace mhs::nonlinear {

    struct NonLinearResult {
        bool converged = false;
        int iterations = 0;
    };

    NonLinearResult solve(const model::InternalModel& model,
        model::GlobalState& state,
        Solver& solver,
        double underrelaxation,
        int max_iterations,
        double tolerance);

} // namespace mhs::nonlinear