#pragma once

#include "model/internal_model.hpp"
#include "model/types.hpp"
#include "solver/solver.hpp"

namespace mhs::nonlinear {

    struct NewtonResult {
        bool converged = false;
        int iterations = 0;
    };

    NewtonResult solve(const model::InternalModel& model,
                       model::GlobalState& state,
                       Solver& solver,
                       double underrelaxation,
                       int max_iterations,
                       double tolerance);

} // namespace mhs::nonlinear