#pragma once

#include "model/internal_model.hpp"
#include "model/types.hpp"
#include "solver/solver.hpp"

namespace mhs::nonlinear {

    struct SolveResult {
        ConvergenceStatus status = ConvergenceStatus::Running;
        int iterations = 0;
    };

    SolveResult solve(const model::InternalModel& model,
                      model::GlobalState& state,
                      Solver& solver,
                      double underrelaxation,
                      int max_iterations,
                      double tolerance);

} // namespace mhs::nonlinear