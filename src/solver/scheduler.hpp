#pragma once

#include "solver/nonlinear_solver.hpp"
#include "runtime/model.hpp"
#include "runtime/solution.hpp"

namespace mhs::sim {

    struct SolveOptions {
        SolverSpec solver;
        NonLinearConfig nonlinear;
    };

    /// Solve a steady or transient thermal model.
    mhs::core::Solution solve(const mhs::core::Model& model, const SolveOptions& options = {});

} // namespace mhs::sim
