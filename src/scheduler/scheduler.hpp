#pragma once

#include "data/model.hpp"
#include "data/solution.hpp"
#include "nonlinear/nonlinear_solver.hpp"

namespace mhs::sim {

    struct SolveOptions {
        SolverSpec solver;
        NonLinearConfig nonlinear;
    };

    /// Solve a steady or transient thermal model.
    mhs::core::Solution solve(const mhs::core::Model& model, const SolveOptions& options = {});

} // namespace mhs::sim
