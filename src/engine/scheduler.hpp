#pragma once

#include "engine/nonlinear_solver.hpp"
#include "engine/runtime_model.hpp"
#include "engine/solution.hpp"

namespace mhs::sim {

    struct SolveOptions {
        SolverSpec solver;
        NonLinearConfig nonlinear;
    };

    /// Solve a steady or transient thermal model.
    mhs::core::Solution solve(const mhs::core::Model& model, const SolveOptions& options = {});

} // namespace mhs::sim
