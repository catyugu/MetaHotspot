#pragma once

#include "data/internal_model.hpp"
#include "data/linear_system.hpp"
#include "linear_solver/linear_solver.hpp"

#include <functional>

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

    // The provider receives the current iteration's GlobalState by const
    // reference; the solver owns the mutable state and is responsible for
    // applying the update each iteration.  Decoupling the data flow this
    // way makes the contract explicit and removes any hidden state
    // captured by the provider closure.
    using LinearSystemProvider = std::function<LinearSystem(const mhs::core::GlobalState&)>;

    NonLinearResult nonlinear_solve(LinearSystemProvider ls_provider, mhs::core::GlobalState& state,
        LinearSolver& solver, const NonLinearConfig& cfg = {});

} // namespace mhs::sim
