#pragma once

#include "linear/solver.hpp"
#include "solver/linear_system.hpp"

#include "core/vector.hpp"
#include <functional>
#include <span>

namespace mhs::solver {

    struct NonLinearConfig {
        double underrelaxation = 1.0;
        int max_iterations = 200;
        double relative_tolerance = 1e-6;
        double absolute_tolerance = 1e-12;
    };

    /// Refresh caller-owned storage at the current state; the view must remain valid
    /// until the next provider call.
    using LinearSystemProvider = std::function<LinearSystemView(std::span<const double>)>;

    bool nonlinear_solve(LinearSystemProvider ls_provider, mhs::core::Vector<double>& state,
        mhs::linear::SolverPtr& solver, const NonLinearConfig& cfg = {});

} // namespace mhs::solver
