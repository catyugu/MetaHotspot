#pragma once

/* Internal solve driver — NOT part of the public API.  Shared between the
   standard solver (solve.cpp) and the callback-driven solve_system. */

#include "core/vector.hpp"
#include "solver/linear_system.hpp"
#include "solver/solution.hpp"
#include "solver/solver.hpp"

#include <functional>
#include <span>

namespace mhs::solver {

    // Refresh caller-owned K/C/f storage. References remain valid until the next
    // call; structural support must remain inside the initial K/C union.
    using SystemAssembler = std::function<const Operators&(std::span<const double> state, double time)>;

    using StateObserver = std::function<void(double time, std::span<const double> state)>;

    /// Generalized callback-driven solver. Standard solve() calls this
    /// internally, transferring ownership of its state buffer.
    mhs::solver::Solution solve_system(const StudyOptions& study, const SystemAssembler& assemble,
        core::Vector<double> state, const SolveOptions& opts = {}, const StateObserver& observe = {});

} // namespace mhs::solver
