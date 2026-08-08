#pragma once

/* Internal solve driver — NOT part of the public API.  Shared between the
   standard solver (solve.cpp) and the callback-driven solve_system. */

#include "common/solution.hpp"
#include "common/solver.hpp"
#include "solver/assembler.hpp"

#include <functional>
#include <span>

namespace mhs::sim {

    using SystemAssembler = std::function<Operators(std::span<const double> state, double time)>;

    using StateObserver = std::function<void(double time, std::span<const double> state)>;

    struct Study {
        mhs::core::StudyType type = mhs::core::StudyType::Steady;
        double duration = 0.0;
        double output_interval = 1.0;
    };

    /// Generalized callback-driven solver. Standard solve() calls this
    /// internally.
    mhs::core::Solution solve_system(const Study& study, const SystemAssembler& assemble,
        std::span<const double> initial_state, const SolveOptions& opts = {}, const StateObserver& observe = {});

} // namespace mhs::sim
