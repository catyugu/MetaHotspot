#pragma once

/* Internal solve driver — NOT part of the public API.  Shared between the
   standard solver (solve.cpp) and the callback-driven solve_system. */

#include "solver/assembler.hpp"
#include "solver/solution.hpp"
#include "solver/solver.hpp"

#include <functional>
#include <span>

namespace mhs::solver {

    using SystemAssembler = std::function<Operators(std::span<const double> state, double time)>;

    using StateObserver = std::function<void(double time, std::span<const double> state)>;

    struct Study {
        mhs::model::StudyType type = mhs::model::StudyType::Steady;
        double duration = 0.0;
        double output_interval = 1.0;
    };

    /// Generalized callback-driven solver. Standard solve() calls this
    /// internally.
    mhs::solver::Solution solve_system(const Study& study, const SystemAssembler& assemble,
        std::span<const double> initial_state, const SolveOptions& opts = {}, const StateObserver& observe = {});

} // namespace mhs::solver
