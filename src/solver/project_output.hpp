#pragma once

#include "model/project.hpp"
#include "solver/solution.hpp"

namespace mhs::solver {
    core::Vector<OutputTrace> project_outputs(const model::Project& project);
    void sample_outputs(
        const model::Project& project, core::Vector<OutputTrace>& traces, std::span<const double> state, double time);
} // namespace mhs::solver
