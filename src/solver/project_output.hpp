#pragma once

#include "model/project.hpp"
#include "solver/solution.hpp"

namespace mhs::solver {
    void append_fom_traces(core::Vector<OutputTrace>& traces, const model::Fom& fom, model::ComponentId component);
    core::Vector<OutputTrace> project_outputs(const model::Project& project);
    void sample_outputs(
        const model::Project& project, core::Vector<OutputTrace>& traces, std::span<const double> state, double time);
} // namespace mhs::solver
