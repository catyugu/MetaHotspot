#pragma once

#include "core/vector.hpp"
#include "model/types.hpp"
#include <string>

namespace mhs::solver {

    // Values follow Solution::snapshot_times; all outputs share that timeline.
    struct OutputTrace {
        std::string name;
        mhs::core::Vector<double> values;
        mhs::model::ComponentId component = mhs::core::invalid_id;
        mhs::model::OutputId output = mhs::core::invalid_id;
    };

    /// Canonical solution returned by any solve path.
    /// Project state contains FOM cell temperatures and EROM modal coordinates.
    /// Output snapshots are stored row-major in snapshot_states:
    /// snapshot_states[record * state_count + state]; empty when state storage is disabled.
    struct Solution {
        mhs::core::Vector<double> state;
        double time = 0.0;
        bool converged = true;
        mhs::core::Vector<OutputTrace> output_traces;
        mhs::core::Vector<double> snapshot_times;
        mhs::core::Vector<double> snapshot_states;

        const OutputTrace* find_output(model::ComponentId component, model::OutputId output) const
        {
            for (const auto& trace : output_traces)
                if (trace.component == component && trace.output == output)
                    return &trace;
            return nullptr;
        }
    };

} // namespace mhs::solver
