#pragma once

#include "core/vector.hpp"
#include <string>

namespace mhs::solver {

    // Values follow Solution::snapshot_times; all probes share that timeline.
    struct ProbeTrace {
        std::string name;
        mhs::core::Vector<double> values;
    };

    /// Canonical solution returned by any solve path.
    /// Temperature is stored directly in state, indexed by CellId.
    /// Output snapshots are stored row-major in snapshot_states:
    /// snapshot_states[record * state_count + state].
    struct Solution {
        mhs::core::Vector<double> state;
        double time = 0.0;
        bool converged = true;
        mhs::core::Vector<ProbeTrace> probe_traces;
        mhs::core::Vector<double> snapshot_times;
        mhs::core::Vector<double> snapshot_states;
    };

} // namespace mhs::solver
