#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace mhs::core {

    struct ProbeTrace {
        std::string name;
        std::vector<double> times;
        std::vector<double> values;
    };

    /// Canonical solution returned by any solve path.
    /// Temperature is stored directly in state, indexed by CellId.
    /// Output snapshots are stored row-major in snapshot_states:
    /// snapshot_states[record * state_count + state].
    struct Solution {
        std::vector<double> state;
        double time = 0.0;
        bool converged = true;
        std::vector<ProbeTrace> probe_traces;
        std::vector<double> snapshot_times;
        std::vector<double> snapshot_states;
    };

} // namespace mhs::core
