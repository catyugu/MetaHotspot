#pragma once

#include "runtime/types.hpp"
#include <string>
#include <vector>

namespace mhs::core {

    /// Sub-range of a combined state vector.
    struct DofRange {
        Index begin = 0;
        Index count = 0;
    };

    /// Describes how the state vector is partitioned.
    struct StateLayout {
        DofRange temperature;   // thermal DoF slice
        Index state_count = 0;  // total DoFs across all groups
    };

    struct ProbeTrace {
        std::string name;
        std::vector<double> times;
        std::vector<double> values;
    };

    struct Solution {
        std::vector<double> state;
        StateLayout layout;
        double time = 0.0;
        std::vector<ProbeTrace> probe_traces;
        bool converged = true;
    };

} // namespace mhs::core
