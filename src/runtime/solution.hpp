#pragma once

#include <string>
#include <vector>

namespace mhs::core {

    struct ProbeTrace {
        std::string name;
        std::vector<double> times;
        std::vector<double> values;
    };

    struct Solution {
        std::vector<double> state;
        std::vector<double> cell_temperature;
        double time = 0.0;
        std::vector<ProbeTrace> probe_traces;
    };

} // namespace mhs::core
