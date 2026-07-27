#pragma once

#include <string>
#include <vector>

namespace mhs::core {

    struct ProbeTrace {
        std::string name;
        std::vector<double> times;
        std::vector<double> values;
    };

    /// Full result returned by solve_system() — the entire state vector.
    /// When used with a pure-thermal provider the state equals the temperature
    /// field; with extra DOFs it may include non-thermal variables.
    struct SolveResult {
        std::vector<double> state;
        double time = 0.0;
        bool converged = true;
    };

    /// Thermal-only result returned by solve_thermal() — temperature field
    /// extracted from the full state plus thermal post-processing.
    struct ThermalSolution {
        std::vector<double> temperature;
        double time = 0.0;
        std::vector<ProbeTrace> probe_traces;
        bool converged = true;
    };

} // namespace mhs::core
