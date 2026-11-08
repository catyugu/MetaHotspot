#pragma once
#include "model/model.hpp"
#include <span>

namespace mhs::solver {
    struct ConnectionFlux {
        double conductance = 0.0;
        double correction = 0.0; // W, outward from side A; added to the two-point flux
        double outward_a = 0.0;
        double interface_temperature = 0.0;
    };
    core::Vector<ConnectionFlux> evaluate_connection_flux(const model::CompiledModel& compiled,
        const model::ThermalConnection& connection, std::span<const double> state, double time);
    // Uses current cell properties and tangential reconstructions. Same kernel for
    // assembly and flux inspection, independent of the nonlinear iteration history.
    core::Vector<ConnectionFlux> evaluate_connection_flux(const model::CompiledModel& compiled,
        const model::ThermalConnection& connection, const core::Vector<core::Vector<double>>& temperatures,
        double time);
}
