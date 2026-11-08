#pragma once
#include "model/model.hpp"
#include <span>

namespace mhs::solver {
    struct WeightedDof {
        model::DofId dof = core::invalid_id;
        double weight = 0.0;
    };

    struct ConnectionFlux {
        double conductance = 0.0;
        double correction = 0.0; // Deferred NCC correction, W
        double bias_flux = 0.0;  // From affine ROM trace offsets, W
        double outward_a = 0.0;
        double interface_temperature = 0.0;
        // Trial and conservative test maps for both endpoints.
        core::Vector<WeightedDof> a, b;
    };

    // Works for FOM-FOM, FOM-ROM, and ROM-ROM; no ROM interior mesh is needed.
    core::Vector<ConnectionFlux> evaluate_connection_flux(const model::CompiledModel& compiled,
        const model::ThermalConnection& connection, std::span<const double> state, double time);
} // namespace mhs::solver
