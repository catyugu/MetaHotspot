#pragma once

#include "engine/runtime_model.hpp"
#include "model/model_definition.hpp"

#include <optional>
#include <vector>

namespace mhs::sim::fluid {

    struct FluidMaterialData {
        std::vector<std::optional<double>> initial_viscosity;
    };

    // Build the assembly-ready frozen flow domain. Geometry, pressure-system,
    // and boundary-parameter scratch data remain private to the implementation.
    void build_domain(mhs::core::Model& model, const std::vector<mhs::model::FluidBoundarySpec>& boundaries,
        double si_scale,
        const FluidMaterialData& materials);

} // namespace mhs::sim::fluid
