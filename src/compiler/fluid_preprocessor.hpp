#pragma once

#include "core/model.hpp"
#include "core/model_definition.hpp"

#include "core/vector.hpp"
#include <optional>

namespace mhs::sim::fluid {

    struct FluidMaterialData {
        mhs::core::Vector<std::optional<double>> initial_viscosity;
    };

    // Build the assembly-ready frozen flow domain. Geometry, pressure-system,
    // and boundary-parameter scratch data remain private to the implementation.
    void build_domain(mhs::core::Model& model, const mhs::core::Vector<mhs::model::FluidBoundarySpec>& boundaries,
        double si_scale, const FluidMaterialData& materials);

} // namespace mhs::sim::fluid
