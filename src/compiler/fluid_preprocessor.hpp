#pragma once

#include "model/model.hpp"
#include "model/model_definition.hpp"

#include "core/vector.hpp"

namespace mhs::compiler::fluid {

    struct FluidMaterialData {
        mhs::core::Vector<double> initial_viscosity;
    };

    // Build the assembly-ready frozen flow domain. Geometry, pressure-system,
    // and boundary-parameter scratch data remain private to the implementation.
    void build_domain(mhs::model::Model& model, const mhs::core::Vector<mhs::model::FluidBoundarySpec>& boundaries,
        const FluidMaterialData& materials);

} // namespace mhs::compiler::fluid
