#pragma once

#include "model/model.hpp"
#include "model/model_definition.hpp"

#include "core/vector.hpp"

namespace mhs::compiler::fluid {

    // Build the assembly-ready frozen flow domain. Geometry, pressure-system,
    // and boundary-parameter scratch data remain private to the implementation.
    void build_domain(mhs::model::Model& model, const mhs::core::Vector<mhs::model::FluidBoundarySpec>& boundaries,
        const mhs::core::Vector<mhs::core::EntityIndex>& facet_boundary_tags,
        const mhs::core::Vector<double>& initial_viscosity);

} // namespace mhs::compiler::fluid
