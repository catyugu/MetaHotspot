#pragma once

#include "core/vector.hpp"
#include "model/surface.hpp"
#include "model/types.hpp"

namespace mhs::model {

    struct FluidBoundaryTerm {
        CellId cell;
        double mass_flow_rate; // kg/s, negative for inflow
        double inlet_temperature; // used for inflow only
    };

    struct FluidModel {
        // Runtime topology. fluid_to_global enables fluid-only assembly;
        // global_to_fluid also acts as the fluid membership map.
        mhs::core::Vector<CellId> fluid_to_global;
        mhs::core::Vector<FluidCellId> global_to_fluid;

        // Frozen volume flux along X+, Y+, Z+ for each fluid cell.
        // A missing fluid neighbor carries zero flux.
        mhs::core::Vector<Vec3> positive_volume_flux;

        // Nu / hydraulic_diameter for each fluid cell. The thermal assembly
        // evaluates h = factor * k_fluid at the current temperature.
        mhs::core::Vector<double> interface_heat_transfer_factor;

        // Frozen transport contributions from exterior facets.
        mhs::core::Vector<FluidBoundaryTerm> boundary_terms;
    };

} // namespace mhs::model
