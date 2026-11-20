#pragma once

#include "core/vector.hpp"
#include "model/types.hpp"

namespace mhs::model {

    struct FluidBoundaryTerm {
        FacetId facet = core::invalid_id;
        double mass_flow_rate; // kg/s, positive outward
        double inlet_temperature; // used only for negative mass flow
    };

    struct FluidModel {
        // Runtime topology. fluid_to_global enables fluid-only assembly;
        // global_to_fluid also acts as the fluid membership map.
        mhs::core::Vector<CellId> fluid_to_global;
        mhs::core::Vector<FluidCellId> global_to_fluid;

        // Frozen hydraulic volume flux by FacetId, positive from minus to plus.
        // Non-fluid and exterior facets carry zero.
        mhs::core::Vector<double> facet_volume_flux;

        // Nu / hydraulic_diameter for each fluid cell. The thermal assembly
        // evaluates h = factor * k_fluid at the current temperature.
        mhs::core::Vector<double> interface_heat_transfer_factor;

        // Frozen transport per exterior facet; only inflow carries temperature.
        mhs::core::Vector<FluidBoundaryTerm> boundary_terms;
    };

} // namespace mhs::model
