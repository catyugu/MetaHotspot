#pragma once

#include "model/mesh.hpp"

#include "core/vector.hpp"
#include <variant>

namespace mhs::model {

    // Nonnegative frozen mass flow rates (kg/s), classified by actual flow
    // direction after the pressure solve, including reversed pressure boundaries.
    struct FluidInflow {
        double mass_flow_rate;
        double temperature;
    };

    struct FluidOutflow {
        double mass_flow_rate;
    };

    using FluidTransportCondition = std::variant<FluidInflow, FluidOutflow>;

    struct FluidBoundaryTerm {
        BoundaryFacetId facet = mhs::core::invalid_id;
        FluidTransportCondition condition;
    };

    struct FluidConnection {
        Connection cells;
        double volume_flux = 0.0; // positive from a to b
    };

    struct FluidModel {
        // Runtime topology. fluid_to_global enables fluid-only assembly;
        // global_to_fluid also acts as the fluid membership map.
        mhs::core::Vector<CellId> fluid_to_global;
        mhs::core::Vector<FluidCellId> global_to_fluid;

        // Only contacts between fluid cells; no derived geometry is retained.
        mhs::core::Vector<FluidConnection> connections;

        // Nu / hydraulic_diameter for each fluid cell. The thermal assembly
        // evaluates h = factor * k_fluid at the current temperature.
        mhs::core::Vector<double> interface_heat_transfer_factor;

        // Frozen transport per exterior facet; only inflow carries temperature.
        mhs::core::Vector<FluidBoundaryTerm> boundary_terms;
    };

} // namespace mhs::model
