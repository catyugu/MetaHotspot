#pragma once

#include "model/discrete_mesh.hpp"

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
        FaceId face = mhs::core::invalid_id;
        FluidTransportCondition condition;
    };

    struct FluidModel {
        // Runtime topology. fluid_to_global enables fluid-only assembly;
        // global_to_fluid also acts as the fluid membership map.
        mhs::core::Vector<CellId> fluid_to_global;
        mhs::core::Vector<FluidCellId> global_to_fluid;

        // Frozen hydraulic volume flux by FaceId, positive from minus to plus along the face axis.
        // Non-fluid and exterior faces carry zero.
        mhs::core::Vector<double> face_volume_flux;

        // Nu / hydraulic_diameter for each fluid cell. The thermal assembly
        // evaluates h = factor * k_fluid at the current temperature.
        mhs::core::Vector<double> interface_heat_transfer_factor;

        // Frozen transport per projected boundary piece; only inflow carries temperature.
        mhs::core::Vector<FluidBoundaryTerm> boundary_terms;
    };

} // namespace mhs::model
