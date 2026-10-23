#pragma once

#include "core/discrete_mesh.hpp"

#include <variant>
#include <vector>

namespace mhs::core {

    // Nonnegative frozen mass flow rates (kg/s), classified by actual flow
    // direction after the pressure solve, including reversed pressure boundaries.
    struct FluidInflow {
        double mass_flow_rate;
        double temperature;
    };

    struct FluidOutflow {
        double mass_flow_rate;
    };

    using FluidBoundaryCondition = std::variant<FluidInflow, FluidOutflow>;

    struct FluidBoundaryTerm {
        FaceId face;
        FluidBoundaryCondition condition;
    };

    struct FluidDomain {
        // Runtime topology. fluid_to_global enables fluid-only assembly;
        // global_to_fluid also acts as the fluid membership map.
        std::vector<Index> fluid_to_global;
        std::vector<Index> global_to_fluid;

        // Frozen hydraulic volume flux by FaceId, positive out of the face owner.
        // Non-fluid and exterior faces carry zero.
        std::vector<double> face_volume_flux;

        // Nu / hydraulic_diameter for each fluid cell. The thermal assembly
        // evaluates h = factor * k_fluid at the current temperature.
        std::vector<double> interface_heat_transfer_factor;

        // Frozen transport per projected boundary piece; only inflow carries temperature.
        std::vector<FluidBoundaryTerm> boundary_terms;
    };

} // namespace mhs::core
