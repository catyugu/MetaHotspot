#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "types.hpp"

namespace mhs::core {

    struct FluidBCParamTable {
        std::vector<double> pressure; // [Pa]   indexed by FluidBCType::PressureType
        std::vector<double> mass_flow_rate; // [kg/s] indexed by FluidBCType::MassFlowRateType
        std::vector<double> velocity; // [m/s]  indexed by FluidBCType::VelocityType
    };

    struct FluidCellBC {
        FluidBCType kind = FluidBCType::None;
        uint16_t param_idx = std::numeric_limits<uint16_t>::max();
    };

    struct FluidDomain {
        // Topology
        Index n_fluid = 0;
        std::vector<Index> fluid_to_global; // [n_fluid] → N_active compact index
        std::vector<Index> global_to_fluid; // [N_active] → n_fluid fluid index, mhs::invalidIndex = non-fluid
        std::vector<uint8_t> is_fluid; // [N_active] flag

        // Pre-solved frozen flow state (built once by solveFluidFlow)
        std::vector<double> dynamic_viscosity;
        std::vector<double> pressure;
        std::vector<int8_t> flow_axes;
        std::array<std::vector<double>, 3> hydroC;
        std::vector<double> hydraulic_diameter;
        std::vector<double> channel_width;
        std::vector<double> channel_height;

        // Fluid BC tables
        std::vector<mhs::core::FluidCellBC> fluid_bcs;
        mhs::core::FluidBCParamTable fluid_bc_params;
        std::vector<double> fluid_face_area;
        std::vector<double> boundary_temperature_fluid;
    };

} // namespace mhs::core
