#pragma once

#include "compiler/engine_types.hpp"

#include <vector>

namespace mhs::core {

    struct FluidDomain {
        // Runtime topology. fluid_to_global enables fluid-only assembly;
        // global_to_fluid also acts as the fluid membership map.
        std::vector<Index> fluid_to_global;
        std::vector<Index> global_to_fluid;

        // Frozen hydraulic volume flux for every directed fluid-cell face.
        // Layout: [fluid_index * FACE_COUNT + face].
        std::vector<double> face_volume_flux;

        // Nu / hydraulic_diameter for each fluid cell. The thermal assembly
        // evaluates h = factor * k_fluid at the current temperature.
        std::vector<double> interface_heat_transfer_factor;

        // Prescribed boundary outflux. NaN means that pressure-derived net
        // outflux is used; finite values (including zero) explicitly override it.
        std::vector<double> boundary_outflux;
        std::vector<double> boundary_temperature;
    };

} // namespace mhs::core
