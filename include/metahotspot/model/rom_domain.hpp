#pragma once

#include "core/vector.hpp"
#include "model/model_definition.hpp"
#include <array>
#include <cstdint>
#include <string>

namespace mhs::model {
    // A boundary facet is physical integration geometry, not a retained FOM cell.
    // The ROM trace maps reduced coordinates to the adjacent temperature DOF.
    // Normal half-resistance has units m^2 K/W (distance / conductivity).
    struct RomPortPatch {
        RegionRect rectangle;
        double half_resistance = 0.0;
        std::array<double, 2> center {};
        core::Vector<double> trace;
        double offset = 0.0;
        // Required for NCC when this port connects to a nonmatching interface.
        // Each gradient row maps modal coordinates to K/m in one tangent direction.
        std::array<core::Vector<double>, 2> tangential_gradient;
    };

    struct RomPort {
        std::string name;
        SurfaceRegion region;
        std::int8_t outward_sign = 1;
        core::Vector<RomPortPatch> patches;
    };

    struct RomObservation {
        PointProbe point;
        core::Vector<double> weights;
        double offset = 0.0;
    };

    // Externally supplied, already projected thermal system.
    // Dense row-major arrays are deliberate: the modal order is small and there
    // is no fictitious internal FOM mesh or material distribution.
    struct RomDomainDefinition {
        std::size_t order = 0;
        core::Vector<double> K, C, f, initial_state;
        core::Vector<RomPort> ports;
        core::Vector<RomObservation> observations;
    };

    using RomDomain = RomDomainDefinition;
} // namespace mhs::model
