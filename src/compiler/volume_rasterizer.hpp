#pragma once

#include "compiler/geometry_compiler.hpp"

namespace mhs::compiler {
    // Dense ownership in X/Y/Z lexicographic order. Only effective region
    // boundaries must align; hidden or cancelled box faces may cross cells.
    mhs::core::Vector<RegionId> rasterize_volumes(
        const std::array<std::span<const double>, 3>& axes, std::span<const VolumeGeometry> volumes);
} // namespace mhs::compiler
