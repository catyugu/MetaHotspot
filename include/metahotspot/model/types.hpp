#pragma once
#include "core/types.hpp"

namespace mhs::model {
    using CellId = core::EntityIndex;
    using FacetId = core::EntityIndex;
    using MaterialId = core::EntityIndex;
    using ComponentId = core::EntityIndex;
    using OutputId = core::EntityIndex;
    using SurfaceId = core::EntityIndex;
    using SourceId = core::EntityIndex;
    using FluidCellId = core::EntityIndex;
    using PointId = core::EntityIndex;

    enum class Axis : std::uint8_t { X, Y, Z };
    enum class MaterialKind : std::uint8_t { Solid, Fluid };

} // namespace mhs::model
