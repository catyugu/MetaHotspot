#pragma once
#include "core/types.hpp"

namespace mhs::model {
    using CellId = core::EntityIndex;
    using FaceId = core::EntityIndex;
    using MaterialId = core::EntityIndex;
    using SourceId = core::EntityIndex;
    using BoundaryConditionId = core::EntityIndex;
    using FluidCellId = core::EntityIndex;
    using PointId = core::EntityIndex;

    enum class Axis : std::uint8_t { X, Y, Z };

    enum class StudyType : std::uint8_t { Steady, Transient };
} // namespace mhs::model
