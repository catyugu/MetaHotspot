#pragma once
#include "core/types.hpp"

namespace mhs::model {
    using CellId = core::Id<struct CellTag>;
    using FaceId = core::Id<struct FaceTag>;
    using MaterialId = core::Id<struct MaterialTag>;
    using SourceId = core::Id<struct SourceTag>;
    using BoundaryConditionId = core::Id<struct BoundaryConditionTag>;
    using FluidCellId = core::Id<struct FluidCellTag>;
    using PointId = core::Id<struct OutputPointTag>;

    enum class StudyType : std::uint8_t { Steady, Transient };
} // namespace mhs::model
