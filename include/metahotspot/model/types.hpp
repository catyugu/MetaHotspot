#pragma once
#include "core/types.hpp"

namespace mhs::model {
    using CellId = core::EntityIndex;
    using FacetId = core::EntityIndex;
    using MaterialId = core::EntityIndex;
    using SourceId = core::EntityIndex;
    using BoundaryConditionId = core::EntityIndex;
    using FluidCellId = core::EntityIndex;
    using PointId = core::EntityIndex;

    enum class Axis : std::uint8_t { X, Y, Z };
    enum class MaterialKind : std::uint8_t { Solid, Fluid };

    enum class StudyType : std::uint8_t { Steady, Transient };

    struct Study {
        StudyType type = StudyType::Steady;
        double initial_temperature = 300.0;
        double duration = 0.0;
        double output_interval = 1.0;
    };
} // namespace mhs::model
