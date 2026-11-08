#pragma once
#include "core/types.hpp"

namespace mhs::model {
    using DomainId = core::EntityIndex;
    using PortId = core::EntityIndex;
    using ConnectionId = core::EntityIndex;
    using FieldId = core::EntityIndex;
    using DofId = core::EntityIndex;
    using CellId = core::EntityIndex;
    using FaceId = core::EntityIndex;
    using MaterialId = core::EntityIndex;
    using SourceId = core::EntityIndex;
    using BoundaryConditionId = core::EntityIndex;
    using FluidCellId = core::EntityIndex;
    using PointId = core::EntityIndex;

    enum class Axis : std::uint8_t { X, Y, Z };

    enum class StudyType : std::uint8_t { Steady, Transient };

    struct Study {
        StudyType type = StudyType::Steady;
        double initial_temperature = 300.0;
        double duration = 0.0;
        double output_interval = 1.0;
    };
} // namespace mhs::model
