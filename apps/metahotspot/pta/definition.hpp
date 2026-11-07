#pragma once

#include "core/vector.hpp"
#include "model/model_definition.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace mhs::app::pta {

    struct RectSpec {
        mhs::model::Expression x;
        mhs::model::Expression y;
        mhs::model::Expression width;
        mhs::model::Expression height;
    };

    struct RectOperation {
        mhs::model::GeometryOperation operation = mhs::model::GeometryOperation::Add;
        RectSpec rect;
    };

    struct BlockSpec {
        std::string material;
        mhs::model::Expression volumetric_heat_source;
        mhs::model::Expression x_offset;
        mhs::model::Expression y_offset;
        std::optional<mhs::model::Expression> thickness;
        mhs::core::Vector<RectOperation> geometry;
    };

    struct LayerSpec {
        mhs::model::Expression thickness;
        mhs::model::Expression x_offset;
        mhs::model::Expression y_offset;
        mhs::core::Vector<BlockSpec> blocks;
    };

    struct ObservationPointSpec {
        std::string name;
        mhs::model::Expression x;
        mhs::model::Expression y;
        mhs::model::Expression z;
    };

    enum class LengthUnit {
        Meter,
        Millimeter,
        Micrometer,
        Nanometer,
        Inch,
        Mil,
    };

    // Parsing scratch state. PTA syntax never enters the library definition.
    struct Definition {
        mhs::model::ModelDefinition model;
        LengthUnit length_unit = LengthUnit::Meter;
        mhs::core::Vector<LayerSpec> layers;
        mhs::core::Vector<ObservationPointSpec> observation_points;
    };

    std::string rewrite_formal_argument(const std::string& expression, std::string_view replacement);
    mhs::model::ModelDefinition normalize(Definition input);

} // namespace mhs::app::pta
