#pragma once
#include "model/model_definition.hpp"
#include <string_view>
namespace mhs::app::pta {
    using namespace mhs::model;
    struct RectSpec {
        Expression x;
        Expression y;
        Expression width;
        Expression height;
    };

    struct RectOperation {
        GeometryOperation operation = GeometryOperation::Add;
        RectSpec rect;
    };

    struct BlockSpec {
        std::string material;
        Expression volumetric_heat_source;
        Expression x_offset;
        Expression y_offset;
        std::optional<Expression> thickness;
        mhs::core::Vector<RectOperation> geometry;
    };

    struct LayerSpec {
        Expression thickness;
        Expression x_offset;
        Expression y_offset;
        mhs::core::Vector<BlockSpec> blocks;
    };

    struct ObservationPointSpec {
        std::string name;
        Expression x, y, z;
    };
    enum class LengthUnit { Meter, Millimeter, Micrometer, Nanometer, Inch, Mil };
    // Parsing scratch state. PTA syntax never enters the library definition.
    struct Definition {
        mhs::model::ModelDefinition model;
        LengthUnit length_unit = LengthUnit::Meter;
        mhs::core::Vector<LayerSpec> layers;
        mhs::core::Vector<ObservationPointSpec> observation_points;
    };
    std::string rewrite_formal_argument(const std::string&, std::string_view);
    mhs::model::ModelDefinition normalize(Definition input);
} // namespace mhs::app::pta
