#pragma once

#include "core/vector.hpp"
#include "model/model_definition.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace mhs::app::pta {

    struct VariableSpec {
        std::string name;
        std::string value;
    };

    struct RectSpec {
        std::string x;
        std::string y;
        std::string width;
        std::string height;
    };

    struct RectOperation {
        mhs::model::GeometryOperation operation = mhs::model::GeometryOperation::Add;
        RectSpec rect;
    };

    struct BlockSpec {
        std::string material;
        mhs::model::Expression volumetric_heat_source;
        std::string x_offset;
        std::string y_offset;
        std::optional<std::string> thickness;
        mhs::core::Vector<RectOperation> geometry;
    };

    struct LayerSpec {
        std::string thickness;
        std::string x_offset;
        std::string y_offset;
        mhs::core::Vector<BlockSpec> blocks;
    };

    struct PointProbeSpec {
        std::string name;
        std::string x;
        std::string y;
        std::string z;
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
        mhs::core::Vector<VariableSpec> variables;
        LengthUnit length_unit = LengthUnit::Meter;
        mhs::core::Vector<LayerSpec> layers;
        mhs::core::Vector<PointProbeSpec> point_probes;
    };

    std::string rewrite_formal_argument(const std::string& expression, std::string_view replacement);
    mhs::model::ModelDefinition normalize(Definition input);

} // namespace mhs::app::pta
