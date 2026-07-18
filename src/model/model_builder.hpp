#pragma once

#include "model/model_definition.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace mhs::model {

    using LayerId = uint32_t;
    using BlockId = uint32_t;

    struct LayerParams {
        Expression thickness;
        Expression x_offset;
        Expression y_offset;
    };

    struct BlockParams {
        std::string material;
        Expression volumetric_heat_source;
        Expression x_offset;
        Expression y_offset;
        std::optional<Expression> thickness;
    };

    class ModelBuilder final {
    public:
        void set_settings(ModelSettings settings) { model_.settings = std::move(settings); }
        void set_mesh(MeshSpec mesh) { model_.mesh = std::move(mesh); }
        void add_variable(VariableSpec variable) { model_.variables.push_back(std::move(variable)); }
        void add_function(NamedFunction function) { model_.functions.push_back(std::move(function)); }
        void add_material(NamedMaterial material) { model_.materials.push_back(std::move(material)); }

        LayerId add_layer(LayerParams layer);
        BlockId add_block(LayerId layer, BlockParams block);
        void add_rect(BlockId block, RectOperation operation);

        void add_boundary(BoundaryPatch boundary) { model_.boundaries.push_back(std::move(boundary)); }
        void set_default_boundary(ThermalBoundary boundary) { model_.default_boundary = std::move(boundary); }
        void add_observation_point(ObservationPointSpec point) { model_.observation_points.push_back(std::move(point)); }
        void add_fluid_boundary(FluidBoundarySpec boundary) { model_.fluid_boundaries.push_back(std::move(boundary)); }

        ModelDefinition finish() && { return std::move(model_); }

    private:
        struct BlockLocation {
            LayerId layer;
            uint32_t block;
        };

        ModelDefinition model_;
        std::vector<BlockLocation> blocks_;
    };

} // namespace mhs::model
