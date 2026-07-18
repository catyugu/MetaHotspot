#include "model/model_builder.hpp"

namespace mhs::model {

    LayerId ModelBuilder::add_layer(LayerParams layer)
    {
        const auto id = static_cast<LayerId>(model_.layers.size());
        model_.layers.push_back({std::move(layer.thickness), std::move(layer.x_offset), std::move(layer.y_offset), {}});
        return id;
    }

    BlockId ModelBuilder::add_block(LayerId layer, BlockParams block)
    {
        auto& blocks = model_.layers[layer].blocks;
        const BlockLocation location {layer, static_cast<uint32_t>(blocks.size())};
        blocks.push_back({std::move(block.material), std::move(block.volumetric_heat_source), std::move(block.x_offset),
            std::move(block.y_offset), std::move(block.thickness), {}});
        blocks_.push_back(location);
        return static_cast<BlockId>(blocks_.size() - 1);
    }

    void ModelBuilder::add_rect(BlockId block, RectOperation operation)
    {
        const auto location = blocks_[block];
        model_.layers[location.layer].blocks[location.block].geometry.push_back(std::move(operation));
    }

} // namespace mhs::model
