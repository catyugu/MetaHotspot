#pragma once

#include "data/internal_model.hpp"
#include "data/io_model.hpp"

namespace mhs::sim {

    // Pre-resolved geometry for a single rect (all values in SI meters)
    struct ResolvedRect {
        bool add_sub;
        double x; // absolute SI x coordinate of rect origin
        double y; // absolute SI y coordinate of rect origin
        double width; // SI width (always positive after normalization)
        double height; // SI height (always positive after normalization)
    };

    // Pre-resolved geometry for a single block
    struct ResolvedBlock {
        std::vector<ResolvedRect> rects;
        std::string material_name;
        std::string ti_reyuan_expr; // kept as string for later mhs::core::parse
    };

    // Pre-resolved geometry for a single layer
    struct ResolvedLayerGeometry {
        std::vector<ResolvedBlock> blocks;
        double z_start; // SI z coordinate of layer bottom
        double z_end; // SI z coordinate of layer top
    };

    // Convert length unit to SI (meters) scale factor
    double length_unit_to_si(mhs::core::LengthUnit unit);

    // Pre-evaluate all geometry expressions for all layers, including Z ranges
    // This eliminates repeated eval_geometry calls in the cell loops
    std::vector<ResolvedLayerGeometry> resolve_geometry(const std::vector<mhs::core::Layer>& layers, double si_scale);

    // Determine which block a cell at (cx, cy, cz) belongs to in a resolved layer
    // Uses pre-evaluated geometry values — no expression evaluation at runtime
    // Traverses blocks in reverse order (last block wins in overlap regions)
    // Returns block index or -1 if cell is virtual
    int find_block_for_cell(const ResolvedLayerGeometry& resolved_layer, double cx, double cy, double cz);

    // Resolve cell validity and material assignment.
    // Returns CellFields (with valid_mask, index_map, material_id) and a temporary
    // layer_id vector (old_idx indexed) used by the preprocessor for heat-source
    // resolution. The layer_id vector is freed by the caller when no longer needed.
    struct LayerResolveResult {
        mhs::core::CellFields cells;
        std::vector<size_t> layer_id_old; // SIZE_MAX for invalid; kept local to caller
    };

    LayerResolveResult resolve_layers(const std::vector<ResolvedLayerGeometry>& resolved_layers,
        const mhs::core::MeshGeometry& mesh, const std::unordered_map<std::string, size_t>& name_to_idx);

} // namespace mhs::sim