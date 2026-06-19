#pragma once

#include "data/internal_model.hpp"
#include "data/io_model.hpp"
#include "face_key_processor.hpp" // for ParsedFaceKey

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

        // 该 Block 在世界坐标系中的 Z 范围
        double z_start = 0.0;
        double z_end = 0.0;
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

    // Resolve cell validity, material, heat-source assignment, and BCs in a single pass.
    // Returns CellFields (index_map full-grid, with invalidIndex marking virtual cells;
    // material_id + heat_source_idx + cell_bcs compact by cell_bcs.size()).
    //
    // `block_hs_map[l][b]` = heat_source_table index for layer l / block b.
    // `parsed_face_keys` comes from parse_all_face_keys() — the flattened boundary list.
    // BCs are assigned in-rect via face key matching; exposed faces not matched receive other_bc.
    // cell_bcs is default-zero-initialized (BcType::None = 0) before BC assignment.
    mhs::core::CellFields resolve_layers(const std::vector<ResolvedLayerGeometry>& resolved_layers,
        const mhs::core::MeshGeometry& mesh, const std::unordered_map<std::string, size_t>& name_to_idx,
        const std::vector<std::vector<uint16_t>>& block_hs_map,
        const std::vector<ParsedFaceKey>& parsed_face_keys,
        mhs::core::BcType other_bc_enum, uint16_t other_bc_idx);

} // namespace mhs::sim