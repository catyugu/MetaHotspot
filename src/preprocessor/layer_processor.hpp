#pragma once

#include "data/io_structure.hpp"
#include "data/model.hpp"
#include "expr/expr.hpp"
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

    // Pre-evaluate all geometry expressions for all layers, including Z ranges
    // This eliminates repeated eval_geometry calls in the cell loops.
    // `symbols` provides the geometry variables each expression may reference.
    std::vector<ResolvedLayerGeometry> resolve_geometry(
        const std::vector<mhs::core::Layer>& layers, double si_scale, const mhs::core::SymbolTable& symbols);

    // Assign every grid cell to its layer + block and write volumetric cell fields.
    // Returns CellFields with index_map (full-grid; invalidIndex = virtual),
    // material_id and heat_source_idx (both compact by active count).
    //
    // `block_hs_map[l][b]` = heat_source_table index for layer l / block b.
    // No BC parameters needed — boundary resolution is a separate step.
    mhs::core::CellFields assign_cell_layers(const std::vector<ResolvedLayerGeometry>& resolved_layers,
        const mhs::core::MeshGeometry& mesh, const std::unordered_map<std::string, size_t>& name_to_idx,
        const std::vector<std::vector<uint16_t>>& block_hs_map);

    // Resolve boundary patches for every exposed face of every active cell.
    // Single grid traversal: writes directly into boundary.face_bcs[]. No
    // prefix-sum or intermediate scan needed — face_bcs is [N_active * 6].
    //
    // `cells` must already have a valid index_map (from assign_cell_layers).
    // `parsed_face_keys` comes from parse_all_face_keys().
    // Other_bc is the fallback BC for faces that don't match any face key.
    void resolve_boundary_patches(const mhs::core::MeshGeometry& mesh, const mhs::core::CellFields& cells,
        const std::vector<ParsedFaceKey>& parsed_face_keys, const OtherBC& other_bc,
        std::vector<mhs::core::FaceBC>& face_bcs);

} // namespace mhs::sim
