#pragma once

#include "runtime/model.hpp"
#include "model/model_definition.hpp"

#include <vector>

namespace mhs::sim {

    struct CompiledBoundaryRegion {
        mhs::model::Axis axis = mhs::model::Axis::Z;
        double coordinate = 0.0;
        std::vector<mhs::model::RegionRect> rectangles;
        mhs::core::BcType type = mhs::core::BcType::None;
        uint16_t parameter_index = 0;
    };

    struct DefaultBoundary {
        mhs::core::BcType type = mhs::core::BcType::None;
        uint16_t parameter_index = 0;
    };

    // Pre-resolved geometry for a single rect (all values in SI meters)
    struct ResolvedRect {
        mhs::model::GeometryOperation operation;
        double x; // absolute SI x coordinate of rect origin
        double y; // absolute SI y coordinate of rect origin
        double width; // SI width (always positive after normalization)
        double height; // SI height (always positive after normalization)
    };

    // Pre-resolved geometry for a single block
    struct ResolvedBlock {
        std::vector<ResolvedRect> rects;
        std::string material;
        std::string volumetric_heat_source;

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
        const std::vector<mhs::model::LayerSpec>& layers, double si_scale, const mhs::core::SymbolTable& symbols);

    // Assign every grid cell to its layer + block and write volumetric cell fields.
    // Returns CellFields with exact inverse topology maps: grid_to_cell
    // (full-grid; invalidIndex = virtual) and cell_to_grid (compact), plus
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
    // `cells` must already have a valid grid_to_cell (from assign_cell_layers).
    // Boundaries are already structured and scaled by compile_boundary_patches().
    // The default boundary is the fallback for exposed faces that do not
    // match any explicit structured region.
    void resolve_boundary_patches(const mhs::core::MeshGeometry& mesh, const mhs::core::CellFields& cells,
        const std::vector<CompiledBoundaryRegion>& boundaries, const DefaultBoundary& default_boundary,
        std::vector<mhs::core::FaceBC>& face_bcs);

} // namespace mhs::sim
