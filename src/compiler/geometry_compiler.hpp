#pragma once

#include "common/model.hpp"
#include "common/model_definition.hpp"

#include <vector>

namespace mhs::sim {

    // ──────────────────────────────────────────────────────────────
    //  Pre-resolved geometry types (internal to the compiler library)
    // ──────────────────────────────────────────────────────────────

    /// Pre-resolved geometry for a single rect (all values in SI meters)
    struct ResolvedRect {
        mhs::model::GeometryOperation operation;
        double x; // absolute SI x coordinate of rect origin
        double y; // absolute SI y coordinate of rect origin
        double width; // SI width (always positive after normalization)
        double height; // SI height (always positive after normalization)
    };

    /// Pre-resolved geometry for a single block
    struct ResolvedBlock {
        std::vector<ResolvedRect> rects;
        mhs::core::TableIndex material_id = 0;
        mhs::core::TableIndex heat_source_idx = 0;
        double z_start = 0.0;
        double z_end = 0.0;
    };

    /// Pre-resolved geometry for a single layer
    struct ResolvedLayerGeometry {
        std::vector<ResolvedBlock> blocks;
        double z_start; // SI z coordinate of layer bottom
        double z_end; // SI z coordinate of layer top
    };

    /// Compiled boundary region with resolved parameters
    struct CompiledBoundaryRegion {
        mhs::model::Axis axis = mhs::model::Axis::Z;
        double coordinate = 0.0;
        std::vector<mhs::model::RegionRect> rectangles;
        mhs::core::BcType type = mhs::core::BcType::None;
        mhs::core::TableIndex parameter_index = 0;
    };

    struct DefaultBoundary {
        mhs::core::BcType type = mhs::core::BcType::None;
        mhs::core::TableIndex parameter_index = 0;
    };

    // ──────────────────────────────────────────────────────────────
    //  Public compiler API
    // ──────────────────────────────────────────────────────────────

    std::vector<ResolvedLayerGeometry> resolve_geometry(
        const std::vector<mhs::model::LayerSpec>& layers, double si_scale, const mhs::core::SymbolTable& symbols);

    mhs::core::CellFields assign_cell_layers(
        const std::vector<ResolvedLayerGeometry>& resolved_layers, const mhs::core::MeshGeometry& mesh);


    void resolve_boundary_patches(const mhs::core::MeshGeometry& mesh, const mhs::core::CellFields& cells,
        const std::vector<CompiledBoundaryRegion>& boundaries, const DefaultBoundary& default_boundary,
        std::vector<mhs::core::FaceBC>& face_bcs);

} // namespace mhs::sim
