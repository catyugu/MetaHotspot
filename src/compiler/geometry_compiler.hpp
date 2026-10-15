#pragma once

#include "core/model.hpp"
#include "core/model_definition.hpp"

#include <vector>

namespace mhs::sim {

    // ──────────────────────────────────────────────────────────────
    //  Pre-resolved geometry types (internal to the compiler library)
    // ──────────────────────────────────────────────────────────────

    /// Pre-resolved geometry for a single rect (all values in SI meters)
    struct ResolvedRect {
        mhs::model::GeometryOperation operation;
        double x;      // absolute SI x coordinate of rect origin
        double y;      // absolute SI y coordinate of rect origin
        double width;  // SI width (always positive after normalization)
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
        double z_end;   // SI z coordinate of layer top
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

    std::vector<ResolvedLayerGeometry> resolve_geometry(const std::vector<mhs::model::LayerSpec> &layers,
                                                        double si_scale, const mhs::core::SymbolTable &symbols);

    std::pair<mhs::core::Index, mhs::core::Index> classify_point(const std::vector<ResolvedLayerGeometry> &layers,
                                                                 const mhs::core::Vec3 &point);
    void project_cell_regions(mhs::core::Model &model, const std::vector<ResolvedLayerGeometry> &layers);
    void compile_cartesian_mesh(mhs::core::Model &model, const mhs::model::MeshSpec &input, double scale,
                                const std::vector<ResolvedLayerGeometry> &layers);
    void build_external_output_mesh(mhs::core::Model &model);

    void project_boundary_patches(mhs::core::Model &model, const std::vector<CompiledBoundaryRegion> &boundaries,
                                  const DefaultBoundary &fallback);

} // namespace mhs::sim
