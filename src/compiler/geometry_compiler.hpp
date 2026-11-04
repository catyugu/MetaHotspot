#pragma once
#include <optional>

#include "model/model.hpp"
#include "model/model_definition.hpp"

#include "core/vector.hpp"

namespace mhs::compiler {

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
        mhs::core::Vector<ResolvedRect> rects;
        mhs::model::MaterialId material_id = mhs::core::invalid_id;
        mhs::model::SourceId heat_source_idx = mhs::core::invalid_id;
        double z_start = 0.0;
        double z_end = 0.0;
    };

    /// Pre-resolved geometry for a single layer
    struct ResolvedLayerGeometry {
        mhs::core::Vector<ResolvedBlock> blocks;
        double z_start; // SI z coordinate of layer bottom
        double z_end; // SI z coordinate of layer top
    };

    /// Compiled boundary region referencing a condition in the caller's table
    struct CompiledBoundaryRegion {
        mhs::model::Axis axis = mhs::model::Axis::Z;
        double coordinate = 0.0;
        mhs::core::Vector<mhs::model::RegionRect> rectangles;
        mhs::model::BoundaryConditionId condition_index = mhs::core::invalid_id;
    };

    // ──────────────────────────────────────────────────────────────
    //  Internal compiler operations
    // ──────────────────────────────────────────────────────────────

    mhs::core::Vector<ResolvedLayerGeometry> resolve_geometry(const mhs::core::Vector<mhs::model::LayerSpec>& layers,
        double si_scale, const mhs::expression::SymbolTable& symbols);

    std::optional<std::pair<std::size_t, std::size_t>> classify_point(
        const mhs::core::Vector<ResolvedLayerGeometry>& layers, const mhs::model::Vec3& point);
    void project_cell_regions(mhs::model::Model& model, const mhs::core::Vector<ResolvedLayerGeometry>& layers);
    void compile_cartesian_mesh(mhs::model::Model& model, const mhs::model::MeshSpec& input, double scale,
        const mhs::core::Vector<ResolvedLayerGeometry>& layers);
    void build_external_output_mesh(mhs::model::Model& model);

    // An invalid fallback omits uncovered surface patches.
    mhs::core::Vector<mhs::model::BoundaryFace> project_surface_regions(const mhs::model::DiscreteMesh& mesh,
        const mhs::core::Vector<CompiledBoundaryRegion>& boundaries,
        mhs::model::BoundaryConditionId fallback = mhs::core::invalid_id);
} // namespace mhs::compiler
