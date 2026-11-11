#pragma once
#include <optional>

#include "model/model.hpp"
#include "model/model_definition.hpp"

#include "core/vector.hpp"

namespace mhs::compiler {

    bool compile_cell(mhs::model::Model& model, const mhs::model::CellGeometry& cell,
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes);
    void compile_nonconformal_mesh(mhs::model::Model& model, const mhs::model::NonConformalMesh& input,
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes);

    // Tags are opaque caller-owned indices; geometry has no physics payload.
    using SurfaceTag = mhs::core::EntityIndex;
    struct TaggedSurfaceRegion {
        mhs::model::SurfaceRegion region;
        SurfaceTag tag;
    };

    std::optional<std::size_t> classify_point(
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes, const mhs::model::Vec3& point);
    void compile_conformal_mesh(mhs::model::Model& model, const mhs::model::ConformalMeshSpec& input,
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes);

    // Partition exterior facets once against the union of thermal and fluid regions,
    // before assigning physics-specific boundary conditions.
    void refine_boundary_facets(mhs::core::Vector<mhs::model::BoundaryFacet>& facets,
        const mhs::core::Vector<mhs::model::SurfaceRegion>& regions);

    // Later regions take precedence. Only external boundary pieces are assigned.
    mhs::core::Vector<SurfaceTag> assign_boundary_regions(const mhs::core::Vector<mhs::model::BoundaryFacet>& facets,
        const mhs::core::Vector<TaggedSurfaceRegion>& regions, SurfaceTag fallback = mhs::core::invalid_id);
} // namespace mhs::compiler
