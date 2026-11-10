#pragma once

#include "model/model.hpp"
#include "model/model_definition.hpp"

#include "core/vector.hpp"

namespace mhs::compiler {

    void build_cell_facet_incidence(mhs::model::DiscreteMesh& mesh);

    // Tags are opaque caller-owned indices; geometry has no physics payload.
    using SurfaceTag = mhs::core::EntityIndex;
    struct TaggedSurfaceRegion {
        mhs::model::SurfaceRegion region;
        SurfaceTag tag;
    };

    // Ordered Add/Subtract operations form one region; overlapping source objects add.
    bool geometry_contains(const mhs::model::VolumeGeometry& geometry, const mhs::model::Vec3& point);
    void compile_cartesian_mesh(mhs::model::Model& model, const mhs::model::MeshSpec& input,
        const mhs::core::Vector<mhs::model::BodySpec>& bodies);
    void project_volume_sources(
        mhs::model::Model& model, const mhs::core::Vector<mhs::model::VolumeSourceSpec>& sources);

    // Partition exterior facets once against the union of thermal and fluid regions,
    // before building incidence and assigning physics-specific boundary conditions.
    void refine_exterior_facets(
        mhs::model::DiscreteMesh& mesh, const mhs::core::Vector<mhs::model::SurfaceRegion>& regions);

    // Later regions take precedence. Interior facets always have the invalid tag.
    mhs::core::Vector<SurfaceTag> assign_exterior_regions(const mhs::model::DiscreteMesh& mesh,
        const mhs::core::Vector<TaggedSurfaceRegion>& regions, SurfaceTag fallback = mhs::core::invalid_id);
} // namespace mhs::compiler
