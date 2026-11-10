#pragma once
#include <span>

#include "model/discrete_mesh.hpp"
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

    // Geometry borrows only box operations. Region IDs index the caller's input order.
    using RegionId = mhs::core::EntityIndex;
    using VolumeGeometry = std::span<const mhs::model::BoxOperation>;
    struct CompiledGeometry {
        mhs::model::DiscreteMesh mesh;
        mhs::core::Vector<RegionId> cell_regions;
    };

    CompiledGeometry compile_conformal_mesh(const mhs::model::MeshSpec& input, std::span<const VolumeGeometry> volumes);

    // Cells are emitted in lexicographic grid order; a shared-plane probe chooses
    // the last active containing cell, preserving the original closed-box rule.
    mhs::model::CellId locate_conformal_cell(
        const mhs::model::DiscreteMesh& mesh, const mhs::model::MeshSpec& input, const mhs::model::Vec3& point);

    struct SurfaceRegionGroup {
        std::span<const TaggedSurfaceRegion> regions;
        SurfaceTag fallback = mhs::core::invalid_id;
    };

    // Share one geometric partition across tag groups. Later regions in each
    // group take precedence; interior facets always receive the invalid tag.
    mhs::core::Vector<mhs::core::Vector<SurfaceTag>> partition_exterior_facets(
        mhs::model::DiscreteMesh& mesh, std::span<const SurfaceRegionGroup> groups);
} // namespace mhs::compiler
