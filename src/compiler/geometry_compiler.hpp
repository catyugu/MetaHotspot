#pragma once
#include <optional>

#include "model/fom.hpp"
#include "model/fom_definition.hpp"

#include "core/vector.hpp"

namespace mhs::compiler {

    void build_cell_facet_incidence(mhs::model::DiscreteMesh& mesh);

    // Tags are opaque caller-owned indices; geometry has no physics payload.
    using SurfaceTag = mhs::core::EntityIndex;
    struct TaggedSurfaceRegion {
        mhs::model::SurfaceRegion region;
        SurfaceTag tag;
    };

    std::optional<std::size_t> classify_point(
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes, const mhs::model::Vec3& point);
    void compile_conformal_mesh(mhs::model::Fom& model, const mhs::model::MeshSpec& input,
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes);

    // Partition exterior facets once against the union of thermal and fluid regions,
    // before building incidence and assigning physics-specific boundary conditions.
    void refine_exterior_facets(
        mhs::model::DiscreteMesh& mesh, const mhs::core::Vector<mhs::model::SurfaceRegion>& regions);

    // Later regions take precedence. Interior facets always have the invalid tag.
    mhs::core::Vector<SurfaceTag> assign_exterior_regions(const mhs::model::DiscreteMesh& mesh,
        const mhs::core::Vector<TaggedSurfaceRegion>& regions, SurfaceTag fallback = mhs::core::invalid_id);
    // Temporary traversal order for consumers that process one surface at a time.
    inline auto group_exterior_facets(const model::Fom& fom)
    {
        core::Vector<core::Vector<model::FacetId>> groups(fom.surfaces.size());
        for (std::size_t i = 0; i < fom.facet_surface.size(); ++i)
            if (const auto id = fom.facet_surface[i]; id != core::invalid_id)
                groups[id].push_back(static_cast<model::FacetId>(i));
        return groups;
    }
} // namespace mhs::compiler
