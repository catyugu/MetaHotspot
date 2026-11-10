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

    // Reused across cells; contains no persistent physical or mesh state.
    struct RegionProjectionWorkspace {
        std::array<mhs::core::Vector<double>, 3> cuts;
        mhs::core::Vector<mhs::model::RegionId> candidates;
        mhs::core::Vector<bool> inside;
    };

    // Returns false for cells wholly outside the material domain. Accepted cells
    // append one material and their source integrals before mesh insertion.
    bool project_cell_regions(mhs::model::Model& model, const mhs::model::CellGeometry& cell,
        const mhs::core::Vector<mhs::model::RegionSpec>& regions,
        const mhs::core::Vector<mhs::model::MaterialAssignment>& materials,
        const mhs::core::Vector<mhs::model::SourceAssignment>& sources, RegionProjectionWorkspace& workspace);
    void compile_cartesian_mesh(mhs::model::Model& model, const mhs::model::MeshSpec& input,
        const mhs::core::Vector<mhs::model::RegionSpec>& regions,
        const mhs::core::Vector<mhs::model::MaterialAssignment>& materials,
        const mhs::core::Vector<mhs::model::SourceAssignment>& sources);

    // Partition exterior facets once against the union of thermal and fluid regions,
    // before building incidence and assigning physics-specific boundary conditions.
    void refine_exterior_facets(
        mhs::model::DiscreteMesh& mesh, const mhs::core::Vector<mhs::model::SurfaceRegion>& regions);

    // Later regions take precedence. Interior facets always have the invalid tag.
    mhs::core::Vector<SurfaceTag> assign_exterior_regions(const mhs::model::DiscreteMesh& mesh,
        const mhs::core::Vector<TaggedSurfaceRegion>& regions, SurfaceTag fallback = mhs::core::invalid_id);
} // namespace mhs::compiler
