#pragma once
#include <optional>

#include "model/model.hpp"
#include "model/model_definition.hpp"

#include "core/vector.hpp"

namespace mhs::compiler {

    void build_cell_face_incidence(mhs::model::DiscreteMesh& mesh);

    // Tags are opaque caller-owned indices; geometry carries no physics payload.
    struct TaggedSurfaceRegion {
        mhs::model::SurfaceRegion region;
        std::size_t tag;
    };

    struct ProjectedSurfacePatch {
        mhs::model::CellId cell;
        mhs::model::Axis axis;
        std::int8_t sign;
        std::size_t tag;
        std::array<double, 4> bounds;
    };

    // ──────────────────────────────────────────────────────────────
    //  Internal compiler operations
    // ──────────────────────────────────────────────────────────────

    std::optional<std::size_t> classify_point(
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes, const mhs::model::Vec3& point);
    void project_cell_regions(mhs::model::Model& model, const mhs::core::Vector<mhs::model::VolumeSpec>& volumes);
    void compile_cartesian_mesh(mhs::model::Model& model, const mhs::model::MeshSpec& input,
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes);

    // With no fallback tag, uncovered exterior patches are omitted.
    mhs::core::Vector<ProjectedSurfacePatch> project_surface_regions(const mhs::model::DiscreteMesh& mesh,
        const mhs::core::Vector<TaggedSurfaceRegion>& boundaries, std::optional<std::size_t> fallback = std::nullopt);
} // namespace mhs::compiler
