#pragma once
#include <optional>
#include <span>
#include <utility>

#include "model/model.hpp"
#include "model/model_definition.hpp"

#include "core/vector.hpp"

namespace mhs::compiler {

    void build_cell_face_incidence(mhs::model::DiscreteMesh& mesh);

    // Tags are opaque caller-owned indices; geometry carries no physics payload.
    struct TaggedSurfaceRegion {
        mhs::model::SurfaceRegion region;
        mhs::core::EntityIndex tag;
    };

    using SurfaceBounds = std::array<double, 4>;
    using TaggedBounds = std::pair<SurfaceBounds, mhs::core::EntityIndex>;
    struct ProjectedSurfacePatch {
        mhs::model::CellId cell;
        mhs::model::Axis axis;
        std::int8_t sign;
        mhs::core::EntityIndex tag;
        SurfaceBounds bounds;
    };

    SurfaceBounds surface_bounds(const mhs::model::RegionRect& rectangle);
    SurfaceBounds intersect_bounds(const SurfaceBounds& a, const SurfaceBounds& b);
    bool positive_bounds(const SurfaceBounds& bounds);
    double surface_area(const SurfaceBounds& bounds);
    mhs::core::Vector<TaggedBounds> partition_surface_bounds(const SurfaceBounds& outer,
        std::span<const TaggedBounds> regions, std::optional<mhs::core::EntityIndex> fallback = std::nullopt);

    // ──────────────────────────────────────────────────────────────
    //  Internal compiler operations
    // ──────────────────────────────────────────────────────────────

    std::optional<std::size_t> classify_point(
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes, const mhs::model::Vec3& point);
    void project_cell_regions(mhs::model::FomDomain& model, const mhs::core::Vector<mhs::model::VolumeSpec>& volumes);
    void compile_cartesian_mesh(mhs::model::FomDomain& model, const mhs::model::MeshSpec& input,
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes);

    // With no fallback tag, uncovered exterior patches are omitted.
    mhs::core::Vector<ProjectedSurfacePatch> project_exterior_surface_regions(const mhs::model::DiscreteMesh& mesh,
        const mhs::core::Vector<TaggedSurfaceRegion>& boundaries,
        std::optional<mhs::core::EntityIndex> fallback = std::nullopt);
} // namespace mhs::compiler
