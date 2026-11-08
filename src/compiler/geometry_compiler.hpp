#pragma once
#include <optional>

#include "model/model.hpp"
#include "model/model_definition.hpp"

#include "core/vector.hpp"

namespace mhs::compiler {

    void build_cell_face_incidence(mhs::model::DiscreteMesh& mesh);

    // Non-owning geometry input; tag is interpreted only by the caller.
    struct SurfaceRegion {
        const mhs::model::FaceRegion& geometry;
        std::size_t tag;
    };

    struct SurfacePatch {
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

    // An invalid fallback omits uncovered surface patches.
    mhs::core::Vector<SurfacePatch> project_surface_regions(const mhs::model::DiscreteMesh& mesh,
        const mhs::core::Vector<SurfaceRegion>& boundaries, std::size_t fallback = mhs::core::invalid_id);
} // namespace mhs::compiler
