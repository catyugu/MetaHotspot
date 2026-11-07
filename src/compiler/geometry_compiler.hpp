#pragma once
#include <optional>

#include "model/model.hpp"
#include "model/model_definition.hpp"

#include "core/vector.hpp"

namespace mhs::compiler {

    void build_cell_face_incidence(mhs::model::DiscreteMesh& mesh);

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

    std::optional<std::size_t> classify_point(
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes, const mhs::model::Vec3& point);
    void project_cell_regions(mhs::model::Model& model, const mhs::core::Vector<mhs::model::VolumeSpec>& volumes);
    void compile_cartesian_mesh(mhs::model::Model& model, const mhs::model::MeshSpec& input,
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes);

    // An invalid fallback omits uncovered surface patches.
    mhs::core::Vector<mhs::model::BoundaryFace> project_surface_regions(const mhs::model::DiscreteMesh& mesh,
        const mhs::core::Vector<CompiledBoundaryRegion>& boundaries,
        mhs::model::BoundaryConditionId fallback = mhs::core::invalid_id);
} // namespace mhs::compiler
