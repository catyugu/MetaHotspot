#pragma once
#include <span>

#include "model/fom.hpp"
#include "model/fom_definition.hpp"

#include "core/vector.hpp"

namespace mhs::compiler {

    // Tags are opaque caller-owned indices; geometry has no physics payload.
    using SurfaceTag = mhs::core::EntityIndex;
    struct TaggedSurfaceRegion {
        mhs::model::SurfaceRegion region;
        SurfaceTag tag;
    };

    void compile_conformal_mesh(mhs::model::Fom& model, const mhs::model::MeshSpec& input,
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes);

    // Partition and tag thermal/fluid boundaries together. Later regions take precedence.
    mhs::core::Vector<SurfaceTag> compile_exterior_facets(mhs::model::ConformalMesh& mesh,
        std::span<const TaggedSurfaceRegion> thermal, std::span<const TaggedSurfaceRegion> fluid,
        SurfaceTag thermal_fallback);
} // namespace mhs::compiler
