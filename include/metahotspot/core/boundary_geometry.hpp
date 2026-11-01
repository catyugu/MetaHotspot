#pragma once

#include "core/model.hpp"

#include <algorithm>
#include <span>

namespace mhs::core {

    inline FacePatch projected_face(const DiscreteMesh& mesh, const BoundaryFace& face)
    {
        auto f = mesh.faces[face.parent.value];
        const int a = f.axis == 0 ? 1 : 0, b = f.axis == 2 ? 1 : 2;
        f.min[a] = face.bounds[0];
        f.max[a] = face.bounds[1];
        f.min[b] = face.bounds[2];
        f.max[b] = face.bounds[3];
        f.area
            = (f.max[(f.axis + 1) % 3] - f.min[(f.axis + 1) % 3]) * (f.max[(f.axis + 2) % 3] - f.min[(f.axis + 2) % 3]);
        return f;
    }

    // Terms are sorted by parent FaceId; lookup needs no per-face cache.
    inline std::span<const BoundaryFace> boundary_faces_on(const Model& model, FaceId parent)
    {
        const auto& faces = model.boundary_faces;
        auto first = std::lower_bound(faces.begin(), faces.end(), parent,
            [](const BoundaryFace& f, FaceId p) { return f.parent.value < p.value; });
        auto last = first;
        while (last != faces.end() && last->parent == parent)
            ++last;
        return {first, last};
    }

} // namespace mhs::core
