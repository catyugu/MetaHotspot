#pragma once

#include "core/model.hpp"

#include <algorithm>
#include <span>

namespace mhs::core {

    inline FacePatch projected_face(const DiscreteMesh& mesh, const BoundaryTerm& term)
    {
        auto f = mesh.faces[term.face.value];
        const int a = f.axis == 0 ? 1 : 0, b = f.axis == 2 ? 1 : 2;
        f.min[a] = term.bounds[0];
        f.max[a] = term.bounds[1];
        f.min[b] = term.bounds[2];
        f.max[b] = term.bounds[3];
        f.area
            = (f.max[(f.axis + 1) % 3] - f.min[(f.axis + 1) % 3]) * (f.max[(f.axis + 2) % 3] - f.min[(f.axis + 2) % 3]);
        return f;
    }

    // Terms are sorted by parent FaceId; lookup needs no per-face cache.
    inline std::span<const BoundaryTerm> boundary_terms_on_face(const Model& model, FaceId face)
    {
        const auto& terms = model.boundary_terms;
        auto first = std::lower_bound(
            terms.begin(), terms.end(), face, [](const BoundaryTerm& t, FaceId f) { return t.face.value < f.value; });
        auto last = first;
        while (last != terms.end() && last->face == face)
            ++last;
        return {first, last};
    }

} // namespace mhs::core
