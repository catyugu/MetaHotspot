#include "compiler/geometry_compiler.hpp"
#include <numeric>

namespace mhs::compiler {
    void build_cell_face_incidence(mhs::model::DiscreteMesh& mesh)
    {
        using namespace mhs::model;
        using namespace mhs::core;
        auto& [cells, faces, cell_face_offsets, cell_face_ids] = mesh;
        cell_face_offsets.assign(cells.size() + 1, 0);
        for (const auto& f : faces) {
            if (f.minus != invalid_id)
                ++cell_face_offsets[f.minus + 1];
            if (f.plus != invalid_id)
                ++cell_face_offsets[f.plus + 1];
        }
        std::partial_sum(cell_face_offsets.begin(), cell_face_offsets.end(), cell_face_offsets.begin());
        cell_face_ids.resize(cell_face_offsets.back());
        auto next = cell_face_offsets;
        for (std::size_t i = 0; i < faces.size(); ++i) {
            const auto& f = faces[i];
            if (f.minus != invalid_id)
                cell_face_ids[next[f.minus]++] = static_cast<FaceId>(i);
            if (f.plus != invalid_id)
                cell_face_ids[next[f.plus]++] = static_cast<FaceId>(i);
        }
    }
} // namespace mhs::compiler
