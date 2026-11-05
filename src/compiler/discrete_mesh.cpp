#include "compiler/geometry_compiler.hpp"
#include <numeric>
#include <stdexcept>

namespace mhs::compiler {
    void prepare_discrete_mesh(mhs::model::DiscreteMesh& mesh)
    {
        using namespace mhs::model;
        using namespace mhs::core;
        auto& [cells, faces, cell_face_offsets, cell_face_ids] = mesh;
        cell_face_offsets.assign(cells.size() + 1, 0);
        for (const auto& f : faces) {
            if (f.owner >= cells.size() || (f.neighbor != core::invalid_id && f.neighbor >= cells.size()))
                throw std::out_of_range("face cell index outside mesh");
            if (static_cast<unsigned>(f.axis) > 2)
                throw std::invalid_argument("face axis must be 0, 1 or 2");
            ++cell_face_offsets[f.owner + 1];
            if (f.neighbor != mhs::core::invalid_id)
                ++cell_face_offsets[f.neighbor + 1];
        }
        std::partial_sum(cell_face_offsets.begin(), cell_face_offsets.end(), cell_face_offsets.begin());
        cell_face_ids.resize(cell_face_offsets.back());
        auto next = cell_face_offsets;
        for (std::size_t i = 0; i < faces.size(); ++i) {
            const auto& f = faces[i];
            cell_face_ids[next[f.owner]++] = static_cast<FaceId>(i);
            if (f.neighbor != mhs::core::invalid_id)
                cell_face_ids[next[f.neighbor]++] = static_cast<FaceId>(i);
        }
    }
} // namespace mhs::compiler
