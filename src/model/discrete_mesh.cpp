#include "model/discrete_mesh.hpp"
#include <numeric>

namespace mhs::model {
    void DiscreteMesh::build_incidence()
    {
        if (!cells.empty())
            (void)CellId {cells.size() - 1};
        if (!faces.empty())
            (void)FaceId {faces.size() - 1};
        cell_face_offsets.assign(cells.size() + 1, 0);
        for (const auto& f : faces) {
            ++cell_face_offsets[f.owner.value + 1];
            if (f.neighbor.valid())
                ++cell_face_offsets[f.neighbor.value + 1];
        }
        std::partial_sum(cell_face_offsets.begin(), cell_face_offsets.end(), cell_face_offsets.begin());
        cell_face_ids.resize(cell_face_offsets.back());
        auto next = cell_face_offsets;
        for (std::size_t i = 0; i < faces.size(); ++i) {
            const auto& f = faces[i];
            cell_face_ids[next[f.owner.value]++] = FaceId {i};
            if (f.neighbor.valid())
                cell_face_ids[next[f.neighbor.value]++] = FaceId {i};
        }
    }

    CellId DiscreteMesh::locate(const Vec3& p) const
    {
        // Cache once per probe. Shared-face ties select the lexicographically
        // greatest lower corner, independent of cell storage/DOF ordering.
        CellId result;
        for (std::size_t i = 0; i < cells.size(); ++i)
            if (cells[i].contains(p) && (!result.valid() || cells[i].min > cells[result.value].min))
                result = CellId {i};
        return result;
    }
} // namespace mhs::model
