#pragma once
#include "core/types.hpp"
#include <algorithm>
#include <array>
#include <numeric>
#include <span>
#include <vector>

namespace mhs::core {
    template <class Tag> struct EntityId {
        Index value = invalidIndex;
        bool valid() const { return value != invalidIndex; }
        auto operator<=>(const EntityId&) const = default;
    };
    using CellId = EntityId<struct CellTag>;
    using FaceId = EntityId<struct FaceTag>;
    using Vec3 = std::array<double, 3>;

    struct CellGeometry {
        Vec3 min {}, max {};
        Vec3 center() const { return {(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2}; }
        Vec3 size() const { return {max[0] - min[0], max[1] - min[1], max[2] - min[2]}; }
        double volume() const
        {
            auto d = size();
            return d[0] * d[1] * d[2];
        }
        bool contains(const Vec3& p, double eps = 0) const
        {
            for (int a = 0; a < 3; ++a)
                if (p[a] < min[a] - eps || p[a] > max[a] + eps)
                    return false;
            return true;
        }
    };
    // Rectangular axis-aligned patches. The sign is the outward owner normal.
    // Geometry only: coefficients and boundary assignments live in Model.
    struct FacePatch {
        CellId owner, neighbor;
        int axis = 0, sign = 1;
        Vec3 min {}, max {};
        double area = 0, owner_distance = 0, neighbor_distance = 0;
        Vec3 center() const { return {(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2}; }
        CellId other(CellId c) const { return c == owner ? neighbor : owner; }
        bool contains(const Vec3& p, double eps = 1e-12) const
        {
            for (int a = 0; a < 3; ++a)
                if (p[a] < min[a] - eps || p[a] > max[a] + eps)
                    return false;
            return true;
        }
    };
    // Geometry, connectivity and TPFA orthogonality are caller preconditions.
    struct DiscreteMesh {
        std::vector<CellGeometry> cells;
        std::vector<FacePatch> faces;
        std::vector<Index> cell_face_offsets;
        std::vector<FaceId> cell_face_ids;
        std::span<const FaceId> incident_faces(CellId c) const
        {
            return std::span(cell_face_ids)
                .subspan(
                    cell_face_offsets[c.value], cell_face_offsets[c.value + 1] - cell_face_offsets[c.value]);
        }
        void build_incidence()
        {
            cell_face_offsets.assign(cells.size() + 1, 0);
            for (const auto& f : faces) {
                ++cell_face_offsets[f.owner.value + 1];
                if (f.neighbor.valid())
                    ++cell_face_offsets[f.neighbor.value + 1];
            }
            std::partial_sum(cell_face_offsets.begin(), cell_face_offsets.end(), cell_face_offsets.begin());
            cell_face_ids.resize(cell_face_offsets.back());
            auto next = cell_face_offsets;
            for (Index i = 0; i < faces.size(); ++i) {
                const auto& f = faces[i];
                cell_face_ids[next[f.owner.value]++] = {i};
                if (f.neighbor.valid())
                    cell_face_ids[next[f.neighbor.value]++] = {i};
            }
        }
        CellId locate(const Vec3& p) const
        {
            // Cache once per probe. Shared-face ties select the lexicographically
            // greatest lower corner, independent of cell storage/DOF ordering.
            CellId result;
            for (Index i = 0; i < cells.size(); ++i)
                if (cells[i].contains(p) && (!result.valid() || cells[i].min > cells[result.value].min))
                    result = {i};
            return result;
        }
    };
} // namespace mhs::core
