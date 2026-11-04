#pragma once
#include "core/vector.hpp"
#include "model/types.hpp"
#include <array>
#include <span>

namespace mhs::model {
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
        FacePatch project(const std::array<double, 4>& bounds) const
        {
            auto patch = *this;
            const int a = axis == 0 ? 1 : 0, b = axis == 2 ? 1 : 2;
            patch.min[a] = bounds[0];
            patch.max[a] = bounds[1];
            patch.min[b] = bounds[2];
            patch.max[b] = bounds[3];
            patch.area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]);
            return patch;
        }
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
        mhs::core::Vector<CellGeometry> cells;
        mhs::core::Vector<FacePatch> faces;
        mhs::core::Vector<Index> cell_face_offsets;
        mhs::core::Vector<FaceId> cell_face_ids;
        std::span<const FaceId> incident_faces(CellId c) const
        {
            return std::span(cell_face_ids)
                .subspan(cell_face_offsets[c.value], cell_face_offsets[c.value + 1] - cell_face_offsets[c.value]);
        }
        void build_incidence();
        CellId locate(const Vec3& p) const;
    };
} // namespace mhs::model
