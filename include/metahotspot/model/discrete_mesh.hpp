#pragma once
#include "core/vector.hpp"
#include "model/types.hpp"
#include <array>
#include <cmath>
#include <span>

namespace mhs::model {
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
    };

    // One physical rectangular flux interface, including coarse/fine subfaces.
    // Bounds follow ascending tangential axes: YZ for X, XZ for Y, XY for Z.
    struct FacePatch {
        CellId owner = core::invalid_id, neighbor = core::invalid_id;
        Axis axis = Axis::X;
        double coordinate = 0;
        std::array<double, 4> bounds {};

        CellId other(CellId c) const { return c == owner ? neighbor : owner; }
        static double area(const std::array<double, 4>& b) { return (b[1] - b[0]) * (b[3] - b[2]); }
        double area() const { return area(bounds); }
        Vec3 center(const std::array<double, 4>& b) const;
        Vec3 center() const { return center(bounds); }
        double distance(const CellGeometry& cell) const
        { return std::abs(coordinate - cell.center()[static_cast<std::size_t>(axis)]); }
        int normal_sign(const CellGeometry& cell) const
        { return coordinate > cell.center()[static_cast<std::size_t>(axis)] ? 1 : -1; }
        bool contains(const Vec3& point, const std::array<double, 4>& b, double eps = 1e-12) const;
    };

    // Compiled geometry and connectivity; compilers build the incidence index.
    // Geometry, connectivity and TPFA orthogonality are caller preconditions.
    struct DiscreteMesh {
        mhs::core::Vector<CellGeometry> cells;
        mhs::core::Vector<FacePatch> faces;
        mhs::core::Vector<std::size_t> cell_face_offsets;
        mhs::core::Vector<FaceId> cell_face_ids;
        std::span<const FaceId> incident_faces(CellId c) const
        {
            return std::span(cell_face_ids)
                .subspan(cell_face_offsets[c], cell_face_offsets[c + 1] - cell_face_offsets[c]);
        }
    };
} // namespace mhs::model
