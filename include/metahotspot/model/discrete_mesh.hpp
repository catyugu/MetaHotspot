#pragma once
#include "core/vector.hpp"
#include "model/types.hpp"
#include <array>
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

    // One rectangular flux integration patch; positive flux runs minus -> plus.
    // An exterior patch has exactly one invalid cell; its outward sign follows from that side.
    // Bounds follow ascending tangential axes: YZ for X, XZ for Y, XY for Z.
    struct FacePatch {
        CellId minus = core::invalid_id, plus = core::invalid_id;
        Axis axis = Axis::X;
        std::array<double, 4> bounds {};

        bool exterior() const { return minus == core::invalid_id || plus == core::invalid_id; }
        double area() const { return (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]); }
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
        CellId locate(const Vec3& point) const
        {
            CellId result = core::invalid_id;
            for (std::size_t i = 0; i < cells.size(); ++i) {
                const auto& cell = cells[i];
                bool contains = true;
                for (std::size_t axis = 0; axis < 3; ++axis)
                    contains = contains && point[axis] >= cell.min[axis] && point[axis] <= cell.max[axis];
                if (contains && (result == core::invalid_id || cell.min > cells[result].min))
                    result = static_cast<CellId>(i);
            }
            return result;
        }
    };
} // namespace mhs::model
