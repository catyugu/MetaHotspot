#pragma once
#include "core/vector.hpp"
#include "model/constants.hpp"
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

    // One geometrically unique rectangular flux integration region.
    // Positive normal points from minus to plus; an exterior facet has one invalid cell.
    // Bounds follow ascending tangential axes: YZ for X, XZ for Y, XY for Z.
    struct Facet {
        CellId minus = core::invalid_id, plus = core::invalid_id;
        Axis axis = Axis::X;
        double coordinate = 0.0;
        std::array<double, 4> bounds {};

        bool exterior() const { return minus == core::invalid_id || plus == core::invalid_id; }
        CellId owner() const { return minus != core::invalid_id ? minus : plus; }
        int outward_sign() const { return minus != core::invalid_id ? 1 : -1; }
        double area() const { return (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]); }
        Vec3 center() const
        {
            const auto a = static_cast<std::size_t>(axis);
            Vec3 p {};
            p[a] = coordinate;
            p[a == 0 ? 1 : 0] = (bounds[0] + bounds[1]) / 2;
            p[a == 2 ? 1 : 2] = (bounds[2] + bounds[3]) / 2;
            return p;
        }
        bool contains(const Vec3& p) const
        {
            constexpr double eps = geometry_eps;
            const auto a = static_cast<std::size_t>(axis);
            const auto u = a == 0 ? 1 : 0, v = a == 2 ? 1 : 2;
            const bool on_plane = std::abs(p[a] - coordinate) <= eps;
            const bool inside_u = p[u] >= bounds[0] - eps && p[u] <= bounds[1] + eps;
            const bool inside_v = p[v] >= bounds[2] - eps && p[v] <= bounds[3] + eps;
            return on_plane && inside_u && inside_v;
        }
    };

    // Canonical integration geometry and cell-facet incidence. A physical boundary
    // can be partitioned for multiple physics without duplicating facet geometry.
    struct DiscreteMesh {
        mhs::core::Vector<CellGeometry> cells;
        mhs::core::Vector<Facet> facets;
        mhs::core::Vector<std::size_t> cell_facet_offsets;
        mhs::core::Vector<FacetId> cell_facet_ids;
        std::span<const FacetId> incident_facets(CellId c) const
        {
            return std::span(cell_facet_ids)
                .subspan(cell_facet_offsets[c], cell_facet_offsets[c + 1] - cell_facet_offsets[c]);
        }
    };
} // namespace mhs::model
