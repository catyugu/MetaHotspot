#pragma once

#include "model/constants.hpp"
#include "model/types.hpp"
#include <array>
#include <cmath>

namespace mhs::model {
    using Vec3 = std::array<double, 3>;

    // Tangential bounds are ordered YZ for X, XZ for Y and XY for Z.
    struct Facet {
        Axis axis = Axis::X;
        std::int8_t normal = 0; // -1/+1 for an exterior facet, 0 for interior
        double coordinate = 0.0;
        std::array<double, 4> bounds {};

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
} // namespace mhs::model
