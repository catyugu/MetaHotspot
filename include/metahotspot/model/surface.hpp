#pragma once

#include "model/constants.hpp"
#include "model/types.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <optional>

namespace mhs::model {
    using Vec3 = std::array<double, 3>;

    using Rect = std::array<double, 4>; // [a_min, a_max, b_min, b_max]

    inline bool positive_rect(const Rect& r) { return r[0] < r[1] && r[2] < r[3]; }
    inline double rect_area(const Rect& r) { return (r[1] - r[0]) * (r[3] - r[2]); }

    inline std::optional<Rect> rect_intersection(const Rect& a, const Rect& b)
    {
        const Rect r {std::max(a[0], b[0]), std::min(a[1], b[1]), std::max(a[2], b[2]), std::min(a[3], b[3])};
        return positive_rect(r) ? std::optional {r} : std::nullopt;
    }

    inline bool rect_contains(const Rect& r, double a, double b)
    {
        return a >= r[0] - geometry_eps && a <= r[1] + geometry_eps && b >= r[2] - geometry_eps
            && b <= r[3] + geometry_eps;
    }

    inline std::array<std::size_t, 2> tangential_axes(Axis axis)
    {
        return {axis == Axis::X ? 1u : 0u, axis == Axis::Z ? 1u : 2u};
    }

    // Tangential bounds are ordered YZ for X, XZ for Y and XY for Z.
    struct Facet {
        Axis axis = Axis::X;
        std::int8_t normal = 0; // -1/+1 for an exterior facet, 0 for interior
        double coordinate = 0.0;
        Rect bounds {};

        double area() const { return rect_area(bounds); }
        Vec3 center() const
        {
            const auto a = static_cast<std::size_t>(axis);
            Vec3 p {};
            p[a] = coordinate;
            const auto [u, v] = tangential_axes(axis);
            p[u] = (bounds[0] + bounds[1]) / 2;
            p[v] = (bounds[2] + bounds[3]) / 2;
            return p;
        }
        bool contains(const Vec3& p) const
        {
            const auto a = static_cast<std::size_t>(axis);
            const auto [u, v] = tangential_axes(axis);
            return std::abs(p[a] - coordinate) <= geometry_eps && rect_contains(bounds, p[u], p[v]);
        }
    };
} // namespace mhs::model
