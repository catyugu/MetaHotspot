#pragma once
#include "core/vector.hpp"
#include "model/constants.hpp"
#include "model/types.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <variant>

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

    // Derived rectangle geometry, returned by value and never cached on connections.
    struct FaceGeometry {
        Axis axis = Axis::X;
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

    struct BoundaryFacet : FaceGeometry {
        CellId cell = core::invalid_id;
        int side = 1;
    };

    struct Connection {
        CellId a, b;
    };

    struct ConformalMesh {
        core::Vector<double> x_vertices, y_vertices, z_vertices;
        const core::Vector<double>& axis(int a) const { return a == 0 ? x_vertices : a == 1 ? y_vertices : z_vertices; }
        // Both mappings are empty for a fully active grid. Z varies fastest.
        core::Vector<CellId> logical_to_cell;
        core::Vector<std::size_t> cell_to_logical;
        std::array<std::size_t, 3> shape() const
        {
            return {x_vertices.empty() ? 0 : x_vertices.size() - 1, y_vertices.empty() ? 0 : y_vertices.size() - 1,
                z_vertices.empty() ? 0 : z_vertices.size() - 1};
        }
    };

    struct NonConformalMesh {
        core::Vector<CellGeometry> cells;
        core::Vector<Connection> connections;
    };
    using Mesh = std::variant<ConformalMesh, NonConformalMesh>;

    inline std::size_t cell_count(const ConformalMesh& mesh)
    {
        const auto n = mesh.shape();
        return mesh.logical_to_cell.empty() ? n[0] * n[1] * n[2] : mesh.cell_to_logical.size();
    }
    inline std::size_t cell_count(const NonConformalMesh& mesh) { return mesh.cells.size(); }
    inline std::size_t cell_count(const Mesh& mesh)
    {
        return std::visit([](const auto& m) { return cell_count(m); }, mesh);
    }
    inline CellGeometry cell_geometry(const NonConformalMesh& mesh, CellId cell) { return mesh.cells[cell]; }
    inline CellGeometry cell_geometry(const ConformalMesh& mesh, CellId cell)
    {
        const auto n = mesh.shape();
        auto logical = mesh.logical_to_cell.empty() ? static_cast<std::size_t>(cell) : mesh.cell_to_logical[cell];
        const std::array<std::size_t, 3> index {logical / (n[1] * n[2]), logical / n[2] % n[1], logical % n[2]};
        CellGeometry result;
        for (int a = 0; a < 3; ++a) {
            result.min[a] = mesh.axis(a)[index[a]];
            result.max[a] = mesh.axis(a)[index[a] + 1];
        }
        return result;
    }
    inline CellGeometry cell_geometry(const Mesh& mesh, CellId cell)
    {
        return std::visit([&](const auto& m) { return cell_geometry(m, cell); }, mesh);
    }
    template <typename Callback> void for_each_connection(const NonConformalMesh& mesh, Callback&& callback)
    {
        for (const auto connection : mesh.connections)
            callback(connection);
    }
    template <typename Callback> void for_each_connection(const ConformalMesh& mesh, Callback&& callback)
    {
        const auto n = mesh.shape();
        const std::array<std::size_t, 3> stride {n[1] * n[2], n[2], 1};
        for (std::size_t c = 0; c < cell_count(mesh); ++c) {
            const auto logical = mesh.logical_to_cell.empty() ? c : mesh.cell_to_logical[c];
            for (int a = 0; a < 3; ++a) {
                if ((logical / stride[a]) % n[a] + 1 == n[a])
                    continue;
                const auto next = logical + stride[a];
                const auto other
                    = mesh.logical_to_cell.empty() ? static_cast<CellId>(next) : mesh.logical_to_cell[next];
                if (other != core::invalid_id)
                    callback(Connection {static_cast<CellId>(c), other});
            }
        }
    }
    template <typename Callback> void for_each_connection(const Mesh& mesh, Callback&& callback)
    {
        std::visit([&](const auto& m) { for_each_connection(m, callback); }, mesh);
    }

    // A valid contact has positive tangential overlap and opposite coplanar faces.
    inline FaceGeometry connection_geometry(const CellGeometry& a, const CellGeometry& b)
    {
        for (int axis = 0; axis < 3; ++axis) {
            if (std::abs(a.max[axis] - b.min[axis]) > geometry_eps
                && std::abs(b.max[axis] - a.min[axis]) > geometry_eps)
                continue;
            const int u = axis == 0 ? 1 : 0, v = axis == 2 ? 1 : 2;
            FaceGeometry result;
            result.axis = static_cast<Axis>(axis);
            result.coordinate = std::abs(a.max[axis] - b.min[axis]) <= geometry_eps ? a.max[axis] : a.min[axis];
            result.bounds = {std::max(a.min[u], b.min[u]), std::min(a.max[u], b.max[u]), std::max(a.min[v], b.min[v]),
                std::min(a.max[v], b.max[v])};
            if (result.bounds[1] > result.bounds[0] && result.bounds[3] > result.bounds[2])
                return result;
        }
        throw std::invalid_argument("Connection requires positive-area face contact between Cartesian cells");
    }
} // namespace mhs::model
