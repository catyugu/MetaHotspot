#pragma once
#include "core/vector.hpp"
#include "model/surface.hpp"
#include <ranges>

namespace mhs::model {
    struct CellGeometry {
        Vec3 min {}, max {};
        Vec3 center() const { return {(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2}; }
        Vec3 size() const { return {max[0] - min[0], max[1] - min[1], max[2] - min[2]}; }
        double face_area(std::size_t axis) const
        {
            const auto [u, v] = tangential_axes(static_cast<Axis>(axis));
            return (max[u] - min[u]) * (max[v] - min[v]);
        }
        double volume() const
        {
            auto d = size();
            return d[0] * d[1] * d[2];
        }
    };

    struct ExteriorFacet {
        Facet geometry;
        CellId owner;
        BoundaryGroupId boundary_group;
    };

    struct ConformalMesh {
        std::array<core::Vector<double>, 3> vertices;
        core::Vector<CellId> grid_to_cell; // invalid_id at inactive grid positions
        core::Vector<std::size_t> cell_to_grid;

        std::size_t cell_count() const { return cell_to_grid.size(); }
        std::size_t extent(std::size_t axis) const { return vertices[axis].size() - 1; }
        std::size_t grid_index(std::size_t x, std::size_t y, std::size_t z) const
        {
            return (x * extent(1) + y) * extent(2) + z;
        }
        std::size_t vertex_index(std::size_t x, std::size_t y, std::size_t z) const
        {
            return (x * vertices[1].size() + y) * vertices[2].size() + z;
        }
        Vec3 vertex(std::size_t x, std::size_t y, std::size_t z) const
        {
            return {vertices[0][x], vertices[1][y], vertices[2][z]};
        }
        std::array<std::size_t, 3> cell_coords(std::size_t id) const
        {
            auto index = cell_to_grid[id];
            const auto nz = extent(2), ny = extent(1);
            return {index / nz / ny, index / nz % ny, index % nz};
        }
        CellGeometry cell(std::size_t id) const
        {
            const auto [x, y, z] = cell_coords(id);
            return {vertex(x, y, z), vertex(x + 1, y + 1, z + 1)};
        }
        double cell_width(std::size_t id, std::size_t axis) const
        {
            const auto p = cell_to_grid[id] / stride(axis) % extent(axis);
            return vertices[axis][p + 1] - vertices[axis][p];
        }
        CellId neighbor(std::size_t id, std::size_t axis, int sign) const
        {
            const auto step = stride(axis), index = cell_to_grid[id];
            const auto position = index / step % extent(axis);
            if (sign < 0)
                return position == 0 ? core::invalid_id : grid_to_cell[index - step];
            return position + 1 == extent(axis) ? core::invalid_id : grid_to_cell[index + step];
        }

        core::Vector<ExteriorFacet> exterior_facets; // grouped by owner
        core::Vector<FacetId> boundary_offsets;
        auto boundary_ids(CellId cell) const
        {
            return std::views::iota(boundary_offsets[cell], boundary_offsets[cell + 1]);
        }

    private:
        std::size_t stride(std::size_t axis) const
        {
            return axis == 0 ? extent(1) * extent(2) : axis == 1 ? extent(2) : 1;
        }
    };
} // namespace mhs::model
