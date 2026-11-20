#pragma once
#include "core/vector.hpp"
#include "model/surface.hpp"
#include <span>

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

    // Positive axis points from minus to plus. An exterior facet has one missing cell.
    struct CellFacet {
        CellId minus = core::invalid_id, plus = core::invalid_id;

        bool exterior() const { return minus == core::invalid_id || plus == core::invalid_id; }
        CellId owner() const { return minus != core::invalid_id ? minus : plus; }
    };

    // Canonical integration geometry and cell-facet incidence. A physical boundary
    // can be partitioned for multiple physics without duplicating facet geometry.
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
        std::array<std::size_t, 3> cell_coords(std::size_t id) const
        {
            auto index = cell_to_grid[id];
            const auto nz = extent(2), ny = extent(1);
            return {index / nz / ny, index / nz % ny, index % nz};
        }
        CellGeometry cell(std::size_t id) const
        {
            const auto p = cell_coords(id);
            return {{vertices[0][p[0]], vertices[1][p[1]], vertices[2][p[2]]},
                {vertices[0][p[0] + 1], vertices[1][p[1] + 1], vertices[2][p[2] + 1]}};
        }
        CellId neighbor(std::size_t id, std::size_t axis, int sign) const
        {
            const auto stride = axis == 0 ? extent(1) * extent(2) : axis == 1 ? extent(2) : 1;
            const auto index = cell_to_grid[id];
            const auto position = index / stride % extent(axis);
            if (sign < 0)
                return position == 0 ? core::invalid_id : grid_to_cell[index - stride];
            return position + 1 == extent(axis) ? core::invalid_id : grid_to_cell[index + stride];
        }

        mhs::core::Vector<Facet> facets;
        mhs::core::Vector<CellFacet> cell_facets; // same FacetId as facets
        mhs::core::Vector<std::size_t> cell_facet_offsets;
        mhs::core::Vector<FacetId> cell_facet_ids;
        std::span<const FacetId> incident_facets(CellId c) const
        {
            return std::span(cell_facet_ids)
                .subspan(cell_facet_offsets[c], cell_facet_offsets[c + 1] - cell_facet_offsets[c]);
        }
    };
} // namespace mhs::model
