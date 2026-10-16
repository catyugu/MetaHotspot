#pragma once
#include "core/discrete_mesh.hpp"

namespace mhs::core {
    // Presentation-only connectivity. Cartesian XML sampling shape belongs here,
    // never in the thermal topology or the algebraic state space.
    struct OutputMesh {
        std::vector<Vec3> points;
        std::vector<std::array<Index, 8>> cell_vertices;
        std::vector<Index> point_cell_offsets;
        std::vector<CellId> point_cell_ids;
        std::array<Index, 3> cartesian_shape{};
        void build_incidence() {
            point_cell_offsets.assign(points.size() + 1, 0);
            for (const auto &vertices : cell_vertices)
                for (auto v : vertices)
                    ++point_cell_offsets[v + 1];
            std::partial_sum(point_cell_offsets.begin(), point_cell_offsets.end(), point_cell_offsets.begin());
            point_cell_ids.resize(point_cell_offsets.back());
            auto next = point_cell_offsets;
            for (Index c = 0; c < cell_vertices.size(); ++c)
                for (auto v : cell_vertices[c])
                    point_cell_ids[next[v]++] = {c};
        }
        std::span<const CellId> incident_cells(Index p) const {
            return std::span(point_cell_ids)
                .subspan(point_cell_offsets[p], point_cell_offsets[p + 1] - point_cell_offsets[p]);
        }
    };
} // namespace mhs::core
