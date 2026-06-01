#include "postprocessor.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace mhs {

    std::vector<double> Postprocessor::interpolate_cell_to_node(
        const model::InternalModel& model,
        const std::vector<double>& cell_temperature) const
    {
        const auto& mesh = model.mesh;
        const auto& cells = model.cells;

        // Node count: (nx+1) * (ny+1) * (nz+1) vertices
        int node_nx = mesh.nx + 1;
        int node_ny = mesh.ny + 1;
        int node_nz = mesh.nz + 1;
        int total_nodes = node_nx * node_ny * node_nz;

        std::vector<double> node_T(total_nodes, std::numeric_limits<double>::quiet_NaN());

        // Each vertex (vx, vy, vz) is shared by up to 8 cells:
        // cells (ix, iy, iz) where ix ∈ {vx-1, vx}, iy ∈ {vy-1, vy}, iz ∈ {vz-1, vz}
        // Distance-weighted interpolation: weight = 1/d where d = distance from cell center to node
        for (int vx = 0; vx < node_nx; vx++) {
            for (int vy = 0; vy < node_ny; vy++) {
                for (int vz = 0; vz < node_nz; vz++) {
                    int node_idx = vx * node_ny * node_nz + vy * node_nz + vz;
                    double node_x = mesh.vertex_x[vx];
                    double node_y = mesh.vertex_y[vy];
                    double node_z = mesh.vertex_z[vz];

                    double weighted_sum = 0.0;
                    double weight_sum = 0.0;

                    // Check all 8 neighboring cells
                    for (int dx = -1; dx <= 0; dx++) {
                        int ix = vx + dx;
                        if (ix < 0 || ix >= mesh.nx) continue;
                        for (int dy = -1; dy <= 0; dy++) {
                            int iy = vy + dy;
                            if (iy < 0 || iy >= mesh.ny) continue;
                            for (int dz = -1; dz <= 0; dz++) {
                                int iz = vz + dz;
                                if (iz < 0 || iz >= mesh.nz) continue;

                                int cell_grid_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                                if (cells.valid_mask[cell_grid_idx] == 0) continue;

                                int compact_idx = (int)cells.index_map[cell_grid_idx];

                                double dist = std::sqrt(
                                    (mesh.cx[ix] - node_x) * (mesh.cx[ix] - node_x)
                                    + (mesh.cy[iy] - node_y) * (mesh.cy[iy] - node_y)
                                    + (mesh.cz[iz] - node_z) * (mesh.cz[iz] - node_z));

                                double w = 1.0 / dist;
                                weighted_sum += w * cell_temperature[compact_idx];
                                weight_sum += w;
                            }
                        }
                    }

                    if (weight_sum > 0.0) {
                        node_T[node_idx] = weighted_sum / weight_sum;
                    }
                }
            }
        }

        return node_T;
    }

    double Postprocessor::max_temperature(const std::vector<double>& T) const
    {
        if (T.empty()) {
            return 0.0;
        }
        double max_val = T[0];
        for (const auto& v : T) {
            if (std::isnan(v)) continue;
            if (v > max_val || std::isnan(max_val)) {
                max_val = v;
            }
        }
        return max_val;
    }

    double Postprocessor::min_temperature(const std::vector<double>& T) const
    {
        if (T.empty()) {
            return 0.0;
        }
        double min_val = T[0];
        for (const auto& v : T) {
            if (std::isnan(v)) continue;
            if (v < min_val || std::isnan(min_val)) {
                min_val = v;
            }
        }
        return min_val;
    }

} // namespace mhs