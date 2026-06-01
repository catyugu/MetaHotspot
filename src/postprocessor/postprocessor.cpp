#include "postprocessor.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace mhs {

    static double compute_boundary_node_temp(
        const model::InternalModel& model,
        int vx, int vy, int vz,
        const std::vector<double>& cell_temperature,
        double current_time)
    {
        const auto& mesh = model.mesh;
        const auto& cells = model.cells;

        double node_x = mesh.vertex_x[vx];
        double node_y = mesh.vertex_y[vy];
        double node_z = mesh.vertex_z[vz];

        double dirichlet_sum = 0.0;
        int dirichlet_count = 0;
        double other_sum = 0.0;
        int other_count = 0;

        // Check the up-to-8 cells sharing this vertex
        for (int dx = -1; dx <= 0; dx++) {
            int ix = vx + dx;
            if (ix < 0 || ix >= mesh.nx)
                continue;
            for (int dy = -1; dy <= 0; dy++) {
                int iy = vy + dy;
                if (iy < 0 || iy >= mesh.ny)
                    continue;
                for (int dz = -1; dz <= 0; dz++) {
                    int iz = vz + dz;
                    if (iz < 0 || iz >= mesh.nz)
                        continue;

                    int grid_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                    if (cells.valid_mask[grid_idx] == 0)
                        continue;
                    int c_idx = (int)cells.index_map[grid_idx];

                    // Which faces of this cell touch the node?
                    FaceDir dirs[3];
                    int num_dirs = 0;
                    dirs[num_dirs++] = (dx == 0) ? FaceDir::XM : FaceDir::XP;
                    dirs[num_dirs++] = (dy == 0) ? FaceDir::YM : FaceDir::YP;
                    dirs[num_dirs++] = (dz == 0) ? FaceDir::ZM : FaceDir::ZP;

                    for (int i = 0; i < num_dirs; i++) {
                        FaceDir dir = dirs[i];
                        BcType bc_type = cells.cell_bcs[c_idx].types[(size_t)dir];
                        if (bc_type == BcType::None)
                            continue;

                        uint16_t param_idx = cells.cell_bcs[c_idx].param_idxs[(size_t)dir];
                        double T_c = cell_temperature[c_idx];

                        if (bc_type == BcType::FirstType) {
                            dirichlet_sum += model.bc_params.dirichlet_T[param_idx].eval(
                                {node_x, node_y, node_z, T_c, current_time});
                            dirichlet_count++;
                        }
                        else if (bc_type == BcType::SecondType || bc_type == BcType::ThirdType) {
                            double k = model.material_table[cells.material_id[grid_idx]].k.eval(
                                {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], T_c, current_time});

                            double half_dist = 0;
                            if (dir == FaceDir::XM || dir == FaceDir::XP)
                                half_dist = mesh.dx[ix] / 2.0;
                            else if (dir == FaceDir::YM || dir == FaceDir::YP)
                                half_dist = mesh.dy[iy] / 2.0;
                            else
                                half_dist = mesh.dz[iz] / 2.0;

                            double T_f = T_c;
                            if (bc_type == BcType::SecondType) {
                                double q = model.bc_params.neumann_q[param_idx].eval(
                                    {node_x, node_y, node_z, T_c, current_time});
                                T_f = T_c + (q * half_dist) / k; // Fourier's Law
                            }
                            else {
                                double h = model.bc_params.cauchy_h[param_idx].eval(
                                    {node_x, node_y, node_z, T_c, current_time});
                                double T_inf = model.bc_params.cauchy_T_inf[param_idx].eval(
                                    {node_x, node_y, node_z, T_c, current_time});
                                double cond = k / half_dist;
                                T_f = (h * T_inf + cond * T_c) / (h + cond); // Heat flux balance
                            }
                            other_sum += T_f;
                            other_count++;
                        }
                    }
                }
            }
        }

        if (dirichlet_count > 0)
            return dirichlet_sum / dirichlet_count;
        if (other_count > 0)
            return other_sum / other_count;
        return std::numeric_limits<double>::quiet_NaN();
    }

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

        // Phase 1: Distance-weighted interpolation from cells to nodes
        // Each vertex (vx, vy, vz) is shared by up to 8 cells:
        // cells (ix, iy, iz) where ix ∈ {vx-1, vx}, iy ∈ {vy-1, vy}, iz ∈ {vz-1, vz}
        // Weight = 1/d where d = distance from cell center to node
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
                        if (ix < 0 || ix >= mesh.nx)
                            continue;
                        for (int dy = -1; dy <= 0; dy++) {
                            int iy = vy + dy;
                            if (iy < 0 || iy >= mesh.ny)
                                continue;
                            for (int dz = -1; dz <= 0; dz++) {
                                int iz = vz + dz;
                                if (iz < 0 || iz >= mesh.nz)
                                    continue;

                                int cell_grid_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                                if (cells.valid_mask[cell_grid_idx] == 0)
                                    continue;

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
        for (int vx = 0; vx < node_nx; vx++) {
            for (int vy = 0; vy < node_ny; vy++) {
                for (int vz = 0; vz < node_nz; vz++) {
                    int node_idx = vx * node_ny * node_nz + vy * node_nz + vz;
                    double T_bc = compute_boundary_node_temp(model, vx, vy, vz, cell_temperature, 0.0);
                    if (!std::isnan(T_bc)) {
                        node_T[node_idx] = T_bc;
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
            if (std::isnan(v))
                continue;
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
            if (std::isnan(v))
                continue;
            if (v < min_val || std::isnan(min_val)) {
                min_val = v;
            }
        }
        return min_val;
    }

} // namespace mhs