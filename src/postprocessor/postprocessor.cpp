#include "postprocessor.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace mhs {

    std::vector<double> Postprocessor::interpolate_cell_to_node(
        const InternalModel& model, const std::vector<double>& cell_temperature) const
    {
        const auto& mesh = model.mesh;
        const auto& cells = model.cells;

        int node_nx = mesh.nx + 1;
        int node_ny = mesh.ny + 1;
        int node_nz = mesh.nz + 1;
        int total_nodes = node_nx * node_ny * node_nz;

        std::vector<double> node_T(total_nodes, std::numeric_limits<double>::quiet_NaN());

        for (int vx = 0; vx < node_nx; vx++) {
            for (int vy = 0; vy < node_ny; vy++) {
                for (int vz = 0; vz < node_nz; vz++) {
                    int node_idx = vx * node_ny * node_nz + vy * node_nz + vz;
                    double node_x = mesh.vertex_x[vx];
                    double node_y = mesh.vertex_y[vy];
                    double node_z = mesh.vertex_z[vz];

                    double weighted_sum = 0.0;
                    double weight_sum = 0.0;

                    double dirichlet_sum = 0.0;
                    int dirichlet_count = 0;

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
                                double T_c = cell_temperature[compact_idx];

                                double k = model.material_table[cells.material_id[cell_grid_idx]].k.eval(
                                    {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], T_c, 0.0});

                                double dx_dist = mesh.cx[ix] - node_x;
                                double dy_dist = mesh.cy[iy] - node_y;
                                double dz_dist = mesh.cz[iz] - node_z;
                                double dist2 = dx_dist * dx_dist + dy_dist * dy_dist + dz_dist * dz_dist;

                                double w = k / dist2;

                                FaceDir dirs[3];
                                int num_dirs = 0;
                                dirs[num_dirs++] = (dx == 0) ? FaceDir::XM : FaceDir::XP;
                                dirs[num_dirs++] = (dy == 0) ? FaceDir::YM : FaceDir::YP;
                                dirs[num_dirs++] = (dz == 0) ? FaceDir::ZM : FaceDir::ZP;

                                double local_extrap_sum = 0.0;
                                int extrap_count = 0;

                                // 2. Check for BC overrides touching this specific node
                                for (int i = 0; i < num_dirs; i++) {
                                    FaceDir dir = dirs[i];
                                    BcType bc_type = cells.cell_bcs[compact_idx].types[(size_t)dir];
                                    if (bc_type == BcType::None)
                                        continue;

                                    uint16_t param_idx = cells.cell_bcs[compact_idx].param_idxs[(size_t)dir];

                                    if (bc_type == BcType::FirstType) {
                                        dirichlet_sum += model.bc_params.dirichlet_T[param_idx].eval(
                                            {node_x, node_y, node_z, T_c, 0.0});
                                        dirichlet_count++;
                                    }
                                    else if (bc_type == BcType::SecondType || bc_type == BcType::ThirdType) {
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
                                                {node_x, node_y, node_z, T_c, 0.0});
                                            T_f = T_c + (q * half_dist) / k;
                                        }
                                        else {
                                            double h = model.bc_params.cauchy_h[param_idx].eval(
                                                {node_x, node_y, node_z, T_c, 0.0});
                                            double T_inf = model.bc_params.cauchy_T_inf[param_idx].eval(
                                                {node_x, node_y, node_z, T_c, 0.0});
                                            double cond_h = k / half_dist;
                                            T_f = (h * T_inf + cond_h * T_c) / (h + cond_h);
                                        }
                                        local_extrap_sum += T_f;
                                        extrap_count++;
                                    }
                                }

                                double T_contributed = T_c;
                                if (extrap_count > 0) {
                                    T_contributed = local_extrap_sum / extrap_count;
                                }

                                weighted_sum += w * T_contributed;
                                weight_sum += w;
                            }
                        }
                    }

                    // 3. Dirichlet wins absolutely. Otherwise, blend using k/V weights.
                    if (dirichlet_count > 0) {
                        node_T[node_idx] = dirichlet_sum / dirichlet_count;
                    }
                    else if (weight_sum > 0.0) {
                        node_T[node_idx] = weighted_sum / weight_sum;
                    }
                }
            }
        }

        return node_T;
    }

    double Postprocessor::max_temperature(const std::vector<double>& T) const
    {
        if (T.empty())
            return 0.0;
        double max_val = T[0];
        for (const auto& v : T) {
            if (std::isnan(v))
                continue;
            if (v > max_val || std::isnan(max_val))
                max_val = v;
        }
        return max_val;
    }

    double Postprocessor::min_temperature(const std::vector<double>& T) const
    {
        if (T.empty())
            return 0.0;
        double min_val = T[0];
        for (const auto& v : T) {
            if (std::isnan(v))
                continue;
            if (v < min_val || std::isnan(min_val))
                min_val = v;
        }
        return min_val;
    }

} // namespace mhs