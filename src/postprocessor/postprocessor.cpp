#include "common/mesh_utils.hpp"
#include "common/sample_point.hpp"
#include "postprocessor.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace mhs::post {

    std::vector<double> interpolate_cell_to_node(
        const mhs::core::InternalModel& model, const std::vector<double>& cell_temperature, double time)
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
                    // 节点坐标从 cx/dx 重建：vx=0 为左边界（左节点），vx=nx 为右边界（右节点）
                    double node_x = (vx == 0) ? mesh.cx[0] - mesh.dx[0] * 0.5 : mesh.cx[vx - 1] + mesh.dx[vx - 1] * 0.5;
                    double node_y = (vy == 0) ? mesh.cy[0] - mesh.dy[0] * 0.5 : mesh.cy[vy - 1] + mesh.dy[vy - 1] * 0.5;
                    double node_z = (vz == 0) ? mesh.cz[0] - mesh.dz[0] * 0.5 : mesh.cz[vz - 1] + mesh.dz[vz - 1] * 0.5;

                    double dirichlet_sum = 0.0;
                    int dirichlet_count = 0;

                    std::vector<mhs::utils::SampleDataPoint> pts;

                    // 遍历该节点周边相接的最多 8 个体元
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

                                double cx = mesh.cx[ix];
                                double cy = mesh.cy[iy];
                                double cz = mesh.cz[iz];
                                const auto& mp = model.material_table[cells.material_id[compact_idx]];
                                mhs::core::FieldContext ctx {cx, cy, cz, T_c, time};
                                double kx_c = mp.kx.eval(ctx);
                                double ky_c = mp.ky.eval(ctx);
                                double kz_c = mp.kz.eval(ctx);
                                double c_dx = cx - node_x;
                                double c_dy = cy - node_y;
                                double c_dz = cz - node_z;
                                double dist_k = (c_dx * c_dx) / kx_c + (c_dy * c_dy) / ky_c + (c_dz * c_dz) / kz_c;
                                double w_cell = 1.0 / dist_k;
                                pts.push_back({cx, cy, cz, T_c, w_cell});

                                // 确认当前体元连接到该节点的 3 个面
                                mhs::core::FaceDir f_x = (dx == -1) ? mhs::core::FaceDir::XP : mhs::core::FaceDir::XM;
                                mhs::core::FaceDir f_y = (dy == -1) ? mhs::core::FaceDir::YP : mhs::core::FaceDir::YM;
                                mhs::core::FaceDir f_z = (dz == -1) ? mhs::core::FaceDir::ZP : mhs::core::FaceDir::ZM;

                                mhs::core::FaceDir dirs[3] = {f_x, f_y, f_z};

                                // 检查面上的边界条件
                                for (mhs::core::FaceDir dir : dirs) {
                                    mhs::core::BcType bc_type = cells.cell_bcs[compact_idx].types[(size_t)dir];
                                    if (bc_type == mhs::core::BcType::None)
                                        continue;

                                    uint16_t param_idx = cells.cell_bcs[compact_idx].param_idxs[(size_t)dir];

                                    if (bc_type == mhs::core::BcType::FirstType) {
                                        dirichlet_sum += model.bc_params.dirichlet_T[param_idx].eval(
                                            {node_x, node_y, node_z, T_c, time});
                                        dirichlet_count++;
                                    }
                                    else {
                                        // 梯度边界：外推面中心温度并作为一个额外的"几何观测点"喂给最小二乘求解器
                                        double k_face = mhs::utils::k_along(dir, kx_c, ky_c, kz_c);
                                        double T_f = mhs::utils::sample_extrapolate_face_temperature(dir, bc_type,
                                            param_idx, T_c, k_face, mesh, ix, iy, iz, model.bc_params, time);

                                        // 【各向异性修正】同理，计算面中心到节点的等效各向异性距离权重
                                        double fx, fy, fz;
                                        mhs::utils::sample_face_center(dir, ix, iy, iz, mesh, fx, fy, fz);
                                        double f_dx = fx - node_x;
                                        double f_dy = fy - node_y;
                                        double f_dz = fz - node_z;
                                        double fdist_k
                                            = (f_dx * f_dx) / kx_c + (f_dy * f_dy) / ky_c + (f_dz * f_dz) / kz_c;
                                        double w_face = 1.0 / fdist_k;
                                        pts.push_back({fx, fy, fz, T_f, w_face});
                                    }
                                }
                            }
                        }
                    }

                    // 物理强条件：Dirichlet 边界具有无条件决定权
                    if (dirichlet_count > 0) {
                        node_T[node_idx] = dirichlet_sum / dirichlet_count;
                    }
                    else if (!pts.empty()) {
                        // 利用最小二乘求解器精确拟合
                        node_T[node_idx] = mhs::utils::sample_solve_least_squares(pts, node_x, node_y, node_z);
                    }
                }
            }
        }

        return node_T;
    }

    double max_temperature(const std::vector<double>& T)
    {
        if (T.empty())
            return 0.0;
        double max_val = std::numeric_limits<double>::quiet_NaN();
        for (double v : T) {
            if (std::isnan(v))
                continue;
            if (std::isnan(max_val) || v > max_val)
                max_val = v;
        }
        return std::isnan(max_val) ? 0.0 : max_val;
    }

    double min_temperature(const std::vector<double>& T)
    {
        if (T.empty())
            return 0.0;
        double min_val = std::numeric_limits<double>::quiet_NaN();
        for (double v : T) {
            if (std::isnan(v))
                continue;
            if (std::isnan(min_val) || v < min_val)
                min_val = v;
        }
        return std::isnan(min_val) ? 0.0 : min_val;
    }

} // namespace mhs::post
