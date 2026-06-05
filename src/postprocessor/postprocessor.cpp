#include "postprocessor.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <vector>

namespace mhs {

    namespace {
        struct DataPoint {
            double x, y, z;
            double T;
            double weight;
        };

        // 最小二乘法求解节点温度 (拟合 T(x,y,z) = T_node + gx*x + gy*y + gz*z)
        double solve_least_squares(const std::vector<DataPoint>& pts, double node_x, double node_y, double node_z)
        {
            int M = static_cast<int>(pts.size());
            if (M == 0)
                return std::numeric_limits<double>::quiet_NaN();
            if (M == 1)
                return pts[0].T;

            // 矩阵维度: M个数据点 + 3个正则化方程
            Eigen::MatrixXd A(M + 3, 4);
            Eigen::VectorXd B(M + 3);
            A.setZero();
            B.setZero();

            double sum_w = 0.0;
            for (int i = 0; i < M; ++i) {
                double sqrt_w = std::sqrt(pts[i].weight);
                A(i, 0) = sqrt_w;
                A(i, 1) = sqrt_w * (pts[i].x - node_x);
                A(i, 2) = sqrt_w * (pts[i].y - node_y);
                A(i, 3) = sqrt_w * (pts[i].z - node_z);
                B(i) = sqrt_w * pts[i].T;
                sum_w += pts[i].weight;
            }

            // Tikhonov 正则化：
            double reg_w = std::sqrt(sum_w * 1e-6);
            A(M, 1) = reg_w;
            A(M + 1, 2) = reg_w;
            A(M + 2, 3) = reg_w;

            // 使用 Householder QR 鲁棒求解超定/欠定方程组
            Eigen::Vector4d X = A.colPivHouseholderQr().solve(B);

            // X(0) 即为多项式在节点 (dx=0, dy=0, dz=0) 处的截距，也即求得的节点温度
            return X(0);
        }

        // 获取体元指定面的中心世界坐标
        void get_face_center(
            FaceDir dir, int ix, int iy, int iz, const MeshGeometry& mesh, double& fx, double& fy, double& fz)
        {
            fx = mesh.cx[ix];
            fy = mesh.cy[iy];
            fz = mesh.cz[iz];
            if (dir == FaceDir::XM)
                fx -= mesh.dx[ix] / 2.0;
            else if (dir == FaceDir::XP)
                fx += mesh.dx[ix] / 2.0;
            else if (dir == FaceDir::YM)
                fy -= mesh.dy[iy] / 2.0;
            else if (dir == FaceDir::YP)
                fy += mesh.dy[iy] / 2.0;
            else if (dir == FaceDir::ZM)
                fz -= mesh.dz[iz] / 2.0;
            else if (dir == FaceDir::ZP)
                fz += mesh.dz[iz] / 2.0;
        }

        // 计算边界面的表面外推温度
        double extrapolate_face_temperature(FaceDir dir, BcType bc_type, uint16_t param_idx, double T_c, double k,
            const MeshGeometry& mesh, int ix, int iy, int iz, const BCParamTable& bc_params)
        {
            double fx, fy, fz;
            get_face_center(dir, ix, iy, iz, mesh, fx, fy, fz);
            FieldContext ctx {fx, fy, fz, T_c, 0.0};

            double half_dist = 0.0;
            if (dir == FaceDir::XM || dir == FaceDir::XP)
                half_dist = mesh.dx[ix] / 2.0;
            else if (dir == FaceDir::YM || dir == FaceDir::YP)
                half_dist = mesh.dy[iy] / 2.0;
            else
                half_dist = mesh.dz[iz] / 2.0;

            if (bc_type == BcType::SecondType) {
                double q = bc_params.neumann_q[param_idx].eval(ctx);
                return T_c + (q * half_dist) / k;
            }
            else if (bc_type == BcType::ThirdType) {
                double h = bc_params.cauchy_h[param_idx].eval(ctx);
                double T_inf = bc_params.cauchy_T_inf[param_idx].eval(ctx);
                double cond_h = k / half_dist;
                return (h * T_inf + cond_h * T_c) / (h + cond_h);
            }
            return T_c;
        }

        // 二分查找 vertex 数组：返回 cell 下标 lo，使得 vertex[lo] ≤ value < vertex[lo+1]。
        // 越界（value < vertex[0] 或 value > vertex[n-1]）时返回 -1，由调用方判定。
        template <typename T> int locate_cell_index(const std::vector<T>& vertex, T value)
        {
            int n = static_cast<int>(vertex.size());
            if (n < 2)
                return -1;
            if (value < vertex.front() || value > vertex.back())
                return -1;
            int lo = 0, hi = n - 1;
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (vertex[mid] <= value)
                    lo = mid;
                else
                    hi = mid;
            }
            return lo;
        }
    }

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

                    double dirichlet_sum = 0.0;
                    int dirichlet_count = 0;

                    std::vector<DataPoint> pts;

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

                                // 提取该体元的热导率
                                double k = model.material_table[cells.material_id[cell_grid_idx]].k.eval(
                                    {cx, cy, cz, T_c, 0.0});

                                // 体元中心点入参 (采用反距离平方为基准权重)
                                double dist2 = (cx - node_x) * (cx - node_x) + (cy - node_y) * (cy - node_y)
                                    + (cz - node_z) * (cz - node_z);
                                double w_cell = k / (dist2 + 1e-16);
                                pts.push_back({cx, cy, cz, T_c, w_cell});

                                // 确认当前体元连接到该节点的 3 个面
                                FaceDir f_x = (dx == -1) ? FaceDir::XP : FaceDir::XM;
                                FaceDir f_y = (dy == -1) ? FaceDir::YP : FaceDir::YM;
                                FaceDir f_z = (dz == -1) ? FaceDir::ZP : FaceDir::ZM;

                                FaceDir dirs[3] = {f_x, f_y, f_z};

                                // 检查面上的边界条件
                                for (FaceDir dir : dirs) {
                                    BcType bc_type = cells.cell_bcs[compact_idx].types[(size_t)dir];
                                    if (bc_type == BcType::None)
                                        continue;

                                    uint16_t param_idx = cells.cell_bcs[compact_idx].param_idxs[(size_t)dir];

                                    if (bc_type == BcType::FirstType) {
                                        dirichlet_sum += model.bc_params.dirichlet_T[param_idx].eval(
                                            {node_x, node_y, node_z, T_c, 0.0});
                                        dirichlet_count++;
                                    }
                                    else {
                                        // 梯度边界：外推面中心温度并作为一个额外的"几何观测点"喂给最小二乘求解器
                                        double fx, fy, fz;
                                        get_face_center(dir, ix, iy, iz, mesh, fx, fy, fz);
                                        double T_f = extrapolate_face_temperature(
                                            dir, bc_type, param_idx, T_c, k, mesh, ix, iy, iz, model.bc_params);

                                        double fdist2 = (fx - node_x) * (fx - node_x) + (fy - node_y) * (fy - node_y)
                                            + (fz - node_z) * (fz - node_z);
                                        double w_face = k / (fdist2 + 1e-16);
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
                        node_T[node_idx] = solve_least_squares(pts, node_x, node_y, node_z);
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
        double max_val = std::numeric_limits<double>::quiet_NaN();
        for (double v : T) {
            if (std::isnan(v))
                continue;
            if (std::isnan(max_val) || v > max_val)
                max_val = v;
        }
        return std::isnan(max_val) ? 0.0 : max_val;
    }

    double Postprocessor::min_temperature(const std::vector<double>& T) const
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

    double Postprocessor::sample_point(
        const std::vector<double>& node_T, const InternalModel& model, const ProbePoint& point) const
    {
        const auto& mesh = model.mesh;
        const auto& cells = model.cells;

        const double px = point.x;
        const double py = point.y;
        const double pz = point.z;

        // 在 vertex 数组中定位包围 cell：vertex[ix] ≤ px < vertex[ix+1]，对 Y/Z 同理。
        int ix = locate_cell_index(mesh.vertex_x, px);
        int iy = locate_cell_index(mesh.vertex_y, py);
        int iz = locate_cell_index(mesh.vertex_z, pz);
        if (ix < 0 || iy < 0 || iz < 0)
            return std::numeric_limits<double>::quiet_NaN();

        int cell_grid_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
        if (cells.valid_mask[cell_grid_idx] == 0)
            return std::numeric_limits<double>::quiet_NaN();

        int compact_idx = static_cast<int>(cells.index_map[cell_grid_idx]);

        // 单次遍历 cell 8 顶点：同时累加 T_c、收集 (vertex, node_T) 用于后续 LSQ。
        // Dirichlet 早返回和 LSQ 回退路径都依赖 T_c。
        const int node_ny = mesh.ny + 1;
        const int node_nz = mesh.nz + 1;
        struct Corner {
            double vx, vy, vz;
            double Tv;
        };
        Corner corners[8];
        int cnt_node = 0;
        double sum_node = 0.0;
        for (int dx = 0; dx <= 1; ++dx)
            for (int dy = 0; dy <= 1; ++dy)
                for (int dz = 0; dz <= 1; ++dz) {
                    int nidx = (ix + dx) * node_ny * node_nz + (iy + dy) * node_nz + (iz + dz);
                    double Tv = (nidx >= 0 && nidx < static_cast<int>(node_T.size()))
                        ? node_T[nidx]
                        : std::numeric_limits<double>::quiet_NaN();
                    corners[(dx << 2) | (dy << 1) | dz]
                        = {mesh.vertex_x[ix + dx], mesh.vertex_y[iy + dy], mesh.vertex_z[iz + dz], Tv};
                    if (!std::isnan(Tv)) {
                        sum_node += Tv;
                        ++cnt_node;
                    }
                }
        if (cnt_node == 0)
            return std::numeric_limits<double>::quiet_NaN();
        const double T_c = sum_node / cnt_node;

        // 探针位于 cell 哪个边界面上（最多两个面），由 ix/iy/iz 端点直接判定。
        // 仅 FirstType (Dirichlet) 触发早返回；Neumann/Cauchy 在 LSQ 中作为外推观测。
        for (size_t d = 0; d < FACE_COUNT; ++d) {
            bool on_face = (d == static_cast<size_t>(FaceDir::XM) && ix == 0)
                || (d == static_cast<size_t>(FaceDir::XP) && ix == mesh.nx - 1)
                || (d == static_cast<size_t>(FaceDir::YM) && iy == 0)
                || (d == static_cast<size_t>(FaceDir::YP) && iy == mesh.ny - 1)
                || (d == static_cast<size_t>(FaceDir::ZM) && iz == 0)
                || (d == static_cast<size_t>(FaceDir::ZP) && iz == mesh.nz - 1);
            if (!on_face)
                continue;
            BcType bc = cells.cell_bcs[compact_idx].types[d];
            if (bc == BcType::FirstType) {
                uint16_t param_idx = cells.cell_bcs[compact_idx].param_idxs[d];
                return model.bc_params.dirichlet_T[param_idx].eval({px, py, pz, T_c, 0.0});
            }
        }

        // LSQ 数据点：cell 8 顶点 + Neumann/Cauchy 面的面中心外推观测。
        double k = model.material_table[cells.material_id[cell_grid_idx]].k.eval(
            {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], T_c, 0.0});

        std::vector<DataPoint> pts;
        pts.reserve(8 + FACE_COUNT);
        for (const auto& c : corners) {
            double Tv = std::isnan(c.Tv) ? T_c : c.Tv;
            double dist2 = (c.vx - px) * (c.vx - px) + (c.vy - py) * (c.vy - py) + (c.vz - pz) * (c.vz - pz);
            pts.push_back({c.vx, c.vy, c.vz, Tv, k / (dist2 + 1e-16)});
        }
        for (size_t d = 0; d < FACE_COUNT; ++d) {
            BcType bc = cells.cell_bcs[compact_idx].types[d];
            if (bc == BcType::None || bc == BcType::FirstType)
                continue;
            FaceDir dir = FACE_DIRS[d];
            uint16_t param_idx = cells.cell_bcs[compact_idx].param_idxs[d];
            double fx, fy, fz;
            get_face_center(dir, ix, iy, iz, mesh, fx, fy, fz);
            double T_f = extrapolate_face_temperature(dir, bc, param_idx, T_c, k, mesh, ix, iy, iz, model.bc_params);
            double fdist2 = (fx - px) * (fx - px) + (fy - py) * (fy - py) + (fz - pz) * (fz - pz);
            pts.push_back({fx, fy, fz, T_f, k / (fdist2 + 1e-16)});
        }

        return solve_least_squares(pts, px, py, pz);
    }

} // namespace mhs