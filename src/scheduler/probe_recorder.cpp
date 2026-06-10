#include "probe_recorder.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <limits>

namespace mhs::sim {

    namespace {

        // 4 个工具（DataPoint / solve_least_squares / get_face_center /
        // extrapolate_face_temperature）从 mhs::post 复制而来，保留算法精度
        // 与边界外推逻辑。探针局部采样 vs 全网格 node 插值是两套不同的入口，
        // 但底层 4 系数平面拟合 + 边界外推是同一份数学；详见 postprocessor.cpp。
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

            Eigen::Vector4d X = A.colPivHouseholderQr().solve(B);
            return X(0);
        }

        // 获取体元指定面的中心世界坐标
        void get_face_center(mhs::core::FaceDir dir, int ix, int iy, int iz, const mhs::core::MeshGeometry& mesh,
            double& fx, double& fy, double& fz)
        {
            fx = mesh.cx[ix];
            fy = mesh.cy[iy];
            fz = mesh.cz[iz];
            if (dir == mhs::core::FaceDir::XM)
                fx -= mesh.dx[ix] / 2.0;
            else if (dir == mhs::core::FaceDir::XP)
                fx += mesh.dx[ix] / 2.0;
            else if (dir == mhs::core::FaceDir::YM)
                fy -= mesh.dy[iy] / 2.0;
            else if (dir == mhs::core::FaceDir::YP)
                fy += mesh.dy[iy] / 2.0;
            else if (dir == mhs::core::FaceDir::ZM)
                fz -= mesh.dz[iz] / 2.0;
            else if (dir == mhs::core::FaceDir::ZP)
                fz += mesh.dz[iz] / 2.0;
        }

        // 计算边界面的表面外推温度
        double extrapolate_face_temperature(mhs::core::FaceDir dir, mhs::core::BcType bc_type, uint16_t param_idx,
            double T_c, double k, const mhs::core::MeshGeometry& mesh, int ix, int iy, int iz,
            const mhs::core::BCParamTable& bc_params, double time)
        {
            double fx, fy, fz;
            get_face_center(dir, ix, iy, iz, mesh, fx, fy, fz);
            mhs::core::FieldContext ctx {fx, fy, fz, T_c, time};

            double half_dist = 0.0;
            if (dir == mhs::core::FaceDir::XM || dir == mhs::core::FaceDir::XP)
                half_dist = mesh.dx[ix] / 2.0;
            else if (dir == mhs::core::FaceDir::YM || dir == mhs::core::FaceDir::YP)
                half_dist = mesh.dy[iy] / 2.0;
            else
                half_dist = mesh.dz[iz] / 2.0;

            if (bc_type == mhs::core::BcType::SecondType) {
                double q = bc_params.neumann_q[param_idx].eval(ctx);
                return T_c + (q * half_dist) / k;
            }
            else if (bc_type == mhs::core::BcType::ThirdType) {
                double h = bc_params.cauchy_h[param_idx].eval(ctx);
                double T_inf = bc_params.cauchy_T_inf[param_idx].eval(ctx);
                double cond_h = k / half_dist;
                return (h * T_inf + cond_h * T_c) / (h + cond_h);
            }
            return T_c;
        }

        // 二分查找 cell 中心数组：返回 cell 下标 lo，使得 cx[lo] ≤ value。
        // 然后验证 value 落在 [cx[lo] - dx[lo]/2, cx[lo] + dx[lo]/2] 内；
        // 边界情况检查下一个 cell 的左半部。越界返回 -1。
        template <typename T> int locate_cell_index(const std::vector<T>& centers, const std::vector<T>& sizes, T value)
        {
            int n = static_cast<int>(centers.size());
            if (n == 0)
                return -1;
            // 域边界：第一个 cell 的左节点 ↔ 最后一个 cell 的右节点
            T lo_bound = centers[0] - sizes[0] * T(0.5);
            T hi_bound = centers[n - 1] + sizes[n - 1] * T(0.5);
            if (value < lo_bound || value > hi_bound)
                return -1;

            // Binary search on centers to find the cell whose center is ≤ value
            int lo = 0, hi = n - 1;
            while (lo < hi) {
                int mid = (lo + hi + 1) / 2;
                if (centers[mid] <= value)
                    lo = mid;
                else
                    hi = mid - 1;
            }
            // lo: largest cell index whose center ≤ value
            T half = sizes[lo] * T(0.5);
            // 上边界（value == centers[lo] + half）归下一个 cell，匹配旧 vertex 二分语义：
            // vertex[lo] ≤ value < vertex[lo+1]（右端点属于下一格）
            if (value >= centers[lo] + half && lo + 1 < n)
                return lo + 1;
            if (value >= centers[lo] - half)
                return lo;
            return lo;
        }
    } // namespace

    void ProbeRecorder::initialize(const mhs::core::InternalModel& model)
    {
        model_ = &model;
        traces_.clear();
        slots_.clear();
        traces_.reserve(model.observation_points.size());
        slots_.reserve(model.observation_points.size());

        for (const auto& op : model.observation_points) {
            mhs::core::ProbeTrace t;
            t.name = op.name;
            traces_.push_back(std::move(t));

            ProbeSlot slot;
            slot.name = op.name;
            slot.px = op.x;
            slot.py = op.y;
            slot.pz = op.z;
            slot.ix = locate_cell_index(model.mesh.cx, model.mesh.dx, op.x);
            slot.iy = locate_cell_index(model.mesh.cy, model.mesh.dy, op.y);
            slot.iz = locate_cell_index(model.mesh.cz, model.mesh.dz, op.z);
            if (slot.ix < 0 || slot.iy < 0 || slot.iz < 0) {
                slot.valid = false;
            }
            else {
                slot.grid_idx = slot.ix * model.mesh.ny * model.mesh.nz + slot.iy * model.mesh.nz + slot.iz;
                slot.valid = (model.cells.valid_mask[slot.grid_idx] != 0);
            }
            slots_.push_back(std::move(slot));
        }
    }

    void ProbeRecorder::record(double time, const std::vector<double>& cell_T)
    {
        if (slots_.empty() || !model_)
            return;

        for (size_t i = 0; i < slots_.size(); ++i) {
            const ProbeSlot& slot = slots_[i];
            double v
                = slot.valid ? sample_one(slot, cell_T, time) : std::numeric_limits<double>::quiet_NaN();
            traces_[i].times.push_back(time);
            traces_[i].values.push_back(v);
        }
    }

    double ProbeRecorder::sample_one(const ProbeSlot& slot, const std::vector<double>& cell_T, double time) const
    {
        const auto& mesh = model_->mesh;
        const auto& cells = model_->cells;
        const int ix = slot.ix;
        const int iy = slot.iy;
        const int iz = slot.iz;
        const int grid_idx = slot.grid_idx;
        const double px = slot.px;
        const double py = slot.py;
        const double pz = slot.pz;

        int compact_idx = static_cast<int>(cells.index_map[grid_idx]);

        // 1. 收集 8 邻接 cell 的 (center, T) 数据点；自身 cell 必定 active（slot.valid 保证）
        //    边界上 ≤ 8 个。
        std::vector<DataPoint> pts;
        pts.reserve(8 + mhs::core::FACE_COUNT);

        double sum_T = 0.0;
        int cnt = 0;
        for (int dx = 0; dx <= 1; ++dx) {
            for (int dy = 0; dy <= 1; ++dy) {
                for (int dz = 0; dz <= 1; ++dz) {
                    int ngx = ix + dx;
                    int ngy = iy + dy;
                    int ngz = iz + dz;
                    if (ngx >= mesh.nx || ngy >= mesh.ny || ngz >= mesh.nz)
                        continue;
                    int ng = ngx * mesh.ny * mesh.nz + ngy * mesh.nz + ngz;
                    if (cells.valid_mask[ng] == 0)
                        continue;
                    sum_T += cell_T[cells.index_map[ng]];
                    ++cnt;
                }
            }
        }
        if (cnt == 0)
            return std::numeric_limits<double>::quiet_NaN();
        const double T_c = sum_T / static_cast<double>(cnt);

        // 2. Dirichlet 面早返回（探针位于 cell 网格边界面上、且该面对应 FirstType）
        for (size_t d = 0; d < mhs::core::FACE_COUNT; ++d) {
            bool on_face = (d == static_cast<size_t>(mhs::core::FaceDir::XM) && ix == 0)
                || (d == static_cast<size_t>(mhs::core::FaceDir::XP) && ix == mesh.nx - 1)
                || (d == static_cast<size_t>(mhs::core::FaceDir::YM) && iy == 0)
                || (d == static_cast<size_t>(mhs::core::FaceDir::YP) && iy == mesh.ny - 1)
                || (d == static_cast<size_t>(mhs::core::FaceDir::ZM) && iz == 0)
                || (d == static_cast<size_t>(mhs::core::FaceDir::ZP) && iz == mesh.nz - 1);
            if (!on_face)
                continue;
            const auto& cell_bc = cells.cell_bcs[compact_idx];
            if (cell_bc.types[d] == mhs::core::BcType::FirstType) {
                uint16_t param_idx = cell_bc.param_idxs[d];
                return model_->bc_params.dirichlet_T[param_idx].eval({px, py, pz, T_c, time});
            }
        }

        // 3. 局部 LSQ：8 cell 中心 + Neumann/Cauchy 面的面中心外推
        const auto& mp = model_->material_table[cells.material_id[compact_idx]];
        double kx_c = mp.kx.eval({px, py, pz, T_c, time});
        double ky_c = mp.ky.eval({px, py, pz, T_c, time});
        double kz_c = mp.kz.eval({px, py, pz, T_c, time});

        for (int dx = 0; dx <= 1; ++dx) {
            for (int dy = 0; dy <= 1; ++dy) {
                for (int dz = 0; dz <= 1; ++dz) {
                    int ngx = ix + dx;
                    int ngy = iy + dy;
                    int ngz = iz + dz;
                    if (ngx >= mesh.nx || ngy >= mesh.ny || ngz >= mesh.nz)
                        continue;
                    int ng = ngx * mesh.ny * mesh.nz + ngy * mesh.nz + ngz;
                    if (cells.valid_mask[ng] == 0)
                        continue;
                    double T_i = cell_T[cells.index_map[ng]];
                    double cdx = mesh.cx[ngx] - px;
                    double cdy = mesh.cy[ngy] - py;
                    double cdz = mesh.cz[ngz] - pz;
                    double dist_k = (cdx * cdx) / kx_c + (cdy * cdy) / ky_c + (cdz * cdz) / kz_c;
                    pts.push_back({mesh.cx[ngx], mesh.cy[ngy], mesh.cz[ngz], T_i, 1.0 / dist_k});
                }
            }
        }

        const auto& cell_bc = cells.cell_bcs[compact_idx];
        for (size_t d = 0; d < mhs::core::FACE_COUNT; ++d) {
            mhs::core::BcType bc = cell_bc.types[d];
            if (bc == mhs::core::BcType::None || bc == mhs::core::BcType::FirstType)
                continue;
            mhs::core::FaceDir dir = mhs::core::FACE_DIRS[d];
            uint16_t param_idx = cell_bc.param_idxs[d];
            double fx, fy, fz;
            get_face_center(dir, ix, iy, iz, mesh, fx, fy, fz);
            double k_face = (dir == mhs::core::FaceDir::XM || dir == mhs::core::FaceDir::XP) ? kx_c
                : (dir == mhs::core::FaceDir::YM || dir == mhs::core::FaceDir::YP)           ? ky_c
                                                                                             : kz_c;
            double T_f = extrapolate_face_temperature(
                dir, bc, param_idx, T_c, k_face, mesh, ix, iy, iz, model_->bc_params, time);

            double fdx = fx - px;
            double fdy = fy - py;
            double fdz = fz - pz;
            double fdist_k = (fdx * fdx) / kx_c + (fdy * fdy) / ky_c + (fdz * fdz) / kz_c;
            pts.push_back({fx, fy, fz, T_f, 1.0 / fdist_k});
        }

        return solve_least_squares(pts, px, py, pz);
    }

} // namespace mhs::sim
