#include "common/logger.hpp"
#include "common/mesh_utils.hpp"
#include "common/physics_utils.hpp"
#include "expr/expr.hpp"
#include "face_key_processor.hpp"
#include "function_helpers.hpp"
#include "linear_solver/linear_solver.hpp"
#include "preprocessor/fluid_preprocessor.hpp"

#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <limits>

namespace mhs::sim {

    namespace { // 匿名命名空间: 内部辅助函数

        // ── 建立反向映射: Compact index → Old grid index ─────────────────────────
        std::vector<int> buildCompactToOld(const mhs::core::CellFields& cells, int totalGrid)
        {
            std::vector<int> compact_to_old(cells.material_id.size(), -1);
            for (int old_idx = 0; old_idx < totalGrid; ++old_idx) {
                int c = static_cast<int>(cells.index_map[old_idx]);
                if (c >= 0)
                    compact_to_old[c] = old_idx;
            }
            return compact_to_old;
        }

        // ── 沿指定轴探索连续流体长度 (替代原本脆弱且冗长的 6 个 while 循环) ───────────
        double measure_fluid_extent(const mhs::core::InternalModel& model, int ix, int iy, int iz, int axis)
        {
            const auto& mesh = model.mesh;
            const auto& cells = model.cells;
            const int sizes[3] = {mesh.nx, mesh.ny, mesh.nz};

            auto is_fluid_cell = [&](int cx, int cy, int cz) {
                if (cx < 0 || cx >= mesh.nx || cy < 0 || cy >= mesh.ny || cz < 0 || cz >= mesh.nz)
                    return false;
                int old_idx = cx * mesh.ny * mesh.nz + cy * mesh.nz + cz;
                int c_idx = static_cast<int>(cells.index_map[old_idx]);
                return c_idx >= 0 && c_idx < (int)model.is_fluid.size() && model.is_fluid[c_idx];
            };

            int idx[3] = {ix, iy, iz};
            int min_idx = idx[axis], max_idx = idx[axis];

            // 向负方向探索
            while (min_idx > 0) {
                idx[axis] = min_idx - 1;
                if (!is_fluid_cell(idx[0], idx[1], idx[2]))
                    break;
                min_idx--;
            }

            // 向正方向探索
            idx[axis] = max_idx; // reset
            while (max_idx < sizes[axis] - 1) {
                idx[axis] = max_idx + 1;
                if (!is_fluid_cell(idx[0], idx[1], idx[2]))
                    break;
                max_idx++;
            }

            // 计算跨度长度
            const auto& c_array = (axis == 0) ? mesh.cx : (axis == 1) ? mesh.cy : mesh.cz;
            const auto& d_array = (axis == 0) ? mesh.dx : (axis == 1) ? mesh.dy : mesh.dz;
            return (c_array[max_idx] + d_array[max_idx] * 0.5) - (c_array[min_idx] - d_array[min_idx] * 0.5);
        }

        // ── 计算通道几何 ────────────────────────────────────────────────────────
        void computeChannelDimensions(mhs::core::InternalModel& model, const std::vector<int>& compact_to_old)
        {
            const auto& mesh = model.mesh;
            for (int fi = 0; fi < model.n_fluid; ++fi) {
                int old_idx = compact_to_old[model.fluid_to_global[fi]];
                int ix, iy, iz;
                mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

                // 使用通用射线函数分别探测 X, Y, Z 的连通长度
                double lengths[3] = {measure_fluid_extent(model, ix, iy, iz, 0),
                    measure_fluid_extent(model, ix, iy, iz, 1), measure_fluid_extent(model, ix, iy, iz, 2)};

                // 最小的两个维度构成截面的宽和高
                std::sort(lengths, lengths + 3);
                double cross_w = lengths[0];
                double cross_h = lengths[1];

                // 计算水力直径
                double dh = (cross_w + cross_h > 1e-12) ? (2.0 * cross_w * cross_h / (cross_w + cross_h)) : 0.0;

                model.hydraulic_diameter[fi] = dh;
                model.channel_width[fi] = cross_w;
                model.channel_height[fi] = cross_h;
            }
        }

        // ── 统一的面匹配检测 ───────────────────────────────────────────────────
        void applyPressureBoundaries(mhs::core::InternalModel& model, const mhs::core::FluidOverlay& overlay,
            double si_scale, const std::vector<int>& compact_to_old)
        {
            const auto& mesh = model.mesh;
            for (const auto& fb : overlay.boundaries) {
                for (const auto& keyStr : fb.face_keys) {
                    FaceKeyInfo fk = parse_face_key(keyStr, si_scale);
                    int target_axis = (fk.axis == 'X') ? 0 : (fk.axis == 'Y') ? 1 : 2;

                    for (int fi = 0; fi < model.n_fluid; ++fi) {
                        int old_idx = compact_to_old[model.fluid_to_global[fi]];
                        int ix, iy, iz;
                        mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

                        double c[3] = {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz]};
                        double d[3] = {mesh.dx[ix], mesh.dy[iy], mesh.dz[iz]};

                        // 检查该流体单元的负向面或正向面是否与给定坐标重合
                        double face_m = c[target_axis] - d[target_axis] * 0.5;
                        double face_p = c[target_axis] + d[target_axis] * 0.5;

                        if (std::abs(face_m - fk.coord_value) < 1e-10 || std::abs(face_p - fk.coord_value) < 1e-10) {
                            // 检查 2D 矩形范围 (a, b 取另外两个轴的坐标)
                            double a = c[(target_axis + 1) % 3];
                            double b = c[(target_axis + 2) % 3];

                            if (point_in_face_rects(fk, a, b) || point_in_face_rects(fk, b, a)) {
                                model.is_pressure_boundary[fi] = 1;
                                model.boundary_pressure[fi] = fb.pressure_bc.pressure;
                                if (!std::isnan(fb.inlet_temperature)) {
                                    model.boundary_temperature_fluid[fi] = fb.inlet_temperature;
                                }
                            }
                        }
                    }
                }
            }
        }

    } // 匿名命名空间

    void applyFluidOverlay(mhs::core::InternalModel& model, const std::optional<mhs::core::FluidOverlay>& overlay,
        const mhs::core::IOStructure& ioStructure)
    {
        if (!overlay.has_value() || overlay->fluid_materials.empty())
            return;

        const double si_scale = mhs::utils::length_unit_to_si(ioStructure.length_unit);
        const int N = static_cast<int>(model.cells.cell_bcs.size());

        // 1. 建立材质名索引映射
        std::unordered_map<std::string, uint16_t> matNameToTableIdx;
        for (const auto& layer : ioStructure.layers) {
            for (const auto& block : layer.blocks) {
                matNameToTableIdx.emplace(block.material_name, static_cast<uint16_t>(matNameToTableIdx.size()));
            }
        }
        std::vector<std::string> matNamesByTableIdx(matNameToTableIdx.size());
        for (const auto& [name, idx] : matNameToTableIdx) {
            matNamesByTableIdx[idx] = name;
        }

        // 2. 标记流体单元并注册动力粘度
        std::unordered_map<std::string, std::string> fluidViscosityMap;
        for (const auto& fm : overlay->fluid_materials)
            fluidViscosityMap[fm.name] = fm.dynamic_viscosity;

        model.is_fluid.assign(N, 0);
        std::vector<double> visc_temp(N, 0.0);

        for (uint16_t matIdx = 0; matIdx < static_cast<uint16_t>(model.material_table.size()); ++matIdx) {
            auto visIt = fluidViscosityMap.find(matNamesByTableIdx[matIdx]);
            if (visIt != fluidViscosityMap.end() && !visIt->second.empty()) {
                model.material_table[matIdx].is_fluid = true;
                model.material_table[matIdx].dynamic_viscosity
                    = mhs::core::parse(substitute_function_args(visIt->second, "T", ioStructure.functions));
            }
        }

        for (int c = 0; c < N; ++c) {
            uint16_t matIdx = model.cells.material_id[c];
            if (matIdx < model.material_table.size() && model.material_table[matIdx].is_fluid) {
                model.is_fluid[c] = 1;
                visc_temp[c]
                    = model.material_table[matIdx].dynamic_viscosity.eval({0, 0, 0, model.initial_temperature, 0});
            }
        }

        if (std::none_of(model.is_fluid.begin(), model.is_fluid.end(), [](uint8_t v) { return v != 0; }))
            return;

        // 3. 构建索引系统及扩展数据数组
        model.fluid_to_global.clear();
        model.global_to_fluid.assign(N, -1);
        for (int c = 0; c < N; ++c) {
            if (model.is_fluid[c]) {
                model.global_to_fluid[c] = static_cast<int>(model.fluid_to_global.size());
                model.fluid_to_global.push_back(c);
            }
        }
        model.n_fluid = static_cast<int>(model.fluid_to_global.size());

        model.dynamic_viscosity.assign(model.n_fluid, 0.0);
        model.pressure.assign(model.n_fluid, 0.0);
        model.flow_axes.assign(model.n_fluid, -1);
        model.hydroC_x.assign(model.n_fluid, 0.0);
        model.hydroC_y.assign(model.n_fluid, 0.0);
        model.hydroC_z.assign(model.n_fluid, 0.0);
        model.is_pressure_boundary.assign(model.n_fluid, 0);
        model.boundary_pressure.assign(model.n_fluid, 0.0);
        model.boundary_temperature_fluid.assign(model.n_fluid, std::numeric_limits<double>::quiet_NaN());
        model.hydraulic_diameter.assign(model.n_fluid, 0.0);
        model.channel_width.assign(model.n_fluid, 0.0);
        model.channel_height.assign(model.n_fluid, 0.0);

        for (int fi = 0; fi < model.n_fluid; ++fi) {
            model.dynamic_viscosity[fi] = visc_temp[model.fluid_to_global[fi]];
        }

        auto compact_to_old = buildCompactToOld(model.cells, model.mesh.nx * model.mesh.ny * model.mesh.nz);

        // 4. 应用压力边界与计算通道几何
        applyPressureBoundaries(model, overlay.value(), si_scale, compact_to_old);
        computeChannelDimensions(model, compact_to_old);
    }

    void solveFluidFlow(mhs::core::InternalModel& model)
    {
        if (model.n_fluid == 0)
            return;

        const auto& mesh = model.mesh;
        const auto& cells = model.cells;
        auto compact_to_old = buildCompactToOld(cells, mesh.nx * mesh.ny * mesh.nz);

        // Phase 1: 等效渗透率计算
        for (int fi = 0; fi < model.n_fluid; ++fi) {
            int old_idx = compact_to_old[model.fluid_to_global[fi]];
            int ix, iy, iz;
            mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

            double dx = mesh.dx[ix], dy = mesh.dy[iy], dz = mesh.dz[iz];
            double K_perm = std::pow(model.hydraulic_diameter[fi], 2.0)
                / (2.0 * utils::f_re_rectangular(model.channel_width[fi], model.channel_height[fi]));

            double coef = K_perm / model.dynamic_viscosity[fi];
            model.hydroC_x[fi] = coef * (dy * dz / dx);
            model.hydroC_y[fi] = coef * (dx * dz / dy);
            model.hydroC_z[fi] = coef * (dx * dy / dz);
        }

        // Phase 2: 构建并求解泊松方程 (流场)
        std::vector<Eigen::Triplet<double>> triplets;
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(model.n_fluid);
        triplets.reserve(model.n_fluid * 7);

        for (int fi = 0; fi < model.n_fluid; ++fi) {
            if (model.is_pressure_boundary[fi]) {
                triplets.emplace_back(fi, fi, 1.0);
                rhs(fi) = model.boundary_pressure[fi];
                continue;
            }

            int old_idx = compact_to_old[model.fluid_to_global[fi]];
            int ix, iy, iz;
            mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

            double diagSum = 0.0;
            for (auto dir : mhs::core::FACE_DIRS) {
                int n_old
                    = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
                if (n_old < 0)
                    continue;

                int n_c = static_cast<int>(cells.index_map[n_old]);
                int fn = (n_c >= 0 && n_c < static_cast<int>(model.global_to_fluid.size())) ? model.global_to_fluid[n_c]
                                                                                            : -1;
                if (fn < 0)
                    continue;

                double C_eff = 0.0;
                switch (mhs::utils::AXIS_OF_DIR[static_cast<size_t>(dir)]) {
                case 0:
                    C_eff = mhs::utils::harmonicAverage(model.hydroC_x[fi], model.hydroC_x[fn]);
                    break;
                case 1:
                    C_eff = mhs::utils::harmonicAverage(model.hydroC_y[fi], model.hydroC_y[fn]);
                    break;
                case 2:
                    C_eff = mhs::utils::harmonicAverage(model.hydroC_z[fi], model.hydroC_z[fn]);
                    break;
                }

                diagSum += C_eff;
                triplets.emplace_back(fi, fn, -C_eff);
            }
            triplets.emplace_back(fi, fi, diagSum);
        }

        Eigen::SparseMatrix<double> A(model.n_fluid, model.n_fluid);
        A.setFromTriplets(triplets.begin(), triplets.end());

        auto solver = mhs::sim::LinearSolver::create(mhs::sim::SolverType::BiCGSTAB);
        auto result = solver->solve(A, rhs);
        if (!result.success) {
            MHS_LOG_WARN("Fluid pressure solve failed (nf={}, nz={})", model.n_fluid, static_cast<int>(A.nonZeros()));
            return;
        }
        model.pressure = std::vector<double>(result.solution.data(), result.solution.data() + result.solution.size());
        // Phase 3: 根据压力梯度计算主导流向
        for (int fi = 0; fi < model.n_fluid; ++fi) {
            int old_idx = compact_to_old[model.fluid_to_global[fi]];
            int ix, iy, iz;
            mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

            double maxFlux = -1.0;
            int bestAxis = 0;

            for (auto dir : mhs::core::FACE_DIRS) {
                int n_old
                    = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
                if (n_old < 0)
                    continue;

                int n_c = static_cast<int>(cells.index_map[n_old]);
                int fn = (n_c >= 0 && n_c < static_cast<int>(model.global_to_fluid.size())) ? model.global_to_fluid[n_c]
                                                                                            : -1;
                if (fn < 0)
                    continue;

                int axis = mhs::utils::AXIS_OF_DIR[static_cast<size_t>(dir)];
                double dp = std::fabs(model.pressure[fi] - model.pressure[fn]);
                double flux = 0.0;
                switch (axis) {
                case 0:
                    flux = dp * mhs::utils::harmonicAverage(model.hydroC_x[fi], model.hydroC_x[fn]);
                    break;
                case 1:
                    flux = dp * mhs::utils::harmonicAverage(model.hydroC_y[fi], model.hydroC_y[fn]);
                    break;
                case 2:
                    flux = dp * mhs::utils::harmonicAverage(model.hydroC_z[fi], model.hydroC_z[fn]);
                    break;
                }

                if (flux > maxFlux) {
                    maxFlux = flux;
                    bestAxis = axis;
                }
            }
            model.flow_axes[fi] = static_cast<int8_t>(bestAxis);
        }
    }

} // namespace mhs::sim