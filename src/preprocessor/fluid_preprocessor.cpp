#include "common/logger.hpp"
#include "common/mesh_utils.hpp"
#include "common/physics_utils.hpp"
#include "data/tolerance_config.hpp"
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

    namespace {
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
                return c_idx >= 0 && c_idx < (int)model.fluid.is_fluid.size() && model.fluid.is_fluid[c_idx];
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
            for (int fi = 0; fi < model.fluid.n_fluid; ++fi) {
                int old_idx = compact_to_old[model.fluid.fluid_to_global[fi]];
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
                double dh = (cross_w + cross_h > mhs::core::geometry_eps)
                    ? (2.0 * cross_w * cross_h / (cross_w + cross_h))
                    : 0.0;

                model.fluid.hydraulic_diameter[fi] = dh;
                model.fluid.channel_width[fi] = cross_w;
                model.fluid.channel_height[fi] = cross_h;
            }
        }

        // ── 统一的面匹配检测 + BC dispatch ────────────────────────
        // 将 fb 按 kind 路由到 FluidBCParamTable 对应子表,返回 param_idx。
        // 注意: pressure / mass_flow_rate / velocity 是 *per-face-key* 的标量;
        // 它们必须先按 face_key 注册一次,再在 per-cell 匹配中按 *实际匹配 cell 数*
        // 分摊到每个 cell,这样总流量 = 用户给定值 (不会被 cell 数放大)。
        // face_area 是 per-cell 的 (VelocityType 用),仍在内层写。
        static uint16_t registerFluidBCParam(
            mhs::core::FluidBCParamTable& params, const mhs::core::FluidBoundaryOverlay& fb, double per_cell_value)
        {
            switch (fb.kind) {
            case mhs::core::FluidBCType::PressureType:
                // Pressure 是 Dirichlet 值,不按 cell 分摊,直接存原值。
                params.pressure.push_back(per_cell_value);
                return static_cast<uint16_t>(params.pressure.size() - 1);
            case mhs::core::FluidBCType::MassFlowRateType:
                // MassFlowRate 是 *总* 通量, 需在 applyFluidBoundaries 中按匹配 cell 数等分。
                params.mass_flow_rate.push_back(per_cell_value);
                return static_cast<uint16_t>(params.mass_flow_rate.size() - 1);
            case mhs::core::FluidBCType::VelocityType:
                // Velocity 是 per-cell 通量,无需分摊 (直接读 = u * area)。
                params.velocity.push_back(per_cell_value);
                return static_cast<uint16_t>(params.velocity.size() - 1);
            case mhs::core::FluidBCType::None:
            default:
                return static_cast<uint16_t>(mhs::core::invalidIndex);
            }
        }

        // 仅 VelocityType 需要 per-cell face 面积.
        // `side == 'E'` -> positive-direction face (XP/YP/ZP), 'W' -> negative.
        static void cacheFaceAreaForVelocity(std::vector<double>& face_area, int fi, const FaceKeyInfo& fk,
            const mhs::core::MeshGeometry& mesh, int ix, int iy, int iz)
        {
            int axis = (fk.axis == 'X') ? 0 : (fk.axis == 'Y') ? 1 : 2;
            double a = (axis == 0) ? mesh.dy[iy] : mesh.dx[ix];
            double b = (axis == 2) ? mesh.dy[iy] : mesh.dz[iz];
            if (face_area.size() < static_cast<size_t>(fi) + 1)
                face_area.resize(fi + 1, 0.0);
            face_area[fi] = a * b;
        }

        void applyFluidBoundaries(mhs::core::InternalModel& model, const mhs::core::FluidOverlay& overlay,
            double si_scale, const std::vector<int>& compact_to_old)
        {
            const auto& mesh = model.mesh;
            for (const auto& fb : overlay.boundaries) {
                for (const auto& keyStr : fb.face_keys) {
                    FaceKeyInfo fk = parse_face_key(keyStr, si_scale);
                    int target_axis = (fk.axis == 'X') ? 0 : (fk.axis == 'Y') ? 1 : 2;

                    // 第一遍: 仅匹配, 计数; 同时缓存匹配 cell 列表以分配 param_idx.
                    std::vector<int> matched;
                    matched.reserve(64);
                    for (int fi = 0; fi < model.fluid.n_fluid; ++fi) {
                        int old_idx = compact_to_old[model.fluid.fluid_to_global[fi]];
                        int ix, iy, iz;
                        mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

                        double c[3] = {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz]};
                        double d[3] = {mesh.dx[ix], mesh.dy[iy], mesh.dz[iz]};

                        // 检查该流体单元的负向面或正向面是否与给定坐标重合
                        double face_m = c[target_axis] - d[target_axis] * 0.5;
                        double face_p = c[target_axis] + d[target_axis] * 0.5;

                        if (std::abs(face_m - fk.coord_value) < mhs::core::geometry_eps
                            || std::abs(face_p - fk.coord_value) < mhs::core::geometry_eps) {
                            double a = c[(target_axis + 1) % 3];
                            double b = c[(target_axis + 2) % 3];

                            if (point_in_face_rects(fk, a, b) || point_in_face_rects(fk, b, a)) {
                                matched.push_back(fi);
                            }
                        }
                    }

                    if (matched.empty())
                        continue;

                    // MassFlowRate 是 *总* 通量, 按匹配 cell 数等分到每个 cell.
                    // Pressure / Velocity 是 per-cell 量 (pressure Dirichlet 不分摊;
                    // velocity 乘 per-cell area 后本身就是单 cell 通量).
                    double per_cell_value = fb.value;
                    if (fb.kind == mhs::core::FluidBCType::MassFlowRateType) {
                        per_cell_value = fb.value / static_cast<double>(matched.size());
                    }

                    uint16_t param_idx = registerFluidBCParam(model.fluid.fluid_bc_params, fb, per_cell_value);

                    // 第二遍: 把 param_idx 写回每个匹配 cell.
                    for (int fi : matched) {
                        int old_idx = compact_to_old[model.fluid.fluid_to_global[fi]];
                        int ix, iy, iz;
                        mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

                        mhs::core::FluidCellBC cell_bc;
                        cell_bc.kind = fb.kind;
                        cell_bc.param_idx = param_idx;
                        model.fluid.fluid_bcs[fi] = cell_bc;
                        if (!std::isnan(fb.inlet_temperature)) {
                            model.fluid.boundary_temperature_fluid[fi] = fb.inlet_temperature;
                        }
                        if (fb.kind == mhs::core::FluidBCType::VelocityType) {
                            cacheFaceAreaForVelocity(model.fluid.fluid_face_area, fi, fk, mesh, ix, iy, iz);
                        }
                    }
                }
            }
        }

        // 在 cell 中心处评估流体密度, 用作 MassFlowRate 体积通量源 m_dot / rho 的分母.
        // 在初始温度处求值: 不可压缩流的压力解不依赖 T, 用 T_init 是合理近似.
        // 接受 compact_to_old, 因为调用点已经在 solveFluidFlow 中构造了它.
        static double evaluateFluidRhoAtInitT(
            const mhs::core::InternalModel& model, int fi, const std::vector<int>& compact_to_old)
        {
            const auto& cells = model.cells;
            const auto& mesh = model.mesh;
            int c_idx = model.fluid.fluid_to_global[fi];
            int old_idx = compact_to_old[c_idx];
            int ix, iy, iz;
            mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);
            int mat_id = cells.material_id[c_idx];
            const auto& mp = model.material_table[mat_id];
            mhs::core::FieldContext ctx {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], model.initial_temperature, 0.0};
            return mp.rho.eval(ctx);
        }

    } // 匿名命名空间

    void applyFluidOverlay(mhs::core::InternalModel& model, const std::optional<mhs::core::FluidOverlay>& overlay,
        const mhs::core::IOStructure& ioStructure, const mhs::core::SymbolTable& symbols)
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

        model.fluid.is_fluid.assign(N, 0);
        std::vector<double> visc_temp(N, 0.0);

        for (uint16_t matIdx = 0; matIdx < static_cast<uint16_t>(model.material_table.size()); ++matIdx) {
            auto visIt = fluidViscosityMap.find(matNamesByTableIdx[matIdx]);
            if (visIt != fluidViscosityMap.end() && !visIt->second.empty()) {
                model.material_table[matIdx].is_fluid = true;
                model.material_table[matIdx].dynamic_viscosity
                    = mhs::core::parse(substitute_function_args(visIt->second, "T", ioStructure.functions), symbols);
            }
        }

        for (int c = 0; c < N; ++c) {
            uint16_t matIdx = model.cells.material_id[c];
            if (matIdx < model.material_table.size() && model.material_table[matIdx].is_fluid) {
                model.fluid.is_fluid[c] = 1;
                visc_temp[c]
                    = model.material_table[matIdx].dynamic_viscosity.eval({0, 0, 0, model.initial_temperature, 0});
            }
        }

        if (std::none_of(model.fluid.is_fluid.begin(), model.fluid.is_fluid.end(), [](uint8_t v) { return v != 0; }))
            return;

        // 3. 构建索引系统及扩展数据数组
        model.fluid.fluid_to_global.clear();
        model.fluid.global_to_fluid.assign(N, -1);
        for (int c = 0; c < N; ++c) {
            if (model.fluid.is_fluid[c]) {
                model.fluid.global_to_fluid[c] = static_cast<int>(model.fluid.fluid_to_global.size());
                model.fluid.fluid_to_global.push_back(c);
            }
        }
        model.fluid.n_fluid = static_cast<int>(model.fluid.fluid_to_global.size());

        model.fluid.dynamic_viscosity.assign(model.fluid.n_fluid, 0.0);
        model.fluid.pressure.assign(model.fluid.n_fluid, 0.0);
        model.fluid.flow_axes.assign(model.fluid.n_fluid, -1);
        model.fluid.hydroC_x.assign(model.fluid.n_fluid, 0.0);
        model.fluid.hydroC_y.assign(model.fluid.n_fluid, 0.0);
        model.fluid.hydroC_z.assign(model.fluid.n_fluid, 0.0);
        model.fluid.hydraulic_diameter.assign(model.fluid.n_fluid, 0.0);
        model.fluid.channel_width.assign(model.fluid.n_fluid, 0.0);
        model.fluid.channel_height.assign(model.fluid.n_fluid, 0.0);

        // 字典化 BC 容器 + 每单元 cell-level tag
        model.fluid.fluid_bcs.assign(model.fluid.n_fluid, mhs::core::FluidCellBC {});
        model.fluid.fluid_bc_params = mhs::core::FluidBCParamTable {};
        model.fluid.fluid_face_area.assign(model.fluid.n_fluid, 0.0);
        model.fluid.boundary_temperature_fluid.assign(model.fluid.n_fluid, std::numeric_limits<double>::quiet_NaN());

        for (int fi = 0; fi < model.fluid.n_fluid; ++fi) {
            model.fluid.dynamic_viscosity[fi] = visc_temp[model.fluid.fluid_to_global[fi]];
        }

        auto compact_to_old = buildCompactToOld(model.cells, model.mesh.nx * model.mesh.ny * model.mesh.nz);

        // 4. 应用流体边界与计算通道几何
        applyFluidBoundaries(model, overlay.value(), si_scale, compact_to_old);
        computeChannelDimensions(model, compact_to_old);
    }

    void solveFluidFlow(mhs::core::InternalModel& model)
    {
        if (model.fluid.n_fluid == 0)
            return;

        const auto& mesh = model.mesh;
        const auto& cells = model.cells;
        auto compact_to_old = buildCompactToOld(cells, mesh.nx * mesh.ny * mesh.nz);

        // Phase 1: 等效渗透率计算
        for (int fi = 0; fi < model.fluid.n_fluid; ++fi) {
            int old_idx = compact_to_old[model.fluid.fluid_to_global[fi]];
            int ix, iy, iz;
            mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

            double dx = mesh.dx[ix], dy = mesh.dy[iy], dz = mesh.dz[iz];
            double K_perm = std::pow(model.fluid.hydraulic_diameter[fi], 2.0)
                / (2.0 * utils::f_re_rectangular(model.fluid.channel_width[fi], model.fluid.channel_height[fi]));

            double coef = K_perm / model.fluid.dynamic_viscosity[fi];
            model.fluid.hydroC_x[fi] = coef * (dy * dz / dx);
            model.fluid.hydroC_y[fi] = coef * (dx * dz / dy);
            model.fluid.hydroC_z[fi] = coef * (dx * dy / dz);
        }

        // Phase 2: 构建并求解泊松方程 (流场)
        std::vector<Eigen::Triplet<double>> triplets;
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(model.fluid.n_fluid);
        triplets.reserve(model.fluid.n_fluid * 7);

        for (int fi = 0; fi < model.fluid.n_fluid; ++fi) {
            int old_idx = compact_to_old[model.fluid.fluid_to_global[fi]];
            int ix, iy, iz;
            mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

            double diagSum = 0.0;
            for (auto dir : mhs::core::FACE_DIRS) {
                int n_old
                    = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
                if (n_old < 0)
                    continue;

                int n_c = static_cast<int>(cells.index_map[n_old]);
                int fn = (n_c >= 0 && n_c < static_cast<int>(model.fluid.global_to_fluid.size())) ? model.fluid.global_to_fluid[n_c]
                                                                                            : -1;
                if (fn < 0)
                    continue;

                int axis = mhs::utils::AXIS_OF_DIR[static_cast<size_t>(dir)];
                double C_eff = 0.0;
                switch (axis) {
                case 0:
                    C_eff = mhs::utils::harmonicAverage(model.fluid.hydroC_x[fi], model.fluid.hydroC_x[fn]);
                    break;
                case 1:
                    C_eff = mhs::utils::harmonicAverage(model.fluid.hydroC_y[fi], model.fluid.hydroC_y[fn]);
                    break;
                case 2:
                    C_eff = mhs::utils::harmonicAverage(model.fluid.hydroC_z[fi], model.fluid.hydroC_z[fn]);
                    break;
                }
                diagSum += C_eff;

                // PressureType 边界 cell 不与内部耦合; 其它 kind 都保留内部耦合
                if (model.fluid.fluid_bcs[fi].kind != mhs::core::FluidBCType::PressureType) {
                    triplets.emplace_back(fi, fn, -C_eff);
                }
            }

            triplets.emplace_back(fi, fi, diagSum);
            // PressureType: Dirichlet (边界 cell 不与内部耦合, RHS 用压力值)
            if (model.fluid.fluid_bcs[fi].kind == mhs::core::FluidBCType::PressureType) {
                const auto& params = model.fluid.fluid_bc_params.pressure;
                rhs(fi) = params[model.fluid.fluid_bcs[fi].param_idx] * diagSum;
            }
            // MassFlowRateType: Neumann 体通量源 m_dot / rho (符号: + 进入 cell, − 离开).
            // 把 m_dot 作为 Poisson RHS 源项, 让入口有 driving force 推动全场压力梯度,
            // 否则 Pressure=0 出口无流, 装配器侧入口带入的能量无法对流到出口.
            else if (model.fluid.fluid_bcs[fi].kind == mhs::core::FluidBCType::MassFlowRateType) {
                const double mdot_cell = model.fluid.fluid_bc_params.mass_flow_rate[model.fluid.fluid_bcs[fi].param_idx];
                const double rho_cell = evaluateFluidRhoAtInitT(model, fi, compact_to_old);
                // rho==0 在 evaluateFluidRhoAtInitT 中是配置错误; 防御性 fallback.
                rhs(fi) = (rho_cell > mhs::core::zero_guard) ? (mdot_cell / rho_cell) : 0.0;
            }
            // VelocityType: Neumann 体通量源 u * A_face (m/s · m² = m³/s).
            else if (model.fluid.fluid_bcs[fi].kind == mhs::core::FluidBCType::VelocityType) {
                const double vel = model.fluid.fluid_bc_params.velocity[model.fluid.fluid_bcs[fi].param_idx];
                rhs(fi) = vel * model.fluid.fluid_face_area[fi];
            }
        }

        Eigen::SparseMatrix<double> A(model.fluid.n_fluid, model.fluid.n_fluid);
        A.setFromTriplets(triplets.begin(), triplets.end());

        auto solver = mhs::sim::LinearSolver::create(mhs::sim::SolverType::SparseLU);
        auto result = solver->solve(A, rhs);
        if (!result.success) {
            MHS_LOG_WARN("Fluid pressure solve failed (nf={}, nz={})", model.fluid.n_fluid, static_cast<int>(A.nonZeros()));
            return;
        }
        model.fluid.pressure = std::vector<double>(result.solution.data(), result.solution.data() + result.solution.size());
        // Phase 3: 根据压力梯度计算主导流向
        for (int fi = 0; fi < model.fluid.n_fluid; ++fi) {
            int old_idx = compact_to_old[model.fluid.fluid_to_global[fi]];
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
                int fn = (n_c >= 0 && n_c < static_cast<int>(model.fluid.global_to_fluid.size())) ? model.fluid.global_to_fluid[n_c]
                                                                                            : -1;
                if (fn < 0)
                    continue;

                int axis = mhs::utils::AXIS_OF_DIR[static_cast<size_t>(dir)];
                double dp = std::fabs(model.fluid.pressure[fi] - model.fluid.pressure[fn]);
                double flux = 0.0;
                switch (axis) {
                case 0:
                    flux = dp * mhs::utils::harmonicAverage(model.fluid.hydroC_x[fi], model.fluid.hydroC_x[fn]);
                    break;
                case 1:
                    flux = dp * mhs::utils::harmonicAverage(model.fluid.hydroC_y[fi], model.fluid.hydroC_y[fn]);
                    break;
                case 2:
                    flux = dp * mhs::utils::harmonicAverage(model.fluid.hydroC_z[fi], model.fluid.hydroC_z[fn]);
                    break;
                }

                if (flux > maxFlux) {
                    maxFlux = flux;
                    bestAxis = axis;
                }
            }
            model.fluid.flow_axes[fi] = static_cast<int8_t>(bestAxis);
        }
    }

} // namespace mhs::sim