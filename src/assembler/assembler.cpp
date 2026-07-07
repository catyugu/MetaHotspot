#include <Eigen/Sparse>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "assembler.hpp"
#include "common/mesh_utils.hpp"
#include "common/physics_utils.hpp"
#include "data/tolerance_config.hpp"

namespace mhs::sim {

    namespace { // anonymous: file-private helpers
        // Per-thread scratch for the TBB parallel_for over grid cells.
        struct ThreadLocalData {
            std::vector<Eigen::Triplet<double>> triplets;
            Eigen::VectorXd b;
            Eigen::VectorXd mass;
            explicit ThreadLocalData(int N) : b(Eigen::VectorXd::Zero(N)), mass(Eigen::VectorXd::Zero(N)) { }
        };
    } // namespace

    AssemblyResult Assembler::assemble(const AssembleContext& ctx) const
    {
        const auto& mesh = model_.mesh;
        const auto& cells = model_.cells;
        const auto& bc_params = model_.bc_params;
        const auto& materials = model_.material_table;

        int N = static_cast<int>(cells.cell_bcs.size());
        int total = mesh.nx * mesh.ny * mesh.nz;

        auto thread_data = tbb::enumerable_thread_specific<ThreadLocalData>([&]() { return ThreadLocalData(N); });

        tbb::parallel_for(0, total, [&](int old_idx) {
            if (cells.index_map[old_idx] == mhs::core::invalidIndex)
                return;

            auto& local = thread_data.local();
            int ix, iy, iz;
            mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

            int c_idx = (int)cells.index_map[old_idx];
            double dx_cell = mesh.dx[ix];
            double dy_cell = mesh.dy[iy];
            double dz_cell = mesh.dz[iz];
            double vol = dx_cell * dy_cell * dz_cell;

            // 缓存 cell 上下文：kx/ky/kz 共享一次构造，BC 分支也复用同一 ctx。
            size_t mat_id = cells.material_id[c_idx];
            const auto& mp = materials[mat_id];
            const mhs::core::FieldContext ctx_c {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], ctx.T[c_idx], ctx.current_time};
            const double kx_c = mp.kx.eval(ctx_c);
            const double ky_c = mp.ky.eval(ctx_c);
            const double kz_c = mp.kz.eval(ctx_c);

            // Mass coefficients evaluated at the same temperature field as the
            // diffusion terms.
            const mhs::core::FieldContext ctx_m {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], ctx.T[c_idx], ctx.current_time};
            const double rho = mp.rho.eval(ctx_m);
            const double c_heat = mp.c.eval(ctx_m);
            local.mass(c_idx) += rho * c_heat * vol;

            // Heat source dictionary evaluation: uint16_t index into the table.
            uint16_t hs_idx = cells.heat_source_idx[c_idx];
            const double Q = model_.heat_source_table[hs_idx].eval(ctx_c);
            local.b(c_idx) += Q * vol;

            const auto& cell_bc = cells.cell_bcs[c_idx];
            double diag = 0.0;
            bool cell_is_fluid = !model_.fluid.is_fluid.empty() && c_idx < (int)model_.fluid.is_fluid.size()
                && model_.fluid.is_fluid[c_idx];
            const double rho_a = cell_is_fluid ? materials[cells.material_id[c_idx]].rho.eval(ctx_c) : 0.0;
            const double cp_c = cell_is_fluid ? materials[cells.material_id[c_idx]].c.eval(ctx_c) : 0.0;
            double netOutflux = 0.0;

            for (size_t f = 0; f < mhs::core::FACE_COUNT; f++) {
                mhs::core::FaceDir dir = mhs::core::FACE_DIRS[f];
                const double A_f = mhs::utils::face_area(dir, dx_cell, dy_cell, dz_cell);
                const double half_dist = mhs::utils::half_length_along(dir, dx_cell, dy_cell, dz_cell);
                const double k_face = mhs::utils::k_along(dir, kx_c, ky_c, kz_c);
                mhs::core::BcType bc_type = cell_bc.types[f];
                uint16_t param_idx = cell_bc.param_idxs[f];

                if (bc_type == mhs::core::BcType::None) {
                    int neighbor_old
                        = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
                    if (neighbor_old < 0)
                        continue;

                    int n_idx = (int)cells.index_map[neighbor_old];
                    int nix = mhs::utils::neighbor_ix(dir, ix);
                    int niy = mhs::utils::neighbor_iy(dir, iy);
                    int niz = mhs::utils::neighbor_iz(dir, iz);

                    const auto& mp_n = materials[cells.material_id[n_idx]];
                    const mhs::core::FieldContext ctx_n {
                        mesh.cx[nix], mesh.cy[niy], mesh.cz[niz], ctx.T[n_idx], ctx.current_time};
                    double k_neighbor
                        = utils::k_along(dir, mp_n.kx.eval(ctx_n), mp_n.ky.eval(ctx_n), mp_n.kz.eval(ctx_n));

                    double d_half_neighbor
                        = mhs::utils::half_length_along(dir, mesh.dx[nix], mesh.dy[niy], mesh.dz[niz]);

                    double cond = 0.0;
                    bool n_is_fluid
                        = (n_idx >= 0 && n_idx < (int)model_.fluid.is_fluid.size()) && model_.fluid.is_fluid[n_idx];

                    // Fluid-solid interface: Nusselt-based convection correction
                    if (cell_is_fluid != n_is_fluid) {
                        int f_id = cell_is_fluid ? c_idx : n_idx;
                        int f_idx = model_.fluid.global_to_fluid[f_id];
                        // Reuse already-evaluated k_face/k_neighbor — no need to re-eval.
                        double kf = cell_is_fluid ? k_face : k_neighbor;
                        double d_h = model_.fluid.hydraulic_diameter[f_idx];
                        double ch_w = model_.fluid.channel_width[f_idx];
                        double ch_h = model_.fluid.channel_height[f_idx];
                        double Nu = mhs::utils::nusselt_rectangular(ch_w, ch_h);
                        double h_f = Nu * kf / d_h;
                        double half_dist_solid = cell_is_fluid ? d_half_neighbor : half_dist;
                        double k_solid = cell_is_fluid ? k_neighbor : k_face;
                        double R = half_dist_solid / (k_solid * A_f) + 1.0 / (h_f * A_f);
                        cond = 1.0 / R;
                    }
                    else {
                        // Standard solid-solid or fluid-fluid conduction
                        double k_face_val = mhs::utils::k_along(dir, kx_c, ky_c, kz_c);
                        cond = A_f / (half_dist / k_face_val + d_half_neighbor / k_neighbor);
                    }
                    diag += cond;
                    local.triplets.emplace_back(c_idx, n_idx, -cond);

                    // Advection: upwind mass flux for fluid-fluid faces
                    if (cell_is_fluid && n_is_fluid) {
                        int f_idx = model_.fluid.global_to_fluid[c_idx];
                        int fn_idx = model_.fluid.global_to_fluid[n_idx];
                        if (f_idx >= 0 && fn_idx >= 0) {
                            int axis = mhs::utils::AXIS_OF_DIR[f];
                            const auto& hc = model_.fluid.hydroC[axis];
                            double hc_a = hc[f_idx];
                            double hc_b = hc[fn_idx];
                            if (hc_a > mhs::core::zero_guard && hc_b > mhs::core::zero_guard) {
                                double C_eff = mhs::utils::harmonicAverage(hc_a, hc_b);
                                double rho_b = mp_n.rho.eval(ctx_n);
                                double rho_avg = 0.5 * (rho_a + rho_b);
                                double dP = model_.fluid.pressure[f_idx] - model_.fluid.pressure[fn_idx];
                                double massFlux = dP * C_eff * rho_avg;
                                netOutflux += massFlux;
                                if (std::fabs(massFlux) > mhs::core::zero_guard) {
                                    if (massFlux > 0) {
                                        local.triplets.emplace_back(c_idx, c_idx, massFlux * cp_c);
                                    }
                                    else {
                                        double cp_n = mp_n.c.eval(ctx_n);
                                        local.triplets.emplace_back(c_idx, n_idx, massFlux * cp_n);
                                    }
                                }
                            }
                        }
                    }
                }
                else if (bc_type == mhs::core::BcType::FirstType) {
                    double T_bc_val = bc_params.dirichlet_T[param_idx].eval(ctx_c);
                    double cond = k_face * A_f / half_dist;
                    diag += cond;
                    local.b(c_idx) += cond * T_bc_val;
                }
                else if (bc_type == mhs::core::BcType::SecondType) {
                    double q = bc_params.neumann_q[param_idx].eval(ctx_c);
                    local.b(c_idx) += q * A_f;
                }
                else if (bc_type == mhs::core::BcType::ThirdType) {
                    double h = bc_params.cauchy_h[param_idx].eval(ctx_c);
                    double T_inf = bc_params.cauchy_T_inf[param_idx].eval(ctx_c);
                    double coeff = k_face * h * A_f / (k_face + h * half_dist);
                    diag += coeff;
                    local.b(c_idx) += coeff * T_inf;
                }
            }

            local.triplets.emplace_back(c_idx, c_idx, diag);

            if (cell_is_fluid) {
                int f_idx = model_.fluid.global_to_fluid[c_idx];
                // MassFlowRateType / VelocityType cell 用 BC 值覆盖
                // pressure 差分得到的 netOutflux, 保证质量守恒由用户给定值驱动.
                // 该分支对 PressureType 不触发 — 其仍走 pressure-driven 路径。
                const auto& bc = model_.fluid.fluid_bcs[f_idx];
                if (bc.kind == mhs::core::FluidBCType::MassFlowRateType) {
                    netOutflux = model_.fluid.fluid_bc_params.mass_flow_rate[bc.param_idx];
                }
                else if (bc.kind == mhs::core::FluidBCType::VelocityType) {
                    netOutflux
                        = model_.fluid.fluid_bc_params.velocity[bc.param_idx] * model_.fluid.fluid_face_area[f_idx];
                }

                if (netOutflux != 0.0) {
                    double T_boundary = model_.fluid.boundary_temperature_fluid[f_idx];

                    // 情况 A：流体流入 (Inlet) -> 作为源项加入右端向量 (RHS)
                    if (netOutflux > 0.0 && !std::isnan(T_boundary)) {
                        local.b(c_idx) += netOutflux * cp_c * T_boundary;
                    }
                    // 情况 B：流体流出 (Outlet) -> 隐式处理，加入对角线矩阵 (LHS)
                    else {
                        local.triplets.emplace_back(c_idx, c_idx, -netOutflux * cp_c);
                    }
                }
            }
        });

        std::vector<Eigen::Triplet<double>> triplets;
        Eigen::VectorXd b = Eigen::VectorXd::Zero(N);
        Eigen::VectorXd M_diag = Eigen::VectorXd::Zero(N);

        thread_data.combine_each([&](const ThreadLocalData& local) {
            triplets.insert(triplets.end(), local.triplets.begin(), local.triplets.end());
            b += local.b;
            M_diag += local.mass;
        });

        Eigen::SparseMatrix<double> K(N, N);
        K.setFromTriplets(triplets.begin(), triplets.end());

        return {std::move(K), std::move(b), std::move(M_diag)};
    }

} // namespace mhs::sim
