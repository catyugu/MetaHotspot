#include <Eigen/Sparse>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "assembler.hpp"
#include "common/logger.hpp"
#include "common/mesh_utils.hpp"
#include "common/physics_utils.hpp"

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

    AssemblyResult Assembler::assemble(const mhs::core::GlobalState& state) const
    {
        const auto& mesh = model_.mesh;
        const auto& cells = model_.cells;
        const auto& bc_params = model_.bc_params;
        const auto& materials = model_.material_table;

        int N = static_cast<int>(cells.cell_bcs.size());
        int total = mesh.nx * mesh.ny * mesh.nz;

        const std::vector<double>* T_eval_mass = (state.accepted.size() > 0) ? &state.accepted.current() : &state.T;

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
            const mhs::core::FieldContext ctx_c {
                mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T[c_idx], state.current_time};
            const double kx_c = mp.kx.eval(ctx_c);
            const double ky_c = mp.ky.eval(ctx_c);
            const double kz_c = mp.kz.eval(ctx_c);

            // Mass coefficients on accepted.current() — see comment above.
            const mhs::core::FieldContext ctx_m {
                mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], (*T_eval_mass)[c_idx], state.current_time};
            const double rho = mp.rho.eval(ctx_m);
            const double c_heat = mp.c.eval(ctx_m);
            local.mass(c_idx) += rho * c_heat * vol;

            // Heat source dictionary evaluation: uint16_t index into the table.
            uint16_t hs_idx = cells.heat_source_idx[c_idx];
            const double Q = model_.heat_source_table[hs_idx].eval(ctx_c);
            local.b(c_idx) += Q * vol;

            const auto& cell_bc = cells.cell_bcs[c_idx];
            double diag = 0.0;
            bool cell_is_fluid
                = !model_.is_fluid.empty() && c_idx < (int)model_.is_fluid.size() && model_.is_fluid[c_idx];
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
                        mesh.cx[nix], mesh.cy[niy], mesh.cz[niz], state.T[n_idx], state.current_time};
                    double k_neighbor = 0.0;
                    switch (mhs::utils::AXIS_OF_DIR[f]) {
                    case 0:
                        k_neighbor = mp_n.kx.eval(ctx_n);
                        break;
                    case 1:
                        k_neighbor = mp_n.ky.eval(ctx_n);
                        break;
                    case 2:
                        k_neighbor = mp_n.kz.eval(ctx_n);
                        break;
                    }
                    double d_half_neighbor
                        = mhs::utils::half_length_along(dir, mesh.dx[nix], mesh.dy[niy], mesh.dz[niz]);

                    double cond = 0.0;
                    bool n_is_fluid = (n_idx >= 0 && n_idx < (int)model_.is_fluid.size()) && model_.is_fluid[n_idx];

                    // Fluid-solid interface: Nusselt-based convection correction
                    if (cell_is_fluid != n_is_fluid) {
                        int f_id = cell_is_fluid ? c_idx : n_idx;
                        int f_idx = model_.global_to_fluid[f_id];
                        int f_ax = static_cast<int>(model_.flow_axes[f_idx]);
                        if (f_ax < 0 || f_ax > 2) {
                            double k_face_val = mhs::utils::k_along(dir, kx_c, ky_c, kz_c);
                            cond = A_f / (half_dist / k_face_val + d_half_neighbor / k_neighbor);
                        }
                        else {
                            // Reuse already-evaluated k_face/k_neighbor — no need to re-eval.
                            double kf = cell_is_fluid ? k_face : k_neighbor;
                            double d_h = model_.hydraulic_diameter[f_idx];
                            double ch_w = model_.channel_width[f_idx];
                            double ch_h = model_.channel_height[f_idx];
                            double Nu = mhs::utils::nusselt_rectangular(ch_w, ch_h);
                            double h_f = Nu * kf / d_h;
                            double half_dist_solid = cell_is_fluid ? d_half_neighbor : half_dist;
                            double k_solid = cell_is_fluid ? k_neighbor : k_face;
                            double R = half_dist_solid / (k_solid * A_f) + 1.0 / (h_f * A_f);
                            cond = 1.0 / R;
                        }
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
                        int f_idx = model_.global_to_fluid[c_idx];
                        int fn_idx = model_.global_to_fluid[n_idx];
                        if (f_idx >= 0 && fn_idx >= 0) {
                            int axis = mhs::utils::AXIS_OF_DIR[f];
                            double hc_a, hc_b;
                            switch (axis) {
                            case 0:
                                hc_a = model_.hydroC_x[f_idx];
                                hc_b = model_.hydroC_x[fn_idx];
                                break;
                            case 1:
                                hc_a = model_.hydroC_y[f_idx];
                                hc_b = model_.hydroC_y[fn_idx];
                                break;
                            default:
                                hc_a = model_.hydroC_z[f_idx];
                                hc_b = model_.hydroC_z[fn_idx];
                                break;
                            }
                            if (hc_a > 1e-30 && hc_b > 1e-30) {
                                double C_eff = mhs::utils::harmonicConductance(hc_a, hc_b);
                                double rho_b = mp_n.rho.eval(ctx_n);
                                double rho_avg = 0.5 * (rho_a + rho_b);
                                double dP = model_.pressure[f_idx] - model_.pressure[fn_idx];
                                double massFlux = dP * C_eff * rho_avg;
                                netOutflux += massFlux;
                                if (std::fabs(massFlux) > 1e-30) {
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

            // netOutflux > 0  → 流体进入域内 (Inlet)  → 可选用 T_boundary
            // netOutflux < 0  → 流体流出域外 (Outlet) → 强制内部迎风
            if (cell_is_fluid && std::fabs(netOutflux) >= 1e-12) {
                int f_idx = model_.global_to_fluid[c_idx];
                if (netOutflux > 0) { // 流体进入 (Inlet)
                    double T_boundary = model_.boundary_temperature_fluid[f_idx];
                    if (!std::isnan(T_boundary)) {
                        local.b(c_idx) += netOutflux * cp_c * T_boundary;
                    }
                    else if (f_idx < (int)model_.is_pressure_boundary.size() && model_.is_pressure_boundary[f_idx]) {
                        MHS_LOG_WARN("Fluid enters near cell {}, no InletTemperature — 0K.", c_idx);
                    }
                }
                else if (netOutflux < 0) { // 流体流出 (Outlet)
                    local.triplets.emplace_back(c_idx, c_idx, -netOutflux * cp_c);
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
