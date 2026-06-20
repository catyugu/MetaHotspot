#include <Eigen/Sparse>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "assembler.hpp"
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
                    const double kx_n = mp_n.kx.eval(ctx_n);
                    const double ky_n = mp_n.ky.eval(ctx_n);
                    const double kz_n = mp_n.kz.eval(ctx_n);

                    double d_half_neighbor
                        = mhs::utils::half_length_along(dir, mesh.dx[nix], mesh.dy[niy], mesh.dz[niz]);

                    double k_neighbor = mhs::utils::k_along(dir, kx_n, ky_n, kz_n);

                    double cond = 0.0;
                    // Fluid-solid interface: apply Nusselt-based convection correction
                    if (!model_.is_fluid.empty() && model_.is_fluid[c_idx] != model_.is_fluid[n_idx]) {
                        // Identify fluid and solid sides
                        int f_id = model_.is_fluid[c_idx] ? c_idx : n_idx;
                        int f_idx = model_.global_to_fluid[f_id]; // compact fluid index
                        int s_id = model_.is_fluid[c_idx] ? n_idx : c_idx;
                        int f_ax = static_cast<int>(model_.flow_axes[f_idx]);
                        if (f_ax < 0 || f_ax > 2) {
                            double k_face_val = mhs::utils::k_along(dir, kx_c, ky_c, kz_c);
                            cond = A_f / (half_dist / k_face_val + d_half_neighbor / k_neighbor);
                        }
                        else {
                            const auto& mp_f = materials[cells.material_id[f_id]];
                            const mhs::core::FieldContext ctx_f {
                                mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T[f_id], state.current_time};

                            double kf = mhs::utils::k_along(
                                dir, mp_f.kx.eval(ctx_f), mp_f.ky.eval(ctx_f), mp_f.kz.eval(ctx_f));

                            double d_h = model_.hydraulic_diameter[f_idx];
                            double ch_w = model_.channel_width[f_idx];
                            double ch_h = model_.channel_height[f_idx];
                            double Nu = mhs::utils::nusselt_rectangular(ch_w, ch_h);
                            double h_f = Nu * kf / d_h;

                            // Solid-side conduction half-distance
                            // Structured grid: shared face areas are equal on both sides.
                            double half_dist_solid = (s_id == c_idx) ? half_dist : d_half_neighbor;
                            double k_solid = (s_id == c_idx) ? k_face : k_neighbor;
                            double A_solid = A_f;

                            // Series thermal resistance: solid conduction + fluid convection
                            double R = half_dist_solid / (k_solid * A_solid) + 1.0 / (h_f * A_solid);
                            cond = 1.0 / R;
                        }
                    }
                    else {
                        // Standard solid-solid conduction (unchanged)
                        double k_face_val = mhs::utils::k_along(dir, kx_c, ky_c, kz_c);
                        cond = A_f / (half_dist / k_face_val + d_half_neighbor / k_neighbor);
                    }
                    diag += cond;
                    local.triplets.emplace_back(c_idx, n_idx, -cond);
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

            // ── Advection: upwind assembly for fluid-fluid faces ──────────────
            // Inlined into the main loop to avoid a second full-grid traversal
            // (saves TLS allocation + combine_each + sparse merge).
            if (!model_.is_fluid.empty() && c_idx < (int)model_.is_fluid.size() && model_.is_fluid[c_idx]) {
                int f_idx = model_.global_to_fluid[c_idx];
                const double rho_a = materials[cells.material_id[c_idx]].rho.eval(ctx_c);
                const double cp_c = materials[cells.material_id[c_idx]].c.eval(ctx_c);
                double netOutflux = 0.0;

                for (size_t f = 0; f < 6; ++f) {
                    mhs::core::FaceDir dir = mhs::core::FACE_DIRS[f];
                    int neighborOld
                        = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
                    if (neighborOld < 0)
                        continue;

                    int n_idx = (int)cells.index_map[neighborOld];
                    int fn_idx = (n_idx >= 0 && n_idx < (int)model_.global_to_fluid.size())
                        ? model_.global_to_fluid[n_idx]
                        : -1;
                    if (fn_idx < 0)
                        continue;

                    int axis = mhs::utils::AXIS_OF_DIR[f];

                    double hc_a = 0.0, hc_b = 0.0;
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

                    // hc uses full cell length L → half-cell conductance = 2*hc.
                    // Series: 1/(1/(2*hc_a)+1/(2*hc_b)) = 2*hc_a*hc_b/(hc_a+hc_b).
                    double C_eff = 2.0 * hc_a * hc_b / (hc_a + hc_b);

                    int nix = mhs::utils::neighbor_ix(dir, ix);
                    int niy = mhs::utils::neighbor_iy(dir, iy);
                    int niz = mhs::utils::neighbor_iz(dir, iz);
                    const mhs::core::FieldContext ctx_n {
                        mesh.cx[nix], mesh.cy[niy], mesh.cz[niz], state.T[n_idx], state.current_time};
                    double rho_b = materials[cells.material_id[n_idx]].rho.eval(ctx_n);
                    double rho_avg = 0.5 * (rho_a + rho_b);

                    double dP = model_.pressure[f_idx] - model_.pressure[fn_idx];
                    double massFlux = dP * C_eff * rho_avg;
                    netOutflux += massFlux;

                    if (massFlux > 0) {
                        local.triplets.emplace_back(c_idx, c_idx, massFlux * cp_c);
                    }
                    else {
                        double cp_n = materials[cells.material_id[n_idx]].c.eval(ctx_n);
                        local.triplets.emplace_back(c_idx, n_idx, massFlux * cp_n);
                    }
                }

                // Temperature injection / outlet loss
                if (std::fabs(netOutflux) >= 1e-15) {
                    double T_boundary = model_.boundary_temperature_fluid[f_idx];
                    if (!std::isnan(T_boundary)) {
                        local.b(c_idx) += netOutflux * cp_c * T_boundary;
                    }
                    else if (netOutflux < 0) {
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
