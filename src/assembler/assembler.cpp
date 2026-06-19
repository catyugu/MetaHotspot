#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <Eigen/Sparse>

#include "assembler.hpp"
#include "common/mesh_utils.hpp"
#include "common/physics_utils.hpp"

namespace mhs::sim {

    namespace { // anonymous: file-private helpers
        void decode_index(int old_idx, int ny, int nz, int& ix, int& iy, int& iz)
        {
            ix = old_idx / (ny * nz);
            iy = (old_idx % (ny * nz)) / nz;
            iz = old_idx % nz;
        }

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
            decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

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
                    if (N > 0 && c_idx >= 0 && c_idx < N && n_idx >= 0 && n_idx < N
                        && model_.is_fluid.size() == static_cast<size_t>(N)
                        && model_.is_fluid[c_idx] != model_.is_fluid[n_idx]) {
                        // Identify fluid and solid sides
                        int f_id = model_.is_fluid[c_idx] ? c_idx : n_idx;
                        int s_id = model_.is_fluid[c_idx] ? n_idx : c_idx;
                        int f_ax = static_cast<int>(model_.flow_axes[f_id]);
                        if (f_ax < 0 || f_ax > 2) {
                            // Fallback to standard conduction if flow_axes not yet computed
                            double k_face_val = mhs::utils::k_along(dir, kx_c, ky_c, kz_c);
                            cond = A_f / (half_dist / k_face_val + d_half_neighbor / k_neighbor);
                        }
                        else {
                            // Fluid-side thermal conductivity along flow axis
                            const auto& mp_f = materials[cells.material_id[f_id]];
                            const mhs::core::FieldContext ctx_f {mesh.cx[(f_ax == 0) ? ix : ((f_ax == 1) ? iy : iz)],
                                mesh.cy[(f_ax == 1) ? iy : ((f_ax == 0 || f_ax == 2) ? iz : ix)],
                                mesh.cz[(f_ax == 2) ? iz : ((f_ax == 0) ? iy : ix)], 0.0, 0.0};
                            double kf = mhs::utils::k_along(static_cast<mhs::core::FaceDir>(f_ax), mp_f.kx.eval(ctx_f),
                                mp_f.ky.eval(ctx_f), mp_f.kz.eval(ctx_f));

                            // Fluid cell cross-section dimensions perpendicular to flow
                            int ax_w = (f_ax + 1) % 3;
                            int ax_h = (f_ax + 2) % 3;
                            double w = (ax_w == 0) ? mesh.dx[ix] : ((ax_w == 1) ? mesh.dy[iy] : mesh.dz[iz]);
                            double h = (ax_h == 0) ? mesh.dx[ix] : ((ax_h == 1) ? mesh.dy[iy] : mesh.dz[iz]);

                            double Nu = mhs::utils::nusselt_rectangular(w, h);
                            double d_h = 2.0 * w * h / (w + h);
                            double h_f = Nu * kf / d_h; // convection coefficient

                            // Solid-side conduction half-distance
                            double half_dist_solid = (s_id == c_idx) ? half_dist : d_half_neighbor;
                            double k_solid = (s_id == c_idx) ? k_face : k_neighbor;
                            double A_solid = (s_id == c_idx)
                                ? A_f
                                : mhs::utils::face_area(dir, mesh.dx[nix], mesh.dy[niy], mesh.dz[niz]);

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

        // Merge advection contributions (upwind) if fluid cells exist
        assembleAdvection(K, b, state);

        return {std::move(K), std::move(b), std::move(M_diag)};
    }

    // ---------------------------------------------------------------------------
    // Advection: upwind formulation on fluid-fluid internal faces
    // ---------------------------------------------------------------------------

    void Assembler::assembleAdvection(
        Eigen::SparseMatrix<double>& K, Eigen::VectorXd& f, const mhs::core::GlobalState& state) const
    {
        const auto& mesh = model_.mesh;
        const auto& cells = model_.cells;
        const auto& materials = model_.material_table;
        const int N = static_cast<int>(cells.cell_bcs.size());
        const int total = mesh.nx * mesh.ny * mesh.nz;

        if (model_.is_fluid.empty() || model_.is_fluid.size() != static_cast<size_t>(N))
            return;

        bool hasFluid = false;
        for (uint8_t v : model_.is_fluid) {
            if (v) {
                hasFluid = true;
                break;
            }
        }
        if (!hasFluid)
            return;

        // Thread-local scratch: triplets + RHS contributions
        struct AdvScratch {
            std::vector<Eigen::Triplet<double>> triplets;
            Eigen::VectorXd b;
            AdvScratch(int n) : b(Eigen::VectorXd::Zero(n)) { }
        };

        auto adv_data = tbb::enumerable_thread_specific<AdvScratch>([N]() { return AdvScratch(N); });

        tbb::parallel_for(0, total, [&](int old_idx) {
            if (cells.index_map[old_idx] == mhs::core::invalidIndex)
                return;

            int ix = old_idx / (mesh.ny * mesh.nz);
            int iy = (old_idx % (mesh.ny * mesh.nz)) / mesh.nz;
            int iz = old_idx % mesh.nz;
            int c_idx = (int)cells.index_map[old_idx];

            if (!model_.is_fluid[c_idx])
                return;

            auto& local = adv_data.local();
            double netOutflux = 0.0; // per-cell net mass outflow over all faces

            for (size_t f = 0; f < 6; ++f) {
                mhs::core::FaceDir dir = mhs::core::FACE_DIRS[f];
                int neighborOld
                    = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
                if (neighborOld < 0)
                    continue;

                int n_idx = (int)cells.index_map[neighborOld];
                if (n_idx < 0 || n_idx >= N || !model_.is_fluid[n_idx])
                    continue;

                int axis = mhs::utils::AXIS_OF_DIR[f];

                // Get conductance along this axis
                double hc_a = 0.0, hc_b = 0.0;
                switch (axis) {
                case 0:
                    hc_a = model_.hydroC_x[c_idx];
                    hc_b = model_.hydroC_x[n_idx];
                    break;
                case 1:
                    hc_a = model_.hydroC_y[c_idx];
                    hc_b = model_.hydroC_y[n_idx];
                    break;
                default:
                    hc_a = model_.hydroC_z[c_idx];
                    hc_b = model_.hydroC_z[n_idx];
                    break;
                }

                if (hc_a < 1e-30 || hc_b < 1e-30)
                    continue;
                double C_eff = 2.0 * hc_a * hc_b / (hc_a + hc_b);

                // Average density
                double rho_a = materials[cells.material_id[c_idx]].rho.eval(
                    {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T[c_idx], state.current_time});
                int nix = mhs::utils::neighbor_ix(dir, ix);
                int niy = mhs::utils::neighbor_iy(dir, iy);
                int niz = mhs::utils::neighbor_iz(dir, iz);
                double rho_b = materials[cells.material_id[n_idx]].rho.eval(
                    {mesh.cx[nix], mesh.cy[niy], mesh.cz[niz], state.T[n_idx], state.current_time});
                double rho_avg = 0.5 * (rho_a + rho_b);
                if (rho_avg < 1e-30)
                    rho_avg = 1e-30;

                // Mass flux = ΔP * C_eff * ρ_avg
                double dP = model_.pressure[c_idx] - model_.pressure[n_idx];
                double massFlux = dP * C_eff * rho_avg;

                netOutflux += massFlux;

                if (std::fabs(massFlux) < 1e-30)
                    continue;

                if (massFlux > 0) {
                    // Outflow: c loses enthalpy at its own temperature
                    // K[c,c] += massFlux * cp_c
                    double cp_c = materials[cells.material_id[c_idx]].c.eval(
                        {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T[c_idx], state.current_time});
                    local.triplets.emplace_back(c_idx, c_idx, massFlux * cp_c);
                } else {
                    // Inflow: c gains enthalpy at neighbor's temperature
                    // K[c,n] += massFlux * cp_n  (massFlux < 0 → effectively −|massFlux| at column n)
                    double cp_n = materials[cells.material_id[n_idx]].c.eval(
                        {mesh.cx[nix], mesh.cy[niy], mesh.cz[niz], state.T[n_idx], state.current_time});
                    local.triplets.emplace_back(c_idx, n_idx, massFlux * cp_n);
                }
            }

            // Boundary temperature injection: for cells with prescribed inlet temperature
            // and non-zero net flow, impose the incoming enthalpy on the RHS.
            // no > 0  → net outflow (inlet cell):  RHS += no * cp * T_boundary
            // no < 0  → net inflow (outlet cell):   RHS += no * cp * T_boundary (no is negative,
            //                                         adds negative contribution — handled by matrix)
            if (std::fabs(netOutflux) >= 1e-30) {
                double T_boundary = model_.boundary_temperature_fluid[c_idx];
                if (!std::isnan(T_boundary)) {
                    double cp = materials[cells.material_id[c_idx]].c.eval(
                        {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T[c_idx], state.current_time});
                    local.b(c_idx) += netOutflux * cp * T_boundary;
                }
            }
        });

        // Combine triplets and RHS
        std::vector<Eigen::Triplet<double>> allAdvTriplets;
        Eigen::VectorXd adv_b = Eigen::VectorXd::Zero(N);

        adv_data.combine_each([&](const AdvScratch& local) {
            allAdvTriplets.insert(allAdvTriplets.end(), local.triplets.begin(), local.triplets.end());
            adv_b += local.b;
        });

        // Merge advection matrix and RHS into K and f
        Eigen::SparseMatrix<double> K_adv(N, N);
        K_adv.setFromTriplets(allAdvTriplets.begin(), allAdvTriplets.end());
        K += K_adv;
        f += adv_b;
    }

} // namespace mhs::sim
