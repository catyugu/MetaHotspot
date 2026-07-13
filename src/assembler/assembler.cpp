#include <Eigen/Sparse>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <unordered_map>

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
        const auto& face_bcs = model_.face_bcs;
        const auto& materials = model_.material_table;

        int N_phys = model_.physical_dofs();
        int N_total = model_.total_dofs();
        int total = mesh.nx * mesh.ny * mesh.nz;

        auto thread_data = tbb::enumerable_thread_specific<ThreadLocalData>([&]() { return ThreadLocalData(N_total); });

        // ═══════════════════════════════════════════════════════════════════
        // Phase 1: 内部传导 + 热源 + 瞬态质量矩阵 + 边界条件
        // 遍历所有 cell，对每个面：
        //   有有效邻居 → 内部扩散和对流
        //   无邻居（暴露面）→ face_bcs[c*6+f] 中取 BC 值直接装配
        // 旧的 Phase 2 面片遍历不再需要 —— face 信息已在 face_bcs 中。
        // ═══════════════════════════════════════════════════════════════════
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

            // 缓存 cell 上下文
            size_t mat_id = cells.material_id[c_idx];
            const auto& mp = materials[mat_id];
            const mhs::core::FieldContext ctx_c {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], ctx.T[c_idx], ctx.current_time};
            const double kx_c = mp.kx.eval(ctx_c);
            const double ky_c = mp.ky.eval(ctx_c);
            const double kz_c = mp.kz.eval(ctx_c);

            // Mass coefficients
            const mhs::core::FieldContext ctx_m {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], ctx.T[c_idx], ctx.current_time};
            const double rho = mp.rho.eval(ctx_m);
            const double c_heat = mp.c.eval(ctx_m);
            local.mass(c_idx) += rho * c_heat * vol;

            // Heat source
            uint16_t hs_idx = cells.heat_source_idx[c_idx];
            const double Q = model_.heat_source_table[hs_idx].eval(ctx_c);
            local.b(c_idx) += Q * vol;

            double diag = 0.0;
            bool cell_is_fluid = !model_.fluid.is_fluid.empty() && c_idx < (int)model_.fluid.is_fluid.size()
                && model_.fluid.is_fluid[c_idx];
            const double rho_a = cell_is_fluid ? materials[cells.material_id[c_idx]].rho.eval(ctx_c) : 0.0;
            const double cp_c = cell_is_fluid ? materials[cells.material_id[c_idx]].c.eval(ctx_c) : 0.0;
            double netOutflux = 0.0;

            // 行基地址 O(1) 预处理
            const auto* fc = &face_bcs[c_idx * mhs::core::FACE_COUNT];

            for (size_t f = 0; f < mhs::core::FACE_COUNT; f++) {
                mhs::core::FaceDir dir = mhs::core::FACE_DIRS[f];

                // 检查是否有有效邻居（内部面）
                int neighbor_old
                    = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
                if (neighbor_old >= 0) {
                    // ── 内部面：扩散 + 可能对流 ──
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

                    const double A_f = mhs::utils::face_area(dir, dx_cell, dy_cell, dz_cell);
                    const double half_dist = mhs::utils::half_length_along(dir, dx_cell, dy_cell, dz_cell);
                    const double k_face = mhs::utils::k_along(dir, kx_c, ky_c, kz_c);

                    double cond = 0.0;
                    bool n_is_fluid
                        = (n_idx >= 0 && n_idx < (int)model_.fluid.is_fluid.size()) && model_.fluid.is_fluid[n_idx];

                    // Fluid-solid interface: Nusselt-based convection correction
                    if (cell_is_fluid != n_is_fluid) {
                        int f_id = cell_is_fluid ? c_idx : n_idx;
                        int f_idx = model_.fluid.global_to_fluid[f_id];
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
                        cond = A_f / (half_dist / k_face + d_half_neighbor / k_neighbor);
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
                else {
                    // ── 暴露面：从 face_bcs 取 BC 直接装配 ──
                    const auto& fb = fc[f];
                    if (fb.type == mhs::core::BcType::None)
                        continue;

                    const double A_f = mhs::utils::face_area(dir, dx_cell, dy_cell, dz_cell);
                    const double half_dist = mhs::utils::half_length_along(dir, dx_cell, dy_cell, dz_cell);
                    const double k_face = mhs::utils::k_along(dir, kx_c, ky_c, kz_c);

                    switch (fb.type) {
                    case mhs::core::BcType::FirstType: {
                        double T_bc_val = bc_params.dirichlet_T[fb.param_idx].eval(ctx_c);
                        double cond = k_face * A_f / half_dist;
                        local.triplets.emplace_back(c_idx, c_idx, cond);
                        local.b(c_idx) += cond * T_bc_val;
                        break;
                    }
                    case mhs::core::BcType::SecondType: {
                        double q = bc_params.neumann_q[fb.param_idx].eval(ctx_c);
                        local.b(c_idx) += q * A_f;
                        break;
                    }
                    case mhs::core::BcType::ThirdType: {
                        double h = bc_params.cauchy_h[fb.param_idx].eval(ctx_c);
                        double T_inf = bc_params.cauchy_T_inf[fb.param_idx].eval(ctx_c);
                        double coeff = k_face * h * A_f / (k_face + h * half_dist);
                        local.triplets.emplace_back(c_idx, c_idx, coeff);
                        local.b(c_idx) += coeff * T_inf;
                        break;
                    }
                    default:
                        break;
                    }
                }
            }

            local.triplets.emplace_back(c_idx, c_idx, diag);

            // ── Fluid BC outlet/inlet handling ──
            if (cell_is_fluid) {
                int f_idx = model_.fluid.global_to_fluid[c_idx];
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

        // ── Combine thread-local data ──
        std::vector<Eigen::Triplet<double>> triplets;
        Eigen::VectorXd b = Eigen::VectorXd::Zero(N_total);
        Eigen::VectorXd M_diag = Eigen::VectorXd::Zero(N_total);

        thread_data.combine_each([&](const ThreadLocalData& local) {
            triplets.insert(triplets.end(), local.triplets.begin(), local.triplets.end());
            b += local.b;
            M_diag += local.mass;
        });

        // ── Scatter extended system per SmartBlock (face-level POD) ──
        for (const auto& sb : model_.smart_blocks) {
            const int n_faces = sb.n_faces;
            const int n_modes = sb.n_modes;
            const int modal_start = sb.modal_start_idx;
            if (n_faces == 0 || n_modes == 0)
                continue;

            // ── Phase 1: compute environment parameters per face ──
            Eigen::VectorXd C_env_vec = Eigen::VectorXd::Zero(n_faces);
            Eigen::VectorXd T_ref_vec = Eigen::VectorXd::Zero(n_faces);
            Eigen::VectorXd Q_ext_vec = Eigen::VectorXd::Zero(n_faces);

            // Per-neighbor aggregation scratch.
            std::unordered_map<uint32_t, std::pair<double, Eigen::VectorXd>> cell_contrib;
            // cell_contrib[neighbor_c] = (C_total, phiC_total)  — phiC_total sized n_modes

            for (int p = 0; p < n_faces; p++) {
                const auto& pfi = sb.faces[p];

                if (pfi.has_neighbor) {
                    // ── Active neighbor: C_env = k_n * A_f / h_n ──
                    const uint32_t act_c = pfi.neighbor_c;
                    const auto& nmp = materials[cells.material_id[act_c]];
                    const mhs::core::FieldContext ctx_n {
                        mesh.cx[pfi.ix], mesh.cy[pfi.iy], mesh.cz[pfi.iz], ctx.T[act_c], ctx.current_time};
                    const double k_active
                        = mhs::utils::k_along(pfi.dir, nmp.kx.eval(ctx_n), nmp.ky.eval(ctx_n), nmp.kz.eval(ctx_n));

                    const double C_env = k_active * pfi.A_f / pfi.half_dist_nbr;

                    C_env_vec(p) = C_env;
                    T_ref_vec(p) = 0.0; // T_neighbor is unknown → enters the system matrix
                    Q_ext_vec(p) = 0.0;

                    // Aggregate C_env and C_env*Φ for this neighbor.
                    auto& [c_total, phi_total] = cell_contrib[act_c];
                    if (phi_total.size() == 0) {
                        phi_total = Eigen::VectorXd::Zero(n_modes);
                    }
                    c_total += C_env;
                    phi_total += C_env * sb.phi_basis.row(p).transpose();
                }
                else {
                    // ── Domain boundary face: BC-type dispatch ──
                    double C_env = 0.0, T_ref = 0.0, Q_ext = 0.0;

                    const auto& bc = model_.bc_params;
                    // Build context from the owner cell (port faces are inside the block).
                    const mhs::core::FieldContext ctx_c {mesh.cx[pfi.ix], mesh.cy[pfi.iy], mesh.cz[pfi.iz],
                        // Use the average of neighbouring physical cells if possible;
                        // for domain-boundary ports the owner cell is inside the block
                        // and has no active DOF, so we use ctx.T values from the
                        // nearest physical cell context. In practice the BC
                        // expressions rarely depend on T (they are ambient conditions),
                        // but we supply the best available context.
                        model_.initial_temperature, ctx.current_time};

                    switch (pfi.bc_type) {
                    case mhs::core::BcType::None:
                        // Adiabatic.
                        break;

                    case mhs::core::BcType::FirstType: {
                        double T_bc_val = bc.dirichlet_T[pfi.bc_param_idx].eval(ctx_c);
                        C_env = 1e10; // penalty
                        T_ref = T_bc_val;
                        break;
                    }

                    case mhs::core::BcType::SecondType: {
                        double q_val = bc.neumann_q[pfi.bc_param_idx].eval(ctx_c);
                        Q_ext = q_val * pfi.A_f;
                        break;
                    }

                    case mhs::core::BcType::ThirdType: {
                        double h_val = bc.cauchy_h[pfi.bc_param_idx].eval(ctx_c);
                        double T_inf = bc.cauchy_T_inf[pfi.bc_param_idx].eval(ctx_c);
                        C_env = h_val * pfi.A_f;
                        T_ref = T_inf;
                        break;
                    }

                    default:
                        break;
                    }

                    C_env_vec(p) = C_env;
                    T_ref_vec(p) = T_ref;
                    Q_ext_vec(p) = Q_ext;
                }
            }

            // ── Phase 2: compute K_modal_eff = K_modal + Φᵀ·diag(C_env)·Φ ──
            Eigen::MatrixXd K_modal_eff = sb.K_modal + sb.phi_basis.transpose() * C_env_vec.asDiagonal() * sb.phi_basis;

            // ── Phase 3: aggregate unique coupled cells ──
            // (re-aggregated every assembly so nonlinear neighbours are correct)
            std::vector<uint32_t> coupled_cells;
            coupled_cells.reserve(cell_contrib.size());
            for (const auto& kv : cell_contrib) {
                coupled_cells.push_back(kv.first);
            }
            std::sort(coupled_cells.begin(), coupled_cells.end());

            const size_t n_coupled = coupled_cells.size();
            Eigen::VectorXd coupled_C(n_coupled);
            Eigen::MatrixXd coupled_phi(n_coupled, n_modes);
            for (size_t ci = 0; ci < n_coupled; ci++) {
                const auto& contrib = cell_contrib[coupled_cells[ci]];
                coupled_C(ci) = contrib.first;
                coupled_phi.row(ci) = contrib.second;
            }

            // ── Phase 4: scatter to global system ──
            // 4a. Active-neighbor coupling.
            for (size_t ci = 0; ci < n_coupled; ci++) {
                int gi = (int)coupled_cells[ci];
                if (gi >= N_phys)
                    continue;
                double C_total = coupled_C(ci);
                if (std::abs(C_total) <= mhs::core::zero_guard)
                    continue;

                triplets.emplace_back(gi, gi, C_total);

                for (int k = 0; k < n_modes; k++) {
                    double val = -coupled_phi(ci, k);
                    if (std::abs(val) > mhs::core::zero_guard) {
                        int mk = modal_start + k;
                        triplets.emplace_back(gi, mk, val);
                        triplets.emplace_back(mk, gi, val);
                    }
                }
            }

            // 4b. Modal block: K_modal_eff.
            for (int k = 0; k < n_modes; k++) {
                for (int kp = 0; kp < n_modes; kp++) {
                    double val = K_modal_eff(k, kp);
                    if (std::abs(val) > mhs::core::zero_guard) {
                        triplets.emplace_back(modal_start + k, modal_start + kp, val);
                    }
                }
            }

            // 4c. Modal RHS: domain-BC contribution
            //     b(mk) += Σ_p φ(p,k) * (C_env(p)*T_ref(p) + Q_ext(p))
            {
                Eigen::VectorXd face_rhs(n_faces);
                for (int p = 0; p < n_faces; p++) {
                    double r = 0.0;
                    if (std::abs(C_env_vec(p)) > mhs::core::zero_guard
                        && std::abs(T_ref_vec(p)) > mhs::core::zero_guard) {
                        r += C_env_vec(p) * T_ref_vec(p);
                    }
                    if (std::abs(Q_ext_vec(p)) > mhs::core::zero_guard) {
                        r += Q_ext_vec(p);
                    }
                    face_rhs(p) = r;
                }
                b.segment(modal_start, n_modes) += sb.phi_basis.transpose() * face_rhs;
            }
        }

        Eigen::SparseMatrix<double> K(N_total, N_total);
        K.setFromTriplets(triplets.begin(), triplets.end());

        return {std::move(K), std::move(b), std::move(M_diag)};
    }

} // namespace mhs::sim
