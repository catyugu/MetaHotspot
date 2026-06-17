#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <Eigen/Sparse>

#include "assembler.hpp"
#include "common/mesh_utils.hpp"

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

        const std::vector<double>* T_eval_mass = (state.history.size() > 0) ? &state.history.latest() : &state.T;

        auto thread_data = tbb::enumerable_thread_specific<ThreadLocalData>([&]() { return ThreadLocalData(N); });

        tbb::parallel_for(0, total, [&](int old_idx) {
            if (cells.valid_mask[old_idx] == 0)
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

            // Mass coefficients on history.latest() — see comment above.
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
                double A_f = mhs::utils::face_area(dir, dx_cell, dy_cell, dz_cell);
                mhs::core::BcType bc_type = cell_bc.types[f];
                uint16_t param_idx = cell_bc.param_idxs[f];

                if (bc_type == mhs::core::BcType::None) {
                    int neighbor_old
                        = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.valid_mask);
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

                    double d_half_cell = mhs::utils::half_length_along(dir, mesh.dx[ix], mesh.dy[iy], mesh.dz[iz]);
                    double d_half_neighbor
                        = mhs::utils::half_length_along(dir, mesh.dx[nix], mesh.dy[niy], mesh.dz[niz]);

                    double k_cell = mhs::utils::k_along(dir, kx_c, ky_c, kz_c);
                    double k_neighbor = mhs::utils::k_along(dir, kx_n, ky_n, kz_n);
                    double cond = A_f / (d_half_cell / k_cell + d_half_neighbor / k_neighbor);
                    diag += cond;
                    local.triplets.emplace_back(c_idx, n_idx, -cond);
                }
                else if (bc_type == mhs::core::BcType::FirstType) {
                    double half_dist = mhs::utils::half_length_along(dir, dx_cell, dy_cell, dz_cell);

                    double T_bc_val = bc_params.dirichlet_T[param_idx].eval(ctx_c);

                    double k_face = mhs::utils::k_along(dir, kx_c, ky_c, kz_c);
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

                    double half_dist = mhs::utils::half_length_along(dir, dx_cell, dy_cell, dz_cell);

                    double k_face = mhs::utils::k_along(dir, kx_c, ky_c, kz_c);
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

        return {std::move(K), std::move(b), std::move(M_diag)};
    }

} // namespace mhs::sim
