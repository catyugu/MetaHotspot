#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <Eigen/Sparse>

#include "assembler.hpp"
#include "common/face_dir_tables.hpp"

namespace mhs::sim {

    namespace { // anonymous: file-private helpers
        void decode_index(int old_idx, int ny, int nz, int& ix, int& iy, int& iz)
        {
            ix = old_idx / (ny * nz);
            iy = (old_idx % (ny * nz)) / nz;
            iz = old_idx % nz;
        }
    } // namespace

    struct ThreadLocalData {
        std::vector<Eigen::Triplet<double>> triplets;
        Eigen::VectorXd b;
        explicit ThreadLocalData(int N) : b(Eigen::VectorXd::Zero(N)) { }
    };

    LinearSystem Assembler::assemble(const mhs::core::GlobalState& state)
    {
        const auto& mesh = model_.mesh;
        const auto& cells = model_.cells;
        const auto& bc_params = model_.bc_params;
        const auto& materials = model_.material_table;

        int N = static_cast<int>(cells.cell_bcs.size());
        int total = mesh.nx * mesh.ny * mesh.nz;

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

            // 材料字典求值：底层已由 TLS 保证线程安全
            size_t mat_id = cells.material_id[c_idx];
            const auto& mp = materials[mat_id];
            double kx_c = mp.kx.eval({mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T[c_idx], state.current_time});
            double ky_c = mp.ky.eval({mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T[c_idx], state.current_time});
            double kz_c = mp.kz.eval({mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T[c_idx], state.current_time});

            // 热源字典求值：利用 uint16_t 索引进行极速查表和计算
            uint16_t hs_idx = cells.heat_source_idx[c_idx];
            double Q = model_.heat_source_table[hs_idx].eval(
                {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T[c_idx], state.current_time});
            local.b(c_idx) += Q * vol;

            const auto& cell_bc = cells.cell_bcs[c_idx];
            double diag = 0.0;

            for (size_t f = 0; f < mhs::core::FACE_COUNT; f++) {
                mhs::core::FaceDir dir = mhs::core::FACE_DIRS[f];
                double A_f = mhs::core::face_area(dir, dx_cell, dy_cell, dz_cell);
                mhs::core::BcType bc_type = cell_bc.types[f];
                uint16_t param_idx = cell_bc.param_idxs[f];

                if (bc_type == mhs::core::BcType::None) {
                    int neighbor_old
                        = mhs::core::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.valid_mask);
                    if (neighbor_old < 0)
                        continue;

                    int n_idx = (int)cells.index_map[neighbor_old];
                    int nix = mhs::core::neighbor_ix(dir, ix);
                    int niy = mhs::core::neighbor_iy(dir, iy);
                    int niz = mhs::core::neighbor_iz(dir, iz);

                    const auto& mp_n = materials[cells.material_id[n_idx]];
                    double kx_n
                        = mp_n.kx.eval({mesh.cx[nix], mesh.cy[niy], mesh.cz[niz], state.T[n_idx], state.current_time});
                    double ky_n
                        = mp_n.ky.eval({mesh.cx[nix], mesh.cy[niy], mesh.cz[niz], state.T[n_idx], state.current_time});
                    double kz_n
                        = mp_n.kz.eval({mesh.cx[nix], mesh.cy[niy], mesh.cz[niz], state.T[n_idx], state.current_time});

                    double d_half_cell = mhs::core::half_length_along(dir, mesh.dx[ix], mesh.dy[iy], mesh.dz[iz]);
                    double d_half_neighbor
                        = mhs::core::half_length_along(dir, mesh.dx[nix], mesh.dy[niy], mesh.dz[niz]);

                    double k_cell = mhs::core::k_along(dir, kx_c, ky_c, kz_c);
                    double k_neighbor = mhs::core::k_along(dir, kx_n, ky_n, kz_n);
                    double cond = A_f / (d_half_cell / k_cell + d_half_neighbor / k_neighbor);
                    diag += cond;
                    local.triplets.emplace_back(c_idx, n_idx, -cond);
                }
                else if (bc_type == mhs::core::BcType::FirstType) {
                    double half_dist = mhs::core::half_length_along(dir, dx_cell, dy_cell, dz_cell);

                    double T_bc_val = bc_params.dirichlet_T[param_idx].eval(
                        {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T[c_idx], state.current_time});

                    double k_face = mhs::core::k_along(dir, kx_c, ky_c, kz_c);
                    double cond = k_face * A_f / half_dist;
                    diag += cond;
                    local.b(c_idx) += cond * T_bc_val;
                }
                else if (bc_type == mhs::core::BcType::SecondType) {
                    double q = bc_params.neumann_q[param_idx].eval(
                        {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T[c_idx], state.current_time});
                    local.b(c_idx) += q * A_f;
                }
                else if (bc_type == mhs::core::BcType::ThirdType) {
                    double h = bc_params.cauchy_h[param_idx].eval(
                        {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T[c_idx], state.current_time});
                    double T_inf = bc_params.cauchy_T_inf[param_idx].eval(
                        {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T[c_idx], state.current_time});

                    double half_dist = mhs::core::half_length_along(dir, dx_cell, dy_cell, dz_cell);

                    double k_face = mhs::core::k_along(dir, kx_c, ky_c, kz_c);
                    double coeff = k_face * h * A_f / (k_face + h * half_dist);
                    diag += coeff;
                    local.b(c_idx) += coeff * T_inf;
                }
            }

            local.triplets.emplace_back(c_idx, c_idx, diag);

            if (model_.study_type == mhs::core::StudyType::Transient && state.dt > 0.0) {
                // 标准 backward Euler：质量项用 T_prev 求值（与 T 解耦），
                // 这样 mass_coeff 在非线性迭代中是常数，避免在更新 T 时反复重算。
                double rho = materials[mat_id].rho.eval(
                    {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T_prev[c_idx], state.current_time});
                double c_heat = materials[mat_id].c.eval(
                    {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T_prev[c_idx], state.current_time});

                double mass_coeff = rho * c_heat * vol / state.dt;
                local.triplets.emplace_back(c_idx, c_idx, mass_coeff);
                local.b(c_idx) += mass_coeff * state.T_prev[c_idx];
            }
        });

        std::vector<Eigen::Triplet<double>> triplets;
        Eigen::VectorXd b = Eigen::VectorXd::Zero(N);

        thread_data.combine_each([&](const ThreadLocalData& local) {
            triplets.insert(triplets.end(), local.triplets.begin(), local.triplets.end());
            b += local.b;
        });

        Eigen::SparseMatrix<double> A(N, N);
        A.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::VectorXd T_vec(N);
        for (int i = 0; i < N; i++)
            T_vec(i) = state.T[i];

        Eigen::VectorXd residual_vec = b - A * T_vec;
        return {A, b, residual_vec};
    }

} // namespace mhs::sim