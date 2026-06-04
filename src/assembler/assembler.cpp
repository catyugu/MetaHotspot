#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <Eigen/Sparse>

#include "assembler.hpp"

namespace mhs::assembler {

    static int grid_index(int ix, int iy, int iz, int ny, int nz)
    {
        return ix * ny * nz + iy * nz + iz;
    }

    static void decode_index(int old_idx, int ny, int nz, int& ix, int& iy, int& iz)
    {
        ix = old_idx / (ny * nz);
        iy = (old_idx % (ny * nz)) / nz;
        iz = old_idx % nz;
    }

    static int neighbor_grid_index(int ix, int iy, int iz, FaceDir dir, int nx, int ny, int nz)
    {
        switch (dir) {
        case FaceDir::XM:
            return ix > 0 ? grid_index(ix - 1, iy, iz, ny, nz) : -1;
        case FaceDir::XP:
            return ix < nx - 1 ? grid_index(ix + 1, iy, iz, ny, nz) : -1;
        case FaceDir::YM:
            return iy > 0 ? grid_index(ix, iy - 1, iz, ny, nz) : -1;
        case FaceDir::YP:
            return iy < ny - 1 ? grid_index(ix, iy + 1, iz, ny, nz) : -1;
        case FaceDir::ZM:
            return iz > 0 ? grid_index(ix, iy, iz - 1, ny, nz) : -1;
        case FaceDir::ZP:
            return iz < nz - 1 ? grid_index(ix, iy, iz + 1, ny, nz) : -1;
        default:
            return -1;
        }
    }

    static double face_area(FaceDir dir, double dx, double dy, double dz)
    {
        switch (dir) {
        case FaceDir::XM:
        case FaceDir::XP:
            return dy * dz;
        case FaceDir::YM:
        case FaceDir::YP:
            return dx * dz;
        case FaceDir::ZM:
        case FaceDir::ZP:
            return dx * dy;
        default:
            return 0.0;
        }
    }

    static int neighbor_ix(FaceDir dir, int ix)
    {
        switch (dir) {
        case FaceDir::XM:
            return ix - 1;
        case FaceDir::XP:
            return ix + 1;
        default:
            return ix;
        }
    }
    static int neighbor_iy(FaceDir dir, int iy)
    {
        switch (dir) {
        case FaceDir::YM:
            return iy - 1;
        case FaceDir::YP:
            return iy + 1;
        default:
            return iy;
        }
    }
    static int neighbor_iz(FaceDir dir, int iz)
    {
        switch (dir) {
        case FaceDir::ZM:
            return iz - 1;
        case FaceDir::ZP:
            return iz + 1;
        default:
            return iz;
        }
    }

    struct ThreadLocalData {
        std::vector<Eigen::Triplet<double>> triplets;
        Eigen::VectorXd b;
        explicit ThreadLocalData(int N) : b(Eigen::VectorXd::Zero(N)) { }
    };

    LinearSystem Assembler::assemble(const GlobalState& state)
    {
        const auto& mesh = model_.mesh;
        const auto& cells = model_.cells;
        const auto& bc_params = model_.bc_params;
        const auto& materials = model_.material_table;

        int N = cells.cell_count;
        int total = mesh.total_cell_count;

        auto thread_data = tbb::enumerable_thread_specific<ThreadLocalData>(
            [&]() { return ThreadLocalData(N); });

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
            size_t mat_id = cells.material_id[old_idx];
            double k = materials[mat_id].k.eval(
                {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T[c_idx], state.current_time});

            // 热源字典求值：利用 uint16_t 索引进行极速查表和计算
            uint16_t hs_idx = cells.heat_source_idx[c_idx];
            double Q = model_.heat_source_table[hs_idx].eval(
                {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T[c_idx], state.current_time});
            local.b(c_idx) += Q * vol;

            const auto& cell_bc = cells.cell_bcs[c_idx];
            double diag = 0.0;

            for (size_t f = 0; f < FACE_COUNT; f++) {
                FaceDir dir = FACE_DIRS[f];
                double A_f = face_area(dir, dx_cell, dy_cell, dz_cell);
                BcType bc_type = cell_bc.types[f];
                uint16_t param_idx = cell_bc.param_idxs[f];

                if (bc_type == BcType::None) {
                    int neighbor_old
                        = neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz);
                    if (neighbor_old < 0 || cells.valid_mask[neighbor_old] == 0)
                        continue;

                    int n_idx = (int)cells.index_map[neighbor_old];
                    int nix = neighbor_ix(dir, ix);
                    int niy = neighbor_iy(dir, iy);
                    int niz = neighbor_iz(dir, iz);

                    double k_neighbor
                        = materials[cells.material_id[neighbor_old]].k.eval({mesh.cx[nix],
                            mesh.cy[niy], mesh.cz[niz], state.T[n_idx], state.current_time});

                    double d_half_cell = (dir == FaceDir::XM || dir == FaceDir::XP)
                        ? mesh.dx[ix] / 2.0
                        : (dir == FaceDir::YM || dir == FaceDir::YP) ? mesh.dy[iy] / 2.0
                                                                     : mesh.dz[iz] / 2.0;

                    double d_half_neighbor = (dir == FaceDir::XM || dir == FaceDir::XP)
                        ? mesh.dx[nix] / 2.0
                        : (dir == FaceDir::YM || dir == FaceDir::YP) ? mesh.dy[niy] / 2.0
                                                                     : mesh.dz[niz] / 2.0;

                    double cond = A_f / (d_half_cell / k + d_half_neighbor / k_neighbor);
                    diag += cond;
                    local.triplets.emplace_back(c_idx, n_idx, -cond);
                }
                else if (bc_type == BcType::FirstType) {
                    double half_dist = (dir == FaceDir::XM || dir == FaceDir::XP) ? dx_cell / 2.0
                        : (dir == FaceDir::YM || dir == FaceDir::YP)              ? dy_cell / 2.0
                                                                                  : dz_cell / 2.0;

                    double T_bc_val = bc_params.dirichlet_T[param_idx].eval({mesh.cx[ix],
                        mesh.cy[iy], mesh.cz[iz], state.T[c_idx], state.current_time});

                    double cond = k * A_f / half_dist;
                    diag += cond;
                    local.b(c_idx) += cond * T_bc_val;
                }
                else if (bc_type == BcType::SecondType) {
                    double q = bc_params.neumann_q[param_idx].eval({mesh.cx[ix], mesh.cy[iy],
                        mesh.cz[iz], state.T[c_idx], state.current_time});
                    local.b(c_idx) += q * A_f;
                }
                else if (bc_type == BcType::ThirdType) {
                    double h = bc_params.cauchy_h[param_idx].eval({mesh.cx[ix], mesh.cy[iy],
                        mesh.cz[iz], state.T[c_idx], state.current_time});
                    double T_inf = bc_params.cauchy_T_inf[param_idx].eval({mesh.cx[ix], mesh.cy[iy],
                        mesh.cz[iz], state.T[c_idx], state.current_time});

                    double half_dist = (dir == FaceDir::XM || dir == FaceDir::XP) ? dx_cell / 2.0
                        : (dir == FaceDir::YM || dir == FaceDir::YP)              ? dy_cell / 2.0
                                                                                  : dz_cell / 2.0;

                    double coeff = k * h * A_f / (k + h * half_dist);
                    diag += coeff;
                    local.b(c_idx) += coeff * T_inf;
                }
            }

            local.triplets.emplace_back(c_idx, c_idx, diag);

            if (model_.study_type == StudyType::Transient && state.dt > 0.0) {
                double rho = materials[mat_id].rho.eval(
                    {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T[c_idx], state.current_time});
                double c_heat = materials[mat_id].c.eval(
                    {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], state.T[c_idx], state.current_time});

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

} // namespace mhs::assembler