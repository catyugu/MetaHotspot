#include "assembler.hpp"
#include <Eigen/Sparse>

namespace mhs::assembler {

    // Helper: convert 3D grid index to flat index
    static int grid_index(int ix, int iy, int iz, int ny, int nz)
    {
        return ix * ny * nz + iy * nz + iz;
    }

    // Helper: get neighbor grid index in a given face direction
    // Returns -1 if neighbor is out of grid bounds
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

    // Helper: face area for a cell at (ix, iy, iz)
    // XM/XP: dy*dz, YM/YP: dx*dz, ZM/ZP: dx*dy
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

    // Helper: distance between cell centers across a face
    static double face_distance(FaceDir dir,
        const model::MeshGeometry& mesh,
        int ix, int iy, int iz)
    {
        switch (dir) {
        case FaceDir::XM:
            return (mesh.cx[ix] - mesh.cx[ix - 1]);
        case FaceDir::XP:
            return (mesh.cx[ix + 1] - mesh.cx[ix]);
        case FaceDir::YM:
            return (mesh.cy[iy] - mesh.cy[iy - 1]);
        case FaceDir::YP:
            return (mesh.cy[iy + 1] - mesh.cy[iy]);
        case FaceDir::ZM:
            return (mesh.cz[iz] - mesh.cz[iz - 1]);
        case FaceDir::ZP:
            return (mesh.cz[iz + 1] - mesh.cz[iz]);
        default:
            return 0.0;
        }
    }

    // Get neighbor cell coordinate for face direction
    static int neighbor_ix(FaceDir dir, int ix) {
        switch (dir) {
        case FaceDir::XM: return ix - 1;
        case FaceDir::XP: return ix + 1;
        default: return ix;
        }
    }
    static int neighbor_iy(FaceDir dir, int iy) {
        switch (dir) {
        case FaceDir::YM: return iy - 1;
        case FaceDir::YP: return iy + 1;
        default: return iy;
        }
    }
    static int neighbor_iz(FaceDir dir, int iz) {
        switch (dir) {
        case FaceDir::ZM: return iz - 1;
        case FaceDir::ZP: return iz + 1;
        default: return iz;
        }
    }

    LinearSystem Assembler::assemble(const model::GlobalState& state)
    {
        const auto& mesh = model_.mesh;
        const auto& cells = model_.cells;
        const auto& bc_params = model_.bc_params;
        const auto& materials = model_.material_table;

        int N = cells.cell_count;

        // Build triplet list for sparse matrix assembly
        // Convention: A*T = b where A has POSITIVE diagonal, NEGATIVE off-diagonal
        std::vector<Eigen::Triplet<double>> triplets;
        Eigen::VectorXd b = Eigen::VectorXd::Zero(N);

        for (int ix = 0; ix < mesh.nx; ix++) {
            for (int iy = 0; iy < mesh.ny; iy++) {
                for (int iz = 0; iz < mesh.nz; iz++) {
                    int old_idx = grid_index(ix, iy, iz, mesh.ny, mesh.nz);

                    if (cells.valid_mask[old_idx] == 0)
                        continue;

                    int c_idx = (int)cells.index_map[old_idx];
                    double dx_cell = mesh.dx[ix];
                    double dy_cell = mesh.dy[iy];
                    double dz_cell = mesh.dz[iz];
                    double vol = dx_cell * dy_cell * dz_cell;

                    // Material thermal conductivity
                    size_t mat_id = cells.material_id[old_idx];
                    double k = materials[mat_id].k.eval({mesh.cx[ix], mesh.cy[iy], mesh.cz[iz],
                        state.T[c_idx], state.current_time});

                    // Heat source term: Q * vol on RHS
                    double Q = cells.heat_source[c_idx].eval(
                        {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz],
                         state.T[c_idx], state.current_time});
                    b(c_idx) += Q * vol;

                    // Process each face
                    const auto& cell_bc = cells.cell_bcs[c_idx];
                    double diag = 0.0;

                    for (size_t f = 0; f < FACE_COUNT; f++) {
                        FaceDir dir = FACE_DIRS[f];
                        double A_f = face_area(dir, dx_cell, dy_cell, dz_cell);
                        BcType bc_type = cell_bc.types[f];
                        uint16_t param_idx = cell_bc.param_idxs[f];

                        if (bc_type == BcType::None) {
                            // Interior face: diffusion flux = k_face * A_f / dist * (T_n - T_c)
                            // In equation: k_face*A_f/dist * T_n - k_face*A_f/dist * T_c = contribution
                            // diag += k_face*A_f/dist, off-diag -= k_face*A_f/dist
                            int neighbor_old = neighbor_grid_index(ix, iy, iz, dir,
                                mesh.nx, mesh.ny, mesh.nz);
                            if (neighbor_old < 0)
                                continue;
                            if (cells.valid_mask[neighbor_old] == 0)
                                continue;

                            int n_idx = (int)cells.index_map[neighbor_old];
                            int nix = neighbor_ix(dir, ix);
                            int niy = neighbor_iy(dir, iy);
                            int niz = neighbor_iz(dir, iz);

                            double dist = face_distance(dir, mesh, ix, iy, iz);
                            double k_neighbor = materials[cells.material_id[neighbor_old]].k.eval(
                                {mesh.cx[nix], mesh.cy[niy], mesh.cz[niz],
                                 state.T[n_idx], state.current_time});

                            double k_face = 2.0 * k * k_neighbor / (k + k_neighbor);
                            double coeff = k_face * A_f / dist;

                            diag += coeff;
                            triplets.emplace_back(c_idx, n_idx, -coeff);
                        }
                        else if (bc_type == BcType::FirstType) {
                            // Dirichlet BC: ghost cell T_ghost = 2*T_bc - T_cell
                            // Flux = k*A_f/half_dist * (T_ghost - T_cell)
                            //      = k*A_f/half_dist * (2*T_bc - 2*T_cell)
                            //      = 2*k*A_f/half_dist * (T_bc - T_cell)
                            // In equation: -2*k*A_f/half_dist*T_c + 2*k*A_f/half_dist*T_bc
                            // diag += 2*k*A_f/half_dist, rhs += 2*k*A_f/half_dist*T_bc
                            double half_dist;
                            switch (dir) {
                            case FaceDir::XM:
                            case FaceDir::XP:
                                half_dist = dx_cell / 2.0;
                                break;
                            case FaceDir::YM:
                            case FaceDir::YP:
                                half_dist = dy_cell / 2.0;
                                break;
                            case FaceDir::ZM:
                            case FaceDir::ZP:
                                half_dist = dz_cell / 2.0;
                                break;
                            default:
                                half_dist = 0.0;
                            }

                            double T_bc_val = bc_params.dirichlet_T[param_idx].eval(
                                {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz],
                                 state.T[c_idx], state.current_time});

                            double coeff = k * A_f / half_dist;
                            diag += 2.0 * coeff;
                            b(c_idx) += 2.0 * coeff * T_bc_val;
                        }
                        else if (bc_type == BcType::SecondType) {
                            // Neumann BC: specified heat flux q entering the cell
                            // Flux = q * A_f entering the cell -> directly adds to RHS
                            double q = bc_params.neumann_q[param_idx].eval(
                                {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz],
                                 state.T[c_idx], state.current_time});
                            b(c_idx) += q * A_f;
                        }
                        else if (bc_type == BcType::ThirdType) {
                            // Cauchy/Robin BC: combined convection + diffusion to boundary
                            // Convective flux = h * A_f * (T_inf - T)
                            // Diffusive flux to boundary = k * A_f / half_dist * (T_inf - T)
                            // Total: (k/half_dist + h) * A_f * (T_inf - T)
                            // diag += (k/half_dist + h) * A_f, rhs += (k/half_dist + h) * A_f * T_inf
                            double h = bc_params.cauchy_h[param_idx].eval(
                                {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz],
                                 state.T[c_idx], state.current_time});
                            double T_inf = bc_params.cauchy_T_inf[param_idx].eval(
                                {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz],
                                 state.T[c_idx], state.current_time});

                            double half_dist;
                            switch (dir) {
                            case FaceDir::XM:
                            case FaceDir::XP:
                                half_dist = dx_cell / 2.0;
                                break;
                            case FaceDir::YM:
                            case FaceDir::YP:
                                half_dist = dy_cell / 2.0;
                                break;
                            case FaceDir::ZM:
                            case FaceDir::ZP:
                                half_dist = dz_cell / 2.0;
                                break;
                            default:
                                half_dist = 0.0;
                            }

                            double coeff_k = k * A_f / half_dist;
                            double coeff_h = h * A_f;
                            double total_coeff = coeff_k + coeff_h;
                            diag += total_coeff;
                            b(c_idx) += total_coeff * T_inf;
                        }
                    }

                    triplets.emplace_back(c_idx, c_idx, diag);

                    // Transient term (backward Euler)
                    if (model_.study_type == StudyType::Transient && state.dt > 0.0) {
                        double rho = materials[mat_id].rho.eval(
                            {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz],
                             state.T[c_idx], state.current_time});
                        double c_heat = materials[mat_id].c.eval(
                            {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz],
                             state.T[c_idx], state.current_time});
                        double mass_coeff = rho * c_heat * vol / state.dt;
                        triplets.emplace_back(c_idx, c_idx, mass_coeff);
                        b(c_idx) += mass_coeff * state.T_prev[c_idx];
                    }
                }
            }
        }

        Eigen::SparseMatrix<double> A(N, N);
        A.setFromTriplets(triplets.begin(), triplets.end());

        // Compute residual: r = b - A*T
        Eigen::VectorXd T_vec(N);
        for (int i = 0; i < N; i++) {
            T_vec(i) = state.T[i];
        }
        Eigen::VectorXd residual_vec = b - A * T_vec;

        return {A, b, residual_vec};
    }

} // namespace mhs::assembler