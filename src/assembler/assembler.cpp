#include "assembler.hpp"
#include "expr/expr.hpp"

namespace mhs {

namespace {

int to_1d(int i, int j, int k, int nx, int ny)
{
    return k * nx * ny + j * nx + i;
}

} // namespace

AssemblerResult Assembler::assemble(const model::InternalModel& model,
                                    const std::vector<double>& T,
                                    double t)
{
    const auto& mesh = model.mesh;
    const auto& cells = model.cells;
    const auto& face_bcs = model.face_bcs;
    const auto& bc_params = model.bc_params;

    int nx = mesh.nx;
    int ny = mesh.ny;
    int nz = mesh.nz;
    int cell_count = mesh.cell_count;

    // Create sparse matrix using triplet format
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<size_t>(cell_count) * 7);

    Eigen::VectorXd b(cell_count);
    b.setZero();

    // Assemble each cell
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int cell_idx = to_1d(i, j, k, nx, ny);

                // Get material properties
                size_t mat_id = cells.material_id[cell_idx];
                if (mat_id >= model.material_table.size()) {
                    mat_id = 0;
                }
                const auto& mat_props = model.material_table[mat_id];

                // Evaluate thermal conductivity
                mhs::FieldContext ctx;
                ctx.x = mesh.cx[i];
                ctx.y = mesh.cy[j];
                ctx.z = mesh.cz[k];
                ctx.T = T[cell_idx];
                ctx.t = t;

                double k_val = mat_props.k.eval(ctx);
                // rho and c reserved for transient analysis
                std::ignore = mat_props.rho.eval(ctx);
                std::ignore = mat_props.c.eval(ctx);

                // Cell volume
                double dx = mesh.dx[i];
                double dy = mesh.dy[j];
                double dz = mesh.dz[k];
                double volume = dx * dy * dz;

                // Initialize matrix row with diagonal
                double diag = 0.0;

                // Heat source
                double Q = 0.0;
                if (cell_idx < static_cast<int>(cells.heat_source.size())) {
                    Q = cells.heat_source[cell_idx].eval(ctx);
                }

                // Volume source term
                b(cell_idx) += Q * volume;

                // Neighbors and boundary contributions
                auto add_flux = [&](int di, int dj, int dk, double area, bool is_bc,
                                    BcType bc_type, uint16_t bc_param_idx) {
                    double flux = 0.0;
                    double neighbor_T = 0.0;

                    if (is_bc) {
                        switch (bc_type) {
                            case BcType::FirstType: {
                                // Dirichlet BC: ghost cell method
                                // T_ghost = 2*T_dirichlet - T_boundary
                                if (bc_param_idx < bc_params.dirichlet_T.size()) {
                                    double T_bc = bc_params.dirichlet_T[bc_param_idx].eval(ctx);
                                    double T_boundary = T[cell_idx];
                                    neighbor_T = 2.0 * T_bc - T_boundary;
                                }
                                break;
                            }
                            case BcType::SecondType: {
                                // Neumann BC: heat flux
                                if (bc_param_idx < bc_params.neumann_q.size()) {
                                    flux = bc_params.neumann_q[bc_param_idx].eval(ctx);
                                    b(cell_idx) -= flux * area; // Negative because flux leaves domain
                                }
                                break;
                            }
                            case BcType::ThirdType: {
                                // Robin/Cauchy BC: -k * dT/dn = h*(T - T_inf)
                                if (bc_param_idx < bc_params.cauchy_h.size()) {
                                    double h = bc_params.cauchy_h[bc_param_idx].eval(ctx);
                                    double T_inf = bc_params.cauchy_T_inf[bc_param_idx].eval(ctx);
                                    // Linearization: h*A*Tn - h*A*T_inf
                                    diag += h * area;
                                    b(cell_idx) += h * area * T_inf;
                                }
                                break;
                            }
                            default:
                                break;
                        }
                    } else {
                        // Interior neighbor
                        int ni = i + di;
                        int nj = j + dj;
                        int nk = k + dk;

                        if (ni >= 0 && ni < nx && nj >= 0 && nj < ny && nk >= 0 && nk < nz) {
                            int neighbor_idx = to_1d(ni, nj, nk, nx, ny);
                            neighbor_T = T[neighbor_idx];

                            // Get neighbor material
                            size_t n_mat_id = cells.material_id[neighbor_idx];
                            if (n_mat_id >= model.material_table.size()) {
                                n_mat_id = 0;
                            }
                            const auto& n_mat = model.material_table[n_mat_id];

                            mhs::FieldContext nctx = ctx;
                            nctx.x = mesh.cx[ni];
                            nctx.y = mesh.cy[nj];
                            nctx.z = mesh.cz[nk];
                            nctx.T = neighbor_T;

                            double k_neighbor = n_mat.k.eval(nctx);
                            double k_avg = (k_val + k_neighbor) * 0.5;

                            // Harmonic mean for conductance
                            if (k_avg > 0) {
                                double conductance = k_avg * area / get_distance(di, dj, dk, mesh, i, j, k);
                                diag += conductance;
                                triplets.emplace_back(cell_idx, neighbor_idx, -conductance);
                            }
                        }
                    }

                    if (!is_bc && neighbor_T != 0.0) {
                        // Already handled via triplet
                    }
                };

                // Calculate face areas and distances
                double dy_dz = dy * dz;
                double dx_dz = dx * dz;
                double dx_dy = dx * dy;

                // Get boundary conditions for this cell's faces
                size_t xy_idx = static_cast<size_t>(j) * nx + i;
                size_t xz_idx = static_cast<size_t>(j) * nx + i;
                size_t yz_idx = static_cast<size_t>(k) * ny + j;

                // X- face
                bool is_xm_bc = face_bcs.bc_type_xm[yz_idx] != BcType::None;
                if (i == 0 || is_xm_bc) {
                    add_flux(-1, 0, 0, dy_dz, is_xm_bc || i == 0,
                              is_xm_bc ? face_bcs.bc_type_xm[yz_idx] : BcType::FirstType,
                              is_xm_bc ? face_bcs.bc_param_idx_xm[yz_idx] : 0);
                } else {
                    add_flux(-1, 0, 0, dy_dz, false, BcType::None, 0);
                }

                // X+ face
                bool is_xp_bc = face_bcs.bc_type_xp[yz_idx] != BcType::None;
                if (i == nx - 1 || is_xp_bc) {
                    add_flux(1, 0, 0, dy_dz, is_xp_bc || i == nx - 1,
                              is_xp_bc ? face_bcs.bc_type_xp[yz_idx] : BcType::FirstType,
                              is_xp_bc ? face_bcs.bc_param_idx_xp[yz_idx] : 0);
                } else {
                    add_flux(1, 0, 0, dy_dz, false, BcType::None, 0);
                }

                // Y- face
                bool is_ym_bc = face_bcs.bc_type_ym[xz_idx] != BcType::None;
                if (j == 0 || is_ym_bc) {
                    add_flux(0, -1, 0, dx_dz, is_ym_bc || j == 0,
                              is_ym_bc ? face_bcs.bc_type_ym[xz_idx] : BcType::FirstType,
                              is_ym_bc ? face_bcs.bc_param_idx_ym[xz_idx] : 0);
                } else {
                    add_flux(0, -1, 0, dx_dz, false, BcType::None, 0);
                }

                // Y+ face
                bool is_yp_bc = face_bcs.bc_type_yp[xz_idx] != BcType::None;
                if (j == ny - 1 || is_yp_bc) {
                    add_flux(0, 1, 0, dx_dz, is_yp_bc || j == ny - 1,
                              is_yp_bc ? face_bcs.bc_type_yp[xz_idx] : BcType::FirstType,
                              is_yp_bc ? face_bcs.bc_param_idx_yp[xz_idx] : 0);
                } else {
                    add_flux(0, 1, 0, dx_dz, false, BcType::None, 0);
                }

                // Z- face
                bool is_zm_bc = face_bcs.bc_type_zm[xy_idx] != BcType::None;
                if (k == 0 || is_zm_bc) {
                    add_flux(0, 0, -1, dx_dy, is_zm_bc || k == 0,
                              is_zm_bc ? face_bcs.bc_type_zm[xy_idx] : BcType::FirstType,
                              is_zm_bc ? face_bcs.bc_param_idx_zm[xy_idx] : 0);
                } else {
                    add_flux(0, 0, -1, dx_dy, false, BcType::None, 0);
                }

                // Z+ face
                bool is_zp_bc = face_bcs.bc_type_zp[xy_idx] != BcType::None;
                if (k == nz - 1 || is_zp_bc) {
                    add_flux(0, 0, 1, dx_dy, is_zp_bc || k == nz - 1,
                              is_zp_bc ? face_bcs.bc_type_zp[xy_idx] : BcType::FirstType,
                              is_zp_bc ? face_bcs.bc_param_idx_zp[xy_idx] : 0);
                } else {
                    add_flux(0, 0, 1, dx_dy, false, BcType::None, 0);
                }

                // Add diagonal term
                triplets.emplace_back(cell_idx, cell_idx, diag);
            }
        }
    }

    Eigen::SparseMatrix<double> A(cell_count, cell_count);
    A.setFromTriplets(triplets.begin(), triplets.end());

    return {A, b};
}

double Assembler::get_distance(int di, int dj, int dk,
                               const model::MeshGeometry& mesh,
                               int i, int j, int k)
{
    double dist = 0.0;
    if (di != 0) {
        dist = mesh.dx[i] * (di > 0 ? 1.0 : -1.0);
    } else if (dj != 0) {
        dist = mesh.dy[j] * (dj > 0 ? 1.0 : -1.0);
    } else if (dk != 0) {
        dist = mesh.dz[k] * (dk > 0 ? 1.0 : -1.0);
    }
    return std::abs(dist) + 1e-10; // Avoid division by zero
}

} // namespace mhs