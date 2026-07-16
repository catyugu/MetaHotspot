#include "assembler.hpp"
#include "data/tolerance_config.hpp"
#include "utils/mesh_utils.hpp"
#include "utils/physics_utils.hpp"
#include <Eigen/Sparse>
#include <cassert>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

namespace mhs::sim {

    namespace {
        struct ThreadLocalData {
            std::vector<Eigen::Triplet<double>> triplets;
        };
    } // namespace

    AssemblyResult Assembler::assemble(const AssembleContext& ctx) const
    {
        const auto& mesh = model_.mesh;
        const auto& cells = model_.cells;
        const auto& bc_params = model_.bc_params;
        const auto& face_bcs = model_.face_bcs;
        const auto& materials = model_.material_table;

        const mhs::Index N = static_cast<mhs::Index>(model_.cells.material_id.size());
        const mhs::Index total = mesh.nx * mesh.ny * mesh.nz;

        auto thread_data = tbb::enumerable_thread_specific<ThreadLocalData>([]() { return ThreadLocalData {}; });

        assert(N <= static_cast<mhs::Index>(std::numeric_limits<Eigen::Index>::max()));
        const auto eigen_N = static_cast<Eigen::Index>(N);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(eigen_N);
        Eigen::VectorXd M_diag = Eigen::VectorXd::Zero(eigen_N);

        tbb::parallel_for(tbb::blocked_range<mhs::Index>(0, total), [&](const tbb::blocked_range<mhs::Index>& r) {
            for (mhs::Index old_idx = r.begin(); old_idx < r.end(); ++old_idx) {
                if (cells.index_map[old_idx] == mhs::invalidIndex)
                    continue;

                auto& local = thread_data.local();
                mhs::Index ix, iy, iz;
                mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

                mhs::Index c_idx = cells.index_map[old_idx];
                assert(c_idx != mhs::invalidIndex);
                assert(c_idx < N);
                double dx_cell = mesh.dx[ix];
                double dy_cell = mesh.dy[iy];
                double dz_cell = mesh.dz[iz];
                double vol = dx_cell * dy_cell * dz_cell;

                const auto& mp = materials[cells.material_id[c_idx]];
                const mhs::core::FieldContext ctx_c {
                    mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], ctx.T[static_cast<Eigen::Index>(c_idx)], ctx.current_time};
                const double kx_c = mp.kx.eval(ctx_c);
                const double ky_c = mp.ky.eval(ctx_c);
                const double kz_c = mp.kz.eval(ctx_c);

                const double rho = mp.rho.eval(ctx_c);
                const double c_heat = mp.c.eval(ctx_c);
                M_diag(static_cast<Eigen::Index>(c_idx)) += rho * c_heat * vol;

                uint16_t hs_idx = cells.heat_source_idx[c_idx];
                const double Q = model_.heat_source_table[hs_idx].eval(ctx_c);
                b(static_cast<Eigen::Index>(c_idx)) += Q * vol;

                double diag = 0.0;
                bool cell_is_fluid = !model_.fluid.is_fluid.empty() && c_idx < model_.fluid.is_fluid.size()
                    && model_.fluid.is_fluid[c_idx];
                const double rho_a = cell_is_fluid ? materials[cells.material_id[c_idx]].rho.eval(ctx_c) : 0.0;
                const double cp_c = cell_is_fluid ? materials[cells.material_id[c_idx]].c.eval(ctx_c) : 0.0;
                double netOutflux = 0.0;

                const auto* fc = &face_bcs[c_idx * mhs::core::FACE_COUNT];

                for (size_t f = 0; f < mhs::core::FACE_COUNT; f++) {
                    mhs::core::FaceDir dir = mhs::core::FACE_DIRS[f];

                    mhs::Index neighbor_old
                        = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
                    if (neighbor_old != mhs::invalidIndex) {
                        mhs::Index n_idx = cells.index_map[neighbor_old];
                        assert(n_idx != mhs::invalidIndex);
                        mhs::Index nix = mhs::utils::neighbor_ix(dir, ix);
                        mhs::Index niy = mhs::utils::neighbor_iy(dir, iy);
                        mhs::Index niz = mhs::utils::neighbor_iz(dir, iz);

                        const auto& mp_n = materials[cells.material_id[n_idx]];
                        const mhs::core::FieldContext ctx_n {mesh.cx[nix], mesh.cy[niy], mesh.cz[niz],
                            ctx.T[static_cast<Eigen::Index>(n_idx)], ctx.current_time};
                        double k_neighbor
                            = utils::k_along(dir, mp_n.kx.eval(ctx_n), mp_n.ky.eval(ctx_n), mp_n.kz.eval(ctx_n));

                        double d_half_neighbor
                            = mhs::utils::half_length_along(dir, mesh.dx[nix], mesh.dy[niy], mesh.dz[niz]);

                        const double A_f = mhs::utils::face_area(dir, dx_cell, dy_cell, dz_cell);
                        const double half_dist = mhs::utils::half_length_along(dir, dx_cell, dy_cell, dz_cell);
                        const double k_face = mhs::utils::k_along(dir, kx_c, ky_c, kz_c);

                        double cond = 0.0;
                        bool n_is_fluid = n_idx < model_.fluid.is_fluid.size() && model_.fluid.is_fluid[n_idx];

                        if (cell_is_fluid != n_is_fluid) {
                            mhs::Index f_id = cell_is_fluid ? c_idx : n_idx;
                            mhs::Index f_idx = model_.fluid.global_to_fluid[f_id];
                            assert(f_idx != mhs::invalidIndex);
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
                            cond = A_f / (half_dist / k_face + d_half_neighbor / k_neighbor);
                        }
                        diag += cond;
                        local.triplets.emplace_back(
                            static_cast<Eigen::Index>(c_idx), static_cast<Eigen::Index>(n_idx), -cond);

                        if (cell_is_fluid && n_is_fluid) {
                            mhs::Index f_idx = model_.fluid.global_to_fluid[c_idx];
                            assert(f_idx != mhs::invalidIndex);
                            mhs::Index fn_idx = model_.fluid.global_to_fluid[n_idx];
                            if (f_idx != mhs::invalidIndex && fn_idx != mhs::invalidIndex) {
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
                                            local.triplets.emplace_back(static_cast<Eigen::Index>(c_idx),
                                                static_cast<Eigen::Index>(c_idx), massFlux * cp_c);
                                        }
                                        else {
                                            double cp_n = mp_n.c.eval(ctx_n);
                                            local.triplets.emplace_back(static_cast<Eigen::Index>(c_idx),
                                                static_cast<Eigen::Index>(n_idx), massFlux * cp_n);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else {
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
                            local.triplets.emplace_back(
                                static_cast<Eigen::Index>(c_idx), static_cast<Eigen::Index>(c_idx), cond);
                            b(static_cast<Eigen::Index>(c_idx)) += cond * T_bc_val;
                            break;
                        }
                        case mhs::core::BcType::SecondType: {
                            double q = bc_params.neumann_q[fb.param_idx].eval(ctx_c);
                            b(static_cast<Eigen::Index>(c_idx)) += q * A_f;
                            break;
                        }
                        case mhs::core::BcType::ThirdType: {
                            double h = bc_params.cauchy_h[fb.param_idx].eval(ctx_c);
                            double T_inf = bc_params.cauchy_T_inf[fb.param_idx].eval(ctx_c);
                            double coeff = k_face * h * A_f / (k_face + h * half_dist);
                            local.triplets.emplace_back(
                                static_cast<Eigen::Index>(c_idx), static_cast<Eigen::Index>(c_idx), coeff);
                            b(static_cast<Eigen::Index>(c_idx)) += coeff * T_inf;
                            break;
                        }
                        default:
                            break;
                        }
                    }
                }

                local.triplets.emplace_back(static_cast<Eigen::Index>(c_idx), static_cast<Eigen::Index>(c_idx), diag);

                if (cell_is_fluid) {
                    mhs::Index f_idx = model_.fluid.global_to_fluid[c_idx];
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
                        if (netOutflux > 0.0 && !std::isnan(T_boundary)) {
                            b(static_cast<Eigen::Index>(c_idx)) += netOutflux * cp_c * T_boundary;
                        }
                        else {
                            local.triplets.emplace_back(
                                static_cast<Eigen::Index>(c_idx), static_cast<Eigen::Index>(c_idx), -netOutflux * cp_c);
                        }
                    }
                }
            }
        });

        std::vector<Eigen::Triplet<double>> triplets;
        thread_data.combine_each([&](const ThreadLocalData& local) {
            triplets.insert(triplets.end(), local.triplets.begin(), local.triplets.end());
        });
        thread_data.clear();

        Eigen::SparseMatrix<double> K(eigen_N, eigen_N);
        K.setFromTriplets(triplets.begin(), triplets.end());

        return {std::move(K), std::move(b), std::move(M_diag)};
    }

} // namespace mhs::sim
