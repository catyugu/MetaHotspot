#include "solver/assembler.hpp"

#include "runtime/mesh.hpp"
#include "solver/fluid_assembler.hpp"

#include <Eigen/Sparse>
#include <cassert>
#include <tbb/blocked_range.h>
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
        const mhs::Index active_count = static_cast<mhs::Index>(cells.material_id.size());

        assert(active_count <= static_cast<mhs::Index>(std::numeric_limits<Eigen::Index>::max()));
        assert(cells.cell_to_grid.size() == cells.material_id.size());
        const auto eigen_count = static_cast<Eigen::Index>(active_count);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(eigen_count);
        Eigen::VectorXd mass_diagonal = Eigen::VectorXd::Zero(eigen_count);
        auto thread_data = tbb::enumerable_thread_specific<ThreadLocalData>([]() { return ThreadLocalData {}; });

        // Base thermal path: diffusion, material mass, heat sources, and thermal
        // boundary conditions only. Fluid physics is appended as a separate
        // increment after this pass.
        tbb::parallel_for(
            tbb::blocked_range<mhs::Index>(0, active_count), [&](const tbb::blocked_range<mhs::Index>& range) {
                for (mhs::Index cell = range.begin(); cell < range.end(); ++cell) {
                    const mhs::Index old = cells.cell_to_grid[cell];
                    auto& entries = thread_data.local().triplets;
                    mhs::Index ix, iy, iz;
                    mhs::utils::decode_index(old, mesh.ny, mesh.nz, ix, iy, iz);
                    const double dx = mesh.dx[ix];
                    const double dy = mesh.dy[iy];
                    const double dz = mesh.dz[iz];
                    const double volume = dx * dy * dz;

                    const auto& material = materials[cells.material_id[cell]];
                    const mhs::core::FieldContext cell_context {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz],
                        ctx.T[static_cast<Eigen::Index>(cell)], ctx.current_time};
                    const double kx = material.kx.eval(cell_context);
                    const double ky = material.ky.eval(cell_context);
                    const double kz = material.kz.eval(cell_context);
                    const double density = material.rho.eval(cell_context);
                    const double heat_capacity = material.c.eval(cell_context);
                    mass_diagonal(static_cast<Eigen::Index>(cell)) += density * heat_capacity * volume;

                    const mhs::core::TableIndex source_index = cells.heat_source_idx[cell];
                    rhs(static_cast<Eigen::Index>(cell))
                        += model_.heat_source_table[source_index].eval(cell_context) * volume;

                    double diagonal = 0.0;
                    const auto* cell_face_bcs = &face_bcs[cell * mhs::core::FACE_COUNT];
                    for (std::size_t face = 0; face < mhs::core::FACE_COUNT; ++face) {
                        const auto dir = mhs::core::FACE_DIRS[face];
                        const mhs::Index neighbor_old = mhs::utils::neighbor_grid_index(
                            ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.grid_to_cell);
                        const double area = mhs::utils::face_area(dir, dx, dy, dz);
                        const double half_distance = mhs::utils::half_length_along(dir, dx, dy, dz);
                        const double face_k = mhs::utils::k_along(dir, kx, ky, kz);

                        if (neighbor_old != mhs::invalidIndex) {
                            const mhs::Index neighbor = cells.grid_to_cell[neighbor_old];
                            assert(neighbor != mhs::invalidIndex);
                            const mhs::Index nix = mhs::utils::neighbor_ix(dir, ix);
                            const mhs::Index niy = mhs::utils::neighbor_iy(dir, iy);
                            const mhs::Index niz = mhs::utils::neighbor_iz(dir, iz);
                            const auto& neighbor_material = materials[cells.material_id[neighbor]];
                            const mhs::core::FieldContext neighbor_context {mesh.cx[nix], mesh.cy[niy], mesh.cz[niz],
                                ctx.T[static_cast<Eigen::Index>(neighbor)], ctx.current_time};
                            const double neighbor_k
                                = mhs::utils::k_along(dir, neighbor_material.kx.eval(neighbor_context),
                                    neighbor_material.ky.eval(neighbor_context),
                                    neighbor_material.kz.eval(neighbor_context));
                            const double neighbor_half_distance
                                = mhs::utils::half_length_along(dir, mesh.dx[nix], mesh.dy[niy], mesh.dz[niz]);
                            const double conductance
                                = area / (half_distance / face_k + neighbor_half_distance / neighbor_k);
                            diagonal += conductance;
                            entries.emplace_back(
                                static_cast<Eigen::Index>(cell), static_cast<Eigen::Index>(neighbor), -conductance);
                            continue;
                        }

                        const auto& boundary = cell_face_bcs[face];
                        switch (boundary.type) {
                        case mhs::core::BcType::FirstType: {
                            const double temperature = bc_params.dirichlet_T[boundary.param_idx].eval(cell_context);
                            const double conductance = face_k * area / half_distance;
                            entries.emplace_back(
                                static_cast<Eigen::Index>(cell), static_cast<Eigen::Index>(cell), conductance);
                            rhs(static_cast<Eigen::Index>(cell)) += conductance * temperature;
                            break;
                        }
                        case mhs::core::BcType::SecondType:
                            rhs(static_cast<Eigen::Index>(cell))
                                += bc_params.neumann_q[boundary.param_idx].eval(cell_context) * area;
                            break;
                        case mhs::core::BcType::ThirdType: {
                            const double h = bc_params.cauchy_h[boundary.param_idx].eval(cell_context);
                            const double ambient = bc_params.cauchy_T_inf[boundary.param_idx].eval(cell_context);
                            const double coefficient = face_k * h * area / (face_k + h * half_distance);
                            entries.emplace_back(
                                static_cast<Eigen::Index>(cell), static_cast<Eigen::Index>(cell), coefficient);
                            rhs(static_cast<Eigen::Index>(cell)) += coefficient * ambient;
                            break;
                        }
                        case mhs::core::BcType::None:
                        default:
                            break;
                        }
                    }
                    entries.emplace_back(static_cast<Eigen::Index>(cell), static_cast<Eigen::Index>(cell), diagonal);
                }
            });

        std::vector<Eigen::Triplet<double>> triplets;
        thread_data.combine_each([&](const ThreadLocalData& local) {
            triplets.insert(triplets.end(), local.triplets.begin(), local.triplets.end());
        });

        auto fluid_increment = mhs::sim::fluid::assemble_increment(model_, ctx.T, ctx.current_time);
        triplets.insert(triplets.end(), fluid_increment.matrix_entries.begin(), fluid_increment.matrix_entries.end());
        rhs += fluid_increment.rhs;

        Eigen::SparseMatrix<double> matrix(eigen_count, eigen_count);
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        return {std::move(matrix), std::move(rhs), std::move(mass_diagonal)};
    }

} // namespace mhs::sim
