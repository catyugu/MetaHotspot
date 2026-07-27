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

        struct AssemblySink {
            explicit AssemblySink(Eigen::Index state_count) : rhs(Eigen::VectorXd::Zero(state_count)) { }

            std::vector<Eigen::Triplet<double>> stiffness;
            std::vector<Eigen::Triplet<double>> capacity;
            Eigen::VectorXd rhs;
        };

        struct ThreadLocalData {
            std::vector<Eigen::Triplet<double>> stiffness;
            std::vector<Eigen::Triplet<double>> capacity;
        };

        void assemble_cells(const mhs::core::Model& model, const AssembleContext& ctx, AssemblySink& sink)
        {
            const auto& mesh = model.mesh;
            const auto& cells = model.cells;
            const auto& bc_params = model.bc_params;
            const auto& face_bcs = model.face_bcs;
            const auto& materials = model.material_table;
            const auto cell_count = static_cast<mhs::core::Index>(cells.material_id.size());

            assert(cells.cell_to_grid.size() == cells.material_id.size());

            auto thread_data = tbb::enumerable_thread_specific<ThreadLocalData>([]() { return ThreadLocalData {}; });
            tbb::parallel_for(tbb::blocked_range<mhs::core::Index>(0, cell_count),
                [&](const tbb::blocked_range<mhs::core::Index>& range) {
                    for (mhs::core::Index cell = range.begin(); cell < range.end(); ++cell) {
                        const mhs::core::Index state = cell;
                        const Eigen::Index row = static_cast<Eigen::Index>(state);
                        const mhs::core::Index grid = cells.cell_to_grid[cell];
                        auto& local = thread_data.local();

                        mhs::core::Index ix, iy, iz;
                        mhs::utils::decode_index(grid, mesh.ny, mesh.nz, ix, iy, iz);
                        const double dx = mesh.dx[ix];
                        const double dy = mesh.dy[iy];
                        const double dz = mesh.dz[iz];
                        const double volume = dx * dy * dz;

                        const auto& material = materials[cells.material_id[cell]];
                        const mhs::core::FieldContext cell_context {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz],
                            ctx.state[static_cast<std::size_t>(state)], ctx.current_time};
                        const double kx = material.kx.eval(cell_context);
                        const double ky = material.ky.eval(cell_context);
                        const double kz = material.kz.eval(cell_context);
                        const double density = material.rho.eval(cell_context);
                        const double heat_capacity = material.c.eval(cell_context);
                        local.capacity.emplace_back(row, row, density * heat_capacity * volume);

                        const mhs::core::TableIndex source_index = cells.heat_source_idx[cell];
                        sink.rhs(row) += model.heat_source_table[source_index].eval(cell_context) * volume;

                        double diagonal = 0.0;
                        const auto* cell_face_bcs = &face_bcs[cell * mhs::core::FACE_COUNT];
                        for (std::size_t face = 0; face < mhs::core::FACE_COUNT; ++face) {
                            const auto dir = mhs::core::FACE_DIRS[face];
                            const mhs::core::Index neighbor_grid = mhs::utils::neighbor_grid_index(
                                ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.grid_to_cell);
                            const double area = mhs::utils::face_area(dir, dx, dy, dz);
                            const double half_distance = mhs::utils::half_length_along(dir, dx, dy, dz);
                            const double face_k = mhs::utils::k_along(dir, kx, ky, kz);

                            if (neighbor_grid != mhs::core::invalidIndex) {
                                const mhs::core::Index neighbor = cells.grid_to_cell[neighbor_grid];
                                assert(neighbor != mhs::core::invalidIndex);
                                const mhs::core::Index neighbor_state = neighbor;
                                const mhs::core::Index nix = mhs::utils::neighbor_ix(dir, ix);
                                const mhs::core::Index niy = mhs::utils::neighbor_iy(dir, iy);
                                const mhs::core::Index niz = mhs::utils::neighbor_iz(dir, iz);
                                const auto& neighbor_material = materials[cells.material_id[neighbor]];
                                const mhs::core::FieldContext neighbor_context {mesh.cx[nix], mesh.cy[niy],
                                    mesh.cz[niz], ctx.state[static_cast<std::size_t>(neighbor_state)],
                                    ctx.current_time};
                                const double neighbor_k
                                    = mhs::utils::k_along(dir, neighbor_material.kx.eval(neighbor_context),
                                        neighbor_material.ky.eval(neighbor_context),
                                        neighbor_material.kz.eval(neighbor_context));
                                const double neighbor_half_distance
                                    = mhs::utils::half_length_along(dir, mesh.dx[nix], mesh.dy[niy], mesh.dz[niz]);
                                const double conductance
                                    = area / (half_distance / face_k + neighbor_half_distance / neighbor_k);
                                diagonal += conductance;
                                local.stiffness.emplace_back(
                                    row, static_cast<Eigen::Index>(neighbor_state), -conductance);
                                continue;
                            }

                            const auto& boundary = cell_face_bcs[face];
                            switch (boundary.type) {
                            case mhs::core::BcType::FirstType: {
                                const double temperature = bc_params.dirichlet_T[boundary.param_idx].eval(cell_context);
                                const double conductance = face_k * area / half_distance;
                                local.stiffness.emplace_back(row, row, conductance);
                                sink.rhs(row) += conductance * temperature;
                                break;
                            }
                            case mhs::core::BcType::SecondType:
                                sink.rhs(row) += bc_params.neumann_q[boundary.param_idx].eval(cell_context) * area;
                                break;
                            case mhs::core::BcType::ThirdType: {
                                const double h = bc_params.cauchy_h[boundary.param_idx].eval(cell_context);
                                const double ambient = bc_params.cauchy_T_inf[boundary.param_idx].eval(cell_context);
                                const double coefficient = face_k * h * area / (face_k + h * half_distance);
                                local.stiffness.emplace_back(row, row, coefficient);
                                sink.rhs(row) += coefficient * ambient;
                                break;
                            }
                            case mhs::core::BcType::None:
                            default:
                                break;
                            }
                        }
                        local.stiffness.emplace_back(row, row, diagonal);
                    }
                });

            thread_data.combine_each([&](const ThreadLocalData& local) {
                sink.stiffness.insert(sink.stiffness.end(), local.stiffness.begin(), local.stiffness.end());
                sink.capacity.insert(sink.capacity.end(), local.capacity.begin(), local.capacity.end());
            });
        }

        void assemble_fluid(const mhs::core::Model& model, const AssembleContext& ctx, AssemblySink& sink)
        {
            auto increment = mhs::sim::fluid::assemble_increment(model, ctx.state, ctx.current_time);
            const Eigen::Index offset = 0;
            for (const auto& entry : increment.matrix_entries) {
                sink.stiffness.emplace_back(entry.row() + offset, entry.col() + offset, entry.value());
            }
            sink.rhs.segment(offset, increment.rhs.size()) += increment.rhs;
        }

    } // namespace

    AssemblyResult assemble_thermal(
        const mhs::core::Model& model, const mhs::core::StateLayout& layout, const AssembleContext& ctx)
    {
        // Extract the temperature subrange from the full combined state.
        assert(layout.temperature.begin + layout.temperature.count <= static_cast<mhs::core::Index>(ctx.state.size()));
        auto temp_span = ctx.state.subspan(
            static_cast<std::size_t>(layout.temperature.begin), static_cast<std::size_t>(layout.temperature.count));
        AssembleContext thermal_ctx {temp_span, ctx.current_time};

        const auto thermal_count = layout.temperature.count;
        assert(thermal_count <= static_cast<mhs::core::Index>(std::numeric_limits<Eigen::Index>::max()));
        const auto eigen_count = static_cast<Eigen::Index>(thermal_count);

        AssemblySink sink(eigen_count);
        assemble_cells(model, thermal_ctx, sink);
        assemble_fluid(model, thermal_ctx, sink);

        Eigen::SparseMatrix<double> stiffness(eigen_count, eigen_count);
        stiffness.setFromTriplets(sink.stiffness.begin(), sink.stiffness.end());
        Eigen::SparseMatrix<double> capacity(eigen_count, eigen_count);
        capacity.setFromTriplets(sink.capacity.begin(), sink.capacity.end());
        return {std::move(stiffness), std::move(capacity), std::move(sink.rhs)};
    }

} // namespace mhs::sim
