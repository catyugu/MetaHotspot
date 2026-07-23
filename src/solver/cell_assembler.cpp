#include "solver/cell_assembler.hpp"

#include "runtime/mesh.hpp"

#include <cassert>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

namespace mhs::sim {
    namespace {

        struct ThreadLocalContribution {
            std::vector<Eigen::Triplet<double>> stiffness;
            std::vector<Eigen::Triplet<double>> capacity;
            std::vector<SourceEntry> source;
        };

    } // namespace

    OperatorContribution assemble_cell_domain(const mhs::core::Model& model, const AssembleContext& context)
    {
        const auto& mesh = model.mesh;
        const auto& cells = model.cells;
        const auto& bc_params = model.bc_params;
        const auto& face_bcs = model.face_bcs;
        const auto& materials = model.material_table;
        const auto cell_count = static_cast<mhs::core::Index>(cells.material_id.size());
        const auto state_offset = model.dofs.cell_states.begin;

        assert(model.dofs.cell_states.count == cell_count);
        assert(cells.cell_to_grid.size() == cells.material_id.size());

        auto thread_data
            = tbb::enumerable_thread_specific<ThreadLocalContribution>([]() { return ThreadLocalContribution {}; });
        tbb::parallel_for(tbb::blocked_range<mhs::core::Index>(0, cell_count),
            [&](const tbb::blocked_range<mhs::core::Index>& range) {
                for (mhs::core::Index cell = range.begin(); cell < range.end(); ++cell) {
                    const mhs::core::Index state = state_offset + cell;
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
                        context.state[static_cast<std::size_t>(state)], context.current_time};
                    const double kx = material.kx.eval(cell_context);
                    const double ky = material.ky.eval(cell_context);
                    const double kz = material.kz.eval(cell_context);
                    const double density = material.rho.eval(cell_context);
                    const double heat_capacity = material.c.eval(cell_context);
                    local.capacity.emplace_back(row, row, density * heat_capacity * volume);

                    const mhs::core::TableIndex source_index = cells.heat_source_idx[cell];
                    double source = model.heat_source_table[source_index].eval(cell_context) * volume;

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
                            const mhs::core::Index neighbor_state = state_offset + neighbor;
                            const mhs::core::Index nix = mhs::utils::neighbor_ix(dir, ix);
                            const mhs::core::Index niy = mhs::utils::neighbor_iy(dir, iy);
                            const mhs::core::Index niz = mhs::utils::neighbor_iz(dir, iz);
                            const auto& neighbor_material = materials[cells.material_id[neighbor]];
                            const mhs::core::FieldContext neighbor_context {mesh.cx[nix], mesh.cy[niy], mesh.cz[niz],
                                context.state[static_cast<std::size_t>(neighbor_state)], context.current_time};
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
                            source += conductance * temperature;
                            break;
                        }
                        case mhs::core::BcType::SecondType:
                            source += bc_params.neumann_q[boundary.param_idx].eval(cell_context) * area;
                            break;
                        case mhs::core::BcType::ThirdType: {
                            const double h = bc_params.cauchy_h[boundary.param_idx].eval(cell_context);
                            const double ambient = bc_params.cauchy_T_inf[boundary.param_idx].eval(cell_context);
                            const double coefficient = face_k * h * area / (face_k + h * half_distance);
                            local.stiffness.emplace_back(row, row, coefficient);
                            source += coefficient * ambient;
                            break;
                        }
                        case mhs::core::BcType::None:
                        default:
                            break;
                        }
                    }
                    local.stiffness.emplace_back(row, row, diagonal);
                    local.source.push_back({row, source});
                }
            });

        OperatorContribution contribution;
        thread_data.combine_each([&](const ThreadLocalContribution& local) {
            contribution.stiffness.insert(
                contribution.stiffness.end(), local.stiffness.begin(), local.stiffness.end());
            contribution.capacity.insert(contribution.capacity.end(), local.capacity.begin(), local.capacity.end());
            contribution.source.insert(contribution.source.end(), local.source.begin(), local.source.end());
        });
        return contribution;
    }

} // namespace mhs::sim
