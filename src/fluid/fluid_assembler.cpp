#include "fluid/fluid_assembler.hpp"

#include "data/tolerance_config.hpp"
#include "utils/mesh_utils.hpp"

#include <cassert>
#include <cmath>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

namespace mhs::sim::fluid {

    namespace {
        struct ThreadLocalEntries {
            std::vector<Eigen::Triplet<double>> matrix_entries;
        };

        bool is_fluid(const mhs::core::FluidDomain& fluid, mhs::Index cell)
        {
            return cell < fluid.global_to_fluid.size() && fluid.global_to_fluid[cell] != mhs::invalidIndex;
        }

        void add_interface_correction(std::vector<Eigen::Triplet<double>>& entries, mhs::Index fluid_cell,
            mhs::Index solid_cell, double correction)
        {
            const auto f = static_cast<int>(fluid_cell);
            const auto s = static_cast<int>(solid_cell);
            entries.emplace_back(f, f, correction);
            entries.emplace_back(f, s, -correction);
            entries.emplace_back(s, s, correction);
            entries.emplace_back(s, f, -correction);
        }

    } // namespace

    FluidAssemblyIncrement assemble_increment(
        const mhs::core::Model& model, Eigen::Ref<const Eigen::VectorXd> temperature, double current_time)
    {
        const mhs::Index active_count = static_cast<mhs::Index>(model.cells.material_id.size());
        assert(active_count <= static_cast<mhs::Index>(std::numeric_limits<Eigen::Index>::max()));
        FluidAssemblyIncrement result;
        result.rhs = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(active_count));
        if (model.fluid.fluid_to_global.empty())
            return result;

        auto thread_entries
            = tbb::enumerable_thread_specific<ThreadLocalEntries>([]() { return ThreadLocalEntries {}; });
        const mhs::Index fluid_count = static_cast<mhs::Index>(model.fluid.fluid_to_global.size());

        tbb::parallel_for(
            tbb::blocked_range<mhs::Index>(0, fluid_count), [&](const tbb::blocked_range<mhs::Index>& range) {
                for (mhs::Index fi = range.begin(); fi < range.end(); ++fi) {
                    const mhs::Index cell = model.fluid.fluid_to_global[fi];
                    const mhs::Index old = model.cells.cell_to_grid[cell];
                    auto& local = thread_entries.local().matrix_entries;
                    mhs::Index ix, iy, iz;
                    mhs::utils::decode_index(old, model.mesh.ny, model.mesh.nz, ix, iy, iz);
                    const double dx = model.mesh.dx[ix];
                    const double dy = model.mesh.dy[iy];
                    const double dz = model.mesh.dz[iz];

                    const auto& material = model.material_table[model.cells.material_id[cell]];
                    const mhs::core::FieldContext cell_context {model.mesh.cx[ix], model.mesh.cy[iy], model.mesh.cz[iz],
                        temperature[static_cast<Eigen::Index>(cell)], current_time};
                    const double kx = material.kx.eval(cell_context);
                    const double ky = material.ky.eval(cell_context);
                    const double kz = material.kz.eval(cell_context);
                    const double rho = material.rho.eval(cell_context);
                    const double heat_capacity = material.c.eval(cell_context);
                    double net_outflux = 0.0;

                    for (std::size_t face = 0; face < mhs::core::FACE_COUNT; ++face) {
                        const auto dir = mhs::core::FACE_DIRS[face];
                        const mhs::Index neighbor_old = mhs::utils::neighbor_grid_index(
                            ix, iy, iz, dir, model.mesh.nx, model.mesh.ny, model.mesh.nz, model.cells.grid_to_cell);
                        if (neighbor_old == mhs::invalidIndex)
                            continue;

                        const mhs::Index neighbor = model.cells.grid_to_cell[neighbor_old];
                        assert(neighbor != mhs::invalidIndex);
                        const mhs::Index nix = mhs::utils::neighbor_ix(dir, ix);
                        const mhs::Index niy = mhs::utils::neighbor_iy(dir, iy);
                        const mhs::Index niz = mhs::utils::neighbor_iz(dir, iz);
                        const auto& neighbor_material = model.material_table[model.cells.material_id[neighbor]];
                        const mhs::core::FieldContext neighbor_context {model.mesh.cx[nix], model.mesh.cy[niy],
                            model.mesh.cz[niz], temperature[static_cast<Eigen::Index>(neighbor)], current_time};

                        if (is_fluid(model.fluid, neighbor)) {
                            const double volume_flux = model.fluid.face_volume_flux[fi * mhs::core::FACE_COUNT + face];
                            if (std::abs(volume_flux) <= mhs::core::zero_guard)
                                continue;
                            const double neighbor_rho = neighbor_material.rho.eval(neighbor_context);
                            const double mass_flux = volume_flux * 0.5 * (rho + neighbor_rho);
                            net_outflux += mass_flux;
                            if (std::abs(mass_flux) <= mhs::core::zero_guard)
                                continue;
                            if (mass_flux > 0.0) {
                                local.emplace_back(
                                    static_cast<int>(cell), static_cast<int>(cell), mass_flux * heat_capacity);
                            }
                            else {
                                local.emplace_back(static_cast<int>(cell), static_cast<int>(neighbor),
                                    mass_flux * neighbor_material.c.eval(neighbor_context));
                            }
                            continue;
                        }

                        const double area = mhs::utils::face_area(dir, dx, dy, dz);
                        const double fluid_half_distance = mhs::utils::half_length_along(dir, dx, dy, dz);
                        const double solid_half_distance = mhs::utils::half_length_along(
                            dir, model.mesh.dx[nix], model.mesh.dy[niy], model.mesh.dz[niz]);
                        const double fluid_k = mhs::utils::k_along(dir, kx, ky, kz);
                        const double solid_k = mhs::utils::k_along(dir, neighbor_material.kx.eval(neighbor_context),
                            neighbor_material.ky.eval(neighbor_context), neighbor_material.kz.eval(neighbor_context));
                        const double base_conductance
                            = area / (fluid_half_distance / fluid_k + solid_half_distance / solid_k);
                        const double heat_transfer = model.fluid.interface_heat_transfer_factor[fi] * fluid_k;
                        const double interface_resistance
                            = solid_half_distance / (solid_k * area) + 1.0 / (heat_transfer * area);
                        const double interface_conductance = 1.0 / interface_resistance;
                        add_interface_correction(local, cell, neighbor, interface_conductance - base_conductance);
                    }

                    if (!std::isnan(model.fluid.boundary_outflux[fi]))
                        net_outflux = model.fluid.boundary_outflux[fi];
                    if (net_outflux == 0.0)
                        continue;

                    const double boundary_temperature = model.fluid.boundary_temperature[fi];
                    if (net_outflux > 0.0 && !std::isnan(boundary_temperature)) {
                        result.rhs(static_cast<Eigen::Index>(cell))
                            += net_outflux * heat_capacity * boundary_temperature;
                    }
                    else {
                        local.emplace_back(
                            static_cast<int>(cell), static_cast<int>(cell), -net_outflux * heat_capacity);
                    }
                }
            });

        thread_entries.combine_each([&](const ThreadLocalEntries& local) {
            result.matrix_entries.insert(
                result.matrix_entries.end(), local.matrix_entries.begin(), local.matrix_entries.end());
        });
        return result;
    }

} // namespace mhs::sim::fluid
