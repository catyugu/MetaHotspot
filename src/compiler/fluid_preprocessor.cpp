#include "compiler/fluid_preprocessor.hpp"
#include "compiler/fluid_physics.hpp"
#include "compiler/geometry_compiler.hpp"
#include "core/csr_ops.hpp"
#include "core/vector.hpp"
#include "linear/solver.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <type_traits>

namespace mhs::compiler::fluid {
    void build_domain(mhs::model::Model& model, const mhs::core::Vector<mhs::model::FluidBoundarySpec>& boundaries,
        double scale, const FluidMaterialData& materials)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        auto& fluid = model.fluid;
        const auto& mesh = model.discrete;
        fluid = {};
        fluid.global_to_fluid.assign(mesh.cells.size(), mhs::core::invalid_id);
        mhs::core::Vector<double> viscosity;
        for (std::size_t c = 0; c < mesh.cells.size(); ++c) {
            const auto mat = model.cell_material[c];
            if (mat >= materials.initial_viscosity.size() || !materials.initial_viscosity[mat])
                continue;
            fluid.global_to_fluid[c] = static_cast<FluidCellId>(fluid.fluid_to_global.size());
            fluid.fluid_to_global.push_back(static_cast<CellId>(c));
            viscosity.push_back(*materials.initial_viscosity[mat]);
        }
        const auto n = fluid.fluid_to_global.size();
        mhs::core::detail::checked_sparse_size(n);
        if (n == 0) {
            fluid.global_to_fluid.clear();
            return;
        }
        fluid.face_volume_flux.assign(mesh.faces.size(), 0);
        fluid.interface_heat_transfer_factor.resize(n);
        mhs::core::Vector<Vec3> extents(n);
        // Rectangular-channel extents from connected fluid runs along each axis.
        for (int axis = 0; axis < 3; ++axis) {
            mhs::core::Vector<std::size_t> parent(n);
            std::iota(parent.begin(), parent.end(), 0);
            auto root = [&](std::size_t i) {
                while (parent[i] != i) {
                    parent[i] = parent[parent[i]];
                    i = parent[i];
                }
                return i;
            };
            for (const auto& f : mesh.faces) {
                if (static_cast<int>(f.axis) != axis || f.exterior())
                    continue;
                auto i = fluid.global_to_fluid[f.minus], j = fluid.global_to_fluid[f.plus];
                if ((i != mhs::core::invalid_id) && (j != mhs::core::invalid_id))
                    parent[root(i)] = root(j);
            }
            mhs::core::Vector<double> lo(n, std::numeric_limits<double>::infinity()),
                hi(n, -std::numeric_limits<double>::infinity());
            for (std::size_t i = 0; i < n; ++i) {
                auto r = root(i);
                const auto& c = mesh.cells[fluid.fluid_to_global[i]];
                lo[r] = std::min(lo[r], c.min[axis]);
                hi[r] = std::max(hi[r], c.max[axis]);
            }
            for (std::size_t i = 0; i < n; ++i)
                extents[i][axis] = hi[root(i)] - lo[root(i)];
        }
        mhs::core::Vector<double> mobility(n), rho(n);
        for (std::size_t i = 0; i < n; ++i) {
            auto lengths = extents[i];
            std::sort(lengths.begin(), lengths.end());
            double w = lengths[0], h = lengths[1], diameter = 2 * w * h / (w + h);
            fluid.interface_heat_transfer_factor[i] = nusselt_rectangular(w, h) / diameter;
            mobility[i] = diameter * diameter / (2 * f_re_rectangular(w, h) * viscosity[i]);
            const auto c = fluid.fluid_to_global[i];
            const auto p = mesh.cells[c].center();
            rho[i] = model.material_table[model.cell_material[c]].rho.eval(
                {p[0], p[1], p[2], model.initial_temperature, 0});
        }
        // Use the same rectangle intersection algorithm as thermal projection,
        // against the original mesh and an independent set of regions.
        mhs::core::Vector<CompiledBoundaryRegion> regions;
        for (std::size_t id = 0; id < boundaries.size(); ++id)
            for (const auto& r : boundaries[id].regions) {
                CompiledBoundaryRegion region;
                region.axis = r.axis;
                region.coordinate = r.coordinate * scale;
                region.condition_index = static_cast<BoundaryConditionId>(id);
                for (const auto& rect : r.rectangles)
                    region.rectangles.push_back(
                        {rect.a_min * scale, rect.a_max * scale, rect.b_min * scale, rect.b_max * scale});
                regions.push_back(std::move(region));
            }
        auto projected = project_surface_regions(mesh, regions);
        std::erase_if(projected, [&](const auto& term) {
            return fluid.global_to_fluid[mesh.faces[term.parent].cell()] == mhs::core::invalid_id;
        });
        mhs::core::Vector<double> total_area(boundaries.size(), 0);
        for (const auto& term : projected)
            total_area[term.condition] += FacePatch::area(term.bounds);
        mhs::core::Vector<double> conductance(mesh.faces.size(), 0), diagonal(n, 0);
        for (std::size_t id = 0; id < mesh.faces.size(); ++id) {
            const auto& f = mesh.faces[id];
            if (f.exterior())
                continue;
            auto i = fluid.global_to_fluid[f.minus], j = fluid.global_to_fluid[f.plus];
            if ((i == mhs::core::invalid_id) || (j == mhs::core::invalid_id))
                continue;
            const auto ii = i, jj = j;
            auto g = f.area()
                / (f.distance(mesh.cells[f.minus]) / mobility[ii] + f.distance(mesh.cells[f.plus]) / mobility[jj]);
            conductance[id] = g;
            diagonal[ii] += g;
            diagonal[jj] += g;
        }
        bool reference = false;
        mhs::core::Vector<double> boundary_conductance(projected.size(), 0);
        mhs::core::Vector<mhs::core::Triplet<double>> entries;
        mhs::core::Vector<double> rhs(n, 0.0);
        for (std::size_t t = 0; t < projected.size(); ++t) {
            const auto& term = projected[t];
            const auto& f = mesh.faces[term.parent];
            const auto i = fluid.global_to_fluid[f.cell()];
            std::visit(
                [&](const auto& bc) {
                    using T = std::decay_t<decltype(bc)>;
                    if constexpr (std::is_same_v<T, mhs::model::MassFlowInlet>)
                        rhs[i] += bc.mass_flow_rate * f.area(term.bounds) / total_area[term.condition] / rho[i];
                    else {
                        reference = true;
                        const double g = f.area(term.bounds) * mobility[i] / f.distance(mesh.cells[f.cell()]);
                        boundary_conductance[t] = g;
                        diagonal[i] += g;
                        rhs[i] += g * bc.pressure;
                    }
                },
                boundaries[term.condition].condition);
        }
        if (!reference)
            throw std::runtime_error(
                "fluid domain has no pressure reference; a pressure inlet or an outlet is required");
        for (std::size_t i = 0; i < n; ++i)
            entries.emplace_back(static_cast<SparseIndex>(i), static_cast<SparseIndex>(i), diagonal[i]);
        for (std::size_t id = 0; id < mesh.faces.size(); ++id)
            if (conductance[id] != 0) {
                const auto& f = mesh.faces[id];
                auto i = fluid.global_to_fluid[f.minus], j = fluid.global_to_fluid[f.plus];
                entries.emplace_back(static_cast<SparseIndex>(i), static_cast<SparseIndex>(j), -conductance[id]);
                entries.emplace_back(static_cast<SparseIndex>(j), static_cast<SparseIndex>(i), -conductance[id]);
            }
        auto matrix = mhs::core::build_csr(n, n, std::move(entries));
        auto solver = mhs::linear::create_solver();
        solver->compute(matrix.view());
        mhs::core::Vector<double> pressure(n, 0.0);
        const auto result = solver->solve(rhs, pressure);
        if (!result.success)
            throw std::runtime_error("fluid pressure solve failed");
        for (std::size_t t = 0; t < projected.size(); ++t) {
            const auto& term = projected[t];
            const auto& f = mesh.faces[term.parent];
            const auto i = fluid.global_to_fluid[f.cell()];
            const auto& condition = boundaries[term.condition].condition;
            const double mass = std::visit(
                [&](const auto& bc) {
                    using T = std::decay_t<decltype(bc)>;
                    if constexpr (std::is_same_v<T, mhs::model::MassFlowInlet>)
                        return -bc.mass_flow_rate * f.area(term.bounds) / total_area[term.condition];
                    else
                        return rho[i] * boundary_conductance[t] * (pressure[i] - bc.pressure);
                },
                condition);
            if (mass < 0) {
                const double temperature = std::visit(
                    [&](const auto& bc) {
                        using T = std::decay_t<decltype(bc)>;
                        if constexpr (std::is_same_v<T, mhs::model::PressureOutlet>)
                            return model.initial_temperature;
                        else
                            return bc.temperature.value_or(model.initial_temperature);
                    },
                    condition);
                fluid.boundary_terms.push_back({term.parent, FluidInflow {-mass, temperature}});
            }
            else {
                fluid.boundary_terms.push_back({term.parent, FluidOutflow {mass}});
            }
        }
        for (std::size_t id = 0; id < mesh.faces.size(); ++id)
            if (conductance[id] != 0) {
                const auto& f = mesh.faces[id];
                auto i = fluid.global_to_fluid[f.minus], j = fluid.global_to_fluid[f.plus];
                fluid.face_volume_flux[id] = (pressure[i] - pressure[j]) * conductance[id];
            }
    }
}
