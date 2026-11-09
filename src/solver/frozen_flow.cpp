#include "solver/frozen_flow.hpp"
#include "core/csr_ops.hpp"
#include "core/vector.hpp"
#include "linear/solver.hpp"
#include "solver/fluid_physics.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>

namespace mhs::solver {
    mhs::solver::FrozenFlow initialize_frozen_flow(const mhs::model::FomDomain& model,
        const mhs::model::CompiledLibrary& library, std::span<const double> temperature, double time)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        const auto& materials = library.material_table;
        const auto& conditions = library.fluid_conditions;
        const auto& fluid = model.fluid;
        const auto& mesh = model.discrete;
        if (temperature.size() != mesh.cells.size())
            throw std::invalid_argument("flow initialization: temperature count differs from cell count");
        FrozenFlow flow;
        core::Vector<double> viscosity;
        for (auto c : fluid.fluid_to_global) {
            const auto p = mesh.cells[c].center();
            const auto& material = materials[model.cell_material[c]];
            const double mu = material.dynamic_viscosity->eval({p[0], p[1], p[2], temperature[c], time});
            if (!std::isfinite(mu) || mu <= 0)
                throw std::invalid_argument("fluid dynamic viscosity must be finite and positive");
            viscosity.push_back(mu);
        }
        const auto n = fluid.fluid_to_global.size();
        mhs::core::detail::checked_sparse_size(n);
        if (n == 0) {
            return flow;
        }
        flow.face_mass_flow.assign(mesh.faces.size(), 0);
        flow.interface_heat_transfer_factor.resize(n);
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
            flow.interface_heat_transfer_factor[i] = nusselt_rectangular(w, h) / diameter;
            mobility[i] = diameter * diameter / (2 * f_re_rectangular(w, h) * viscosity[i]);
            const auto c = fluid.fluid_to_global[i];
            const auto p = mesh.cells[c].center();
            rho[i] = materials[model.cell_material[c]].rho.eval({p[0], p[1], p[2], temperature[c], time});
            if (!std::isfinite(rho[i]) || rho[i] <= 0)
                throw std::invalid_argument("fluid density must be finite and positive");
            // Pressure equation is assembled in mass units, so frozen transport
            // remains conservative for spatially varying initial density.
            mobility[i] *= rho[i];
        }
        const auto& projected = model.fluid_boundaries;
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
                / ((mesh.cells[f.minus].size()[static_cast<std::size_t>(f.axis)] / 2) / mobility[ii]
                    + (mesh.cells[f.plus].size()[static_cast<std::size_t>(f.axis)] / 2) / mobility[jj]);
            conductance[id] = g;
            diagonal[ii] += g;
            diagonal[jj] += g;
        }
        core::Vector<bool> pressure_references(fluid.component_count, false);
        mhs::core::Vector<double> boundary_conductance(projected.size(), 0);
        mhs::core::Vector<mhs::core::Triplet<double>> entries;
        mhs::core::Vector<double> rhs(n, 0.0);
        for (std::size_t t = 0; t < projected.size(); ++t) {
            const auto& term = projected[t];
            const auto i = fluid.global_to_fluid[term.cell];
            std::visit(
                [&](const auto& bc) {
                    using T = std::decay_t<decltype(bc)>;
                    if constexpr (std::is_same_v<T, mhs::model::MassFlowInlet>)
                        rhs[i] += bc.mass_flow_rate
                            * ((term.bounds[1] - term.bounds[0]) * (term.bounds[3] - term.bounds[2]))
                            / model.fluid_assignments[term.assignment].area;
                    else {
                        pressure_references[fluid.component_by_fluid_cell[i]] = true;
                        const double g = ((term.bounds[1] - term.bounds[0]) * (term.bounds[3] - term.bounds[2]))
                            * mobility[i] / (mesh.cells[term.cell].size()[static_cast<std::size_t>(term.axis)] / 2);
                        boundary_conductance[t] = g;
                        diagonal[i] += g;
                        rhs[i] += g * bc.pressure;
                    }
                },
                conditions[model.fluid_assignments[term.assignment].condition]);
        }
        if (std::find(pressure_references.begin(), pressure_references.end(), false) != pressure_references.end())
            throw std::runtime_error("every connected fluid component needs a pressure inlet or outlet");
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
            const auto i = fluid.global_to_fluid[term.cell];
            const auto& condition = conditions[model.fluid_assignments[term.assignment].condition];
            const double mass = std::visit(
                [&](const auto& bc) {
                    using T = std::decay_t<decltype(bc)>;
                    if constexpr (std::is_same_v<T, mhs::model::MassFlowInlet>)
                        return -bc.mass_flow_rate
                            * ((term.bounds[1] - term.bounds[0]) * (term.bounds[3] - term.bounds[2]))
                            / model.fluid_assignments[term.assignment].area;
                    else
                        return boundary_conductance[t] * (pressure[i] - bc.pressure);
                },
                condition);
            if (mass < 0) {
                const double incoming_temperature = std::visit(
                    [&](const auto& bc) {
                        if (!bc.temperature)
                            throw std::invalid_argument(
                                "incoming fluid requires an explicit inlet/backflow temperature");
                        return *bc.temperature;
                    },
                    condition);
                flow.boundary_terms.push_back({term.cell, FluidInflow {-mass, incoming_temperature}});
            }
            else {
                flow.boundary_terms.push_back({term.cell, FluidOutflow {mass}});
            }
        }
        for (std::size_t id = 0; id < mesh.faces.size(); ++id)
            if (conductance[id] != 0) {
                const auto& f = mesh.faces[id];
                auto i = fluid.global_to_fluid[f.minus], j = fluid.global_to_fluid[f.plus];
                flow.face_mass_flow[id] = (pressure[i] - pressure[j]) * conductance[id];
            }
        return flow;
    }
}
