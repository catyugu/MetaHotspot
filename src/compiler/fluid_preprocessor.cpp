#include "compiler/fluid_preprocessor.hpp"
#include "compiler/fluid_physics.hpp"
#include "compiler/geometry_compiler.hpp"
#include "core/csr_ops.hpp"
#include "core/vector.hpp"
#include "linear/solver.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace mhs::compiler::fluid {
    void build_domain(mhs::model::Model& model, const mhs::core::Vector<mhs::model::FluidBoundarySpec>& boundaries,
        const mhs::core::Vector<mhs::core::EntityIndex>& facet_boundary_tags,
        const mhs::core::Vector<double>& initial_viscosity)
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
            if (mat >= initial_viscosity.size() || model.material_table[mat].kind != MaterialKind::Fluid)
                continue;
            fluid.global_to_fluid[c] = static_cast<FluidCellId>(fluid.fluid_to_global.size());
            fluid.fluid_to_global.push_back(static_cast<CellId>(c));
            viscosity.push_back(initial_viscosity[mat]);
        }
        const auto n = fluid.fluid_to_global.size();
        mhs::core::detail::checked_sparse_size(n);
        if (n == 0) {
            fluid.global_to_fluid.clear();
            return;
        }
        fluid.facet_volume_flux.assign(mesh.facets.size(), 0);
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
            for (const auto& f : mesh.facets) {
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
                {p[0], p[1], p[2], model.reference_temperature, 0});
        }
        // Use canonical exterior facets. The thermal and hydraulic laws share
        // their geometry but retain independent condition assignments.
        mhs::core::Vector<FacetId> projected;
        mhs::core::Vector<double> total_area(boundaries.size(), 0);
        for (std::size_t id = 0; id < mesh.facets.size(); ++id) {
            if (facet_boundary_tags[id] >= boundaries.size() || !mesh.facets[id].exterior())
                continue;
            const auto cell = mesh.facets[id].owner();
            if (fluid.global_to_fluid[cell] == invalid_id)
                continue;
            projected.push_back(static_cast<FacetId>(id));
            total_area[facet_boundary_tags[id]] += mesh.facets[id].area();
        }
        // Pump resistance is in series with the entire inlet patch. Each
        // resistive inlet uses one auxiliary pressure, shared by its facets.
        constexpr std::size_t no_pump = std::numeric_limits<std::size_t>::max();
        mhs::core::Vector<std::size_t> pump_dof(boundaries.size(), no_pump);
        std::size_t system_size = n;
        for (std::size_t tag = 0; tag < boundaries.size(); ++tag) {
            const auto* inlet = std::get_if<PressureInlet>(&boundaries[tag].condition);
            if (!inlet)
                continue;
            if (!std::isfinite(inlet->pump_resistance) || inlet->pump_resistance < 0)
                throw std::invalid_argument("pressure inlet PumpResistance must be finite and nonnegative");
            if (inlet->pump_resistance > 0 && total_area[tag] > 0)
                pump_dof[tag] = system_size++;
        }
        mhs::core::detail::checked_sparse_size(system_size);

        mhs::core::Vector<double> conductance(mesh.facets.size(), 0), diagonal(system_size, 0);
        for (std::size_t id = 0; id < mesh.facets.size(); ++id) {
            const auto& f = mesh.facets[id];
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
        bool reference = false;
        mhs::core::Vector<double> boundary_conductance(projected.size(), 0);
        mhs::core::Vector<mhs::core::Triplet<double>> entries;
        mhs::core::Vector<double> rhs(system_size, 0.0);
        for (std::size_t t = 0; t < projected.size(); ++t) {
            const auto facet_id = projected[t];
            const auto& facet = mesh.facets[facet_id];
            const auto tag = facet_boundary_tags[facet_id];
            const auto i = fluid.global_to_fluid[facet.owner()];
            std::visit(
                [&](const auto& bc) {
                    using T = std::decay_t<decltype(bc)>;
                    if constexpr (std::is_same_v<T, mhs::model::MassFlowInlet>)
                        rhs[i] += bc.mass_flow_rate * facet.area() / total_area[tag] / rho[i];
                    else {
                        reference = true;
                        const auto axis = static_cast<std::size_t>(facet.axis);
                        const double distance = mesh.cells[facet.owner()].size()[axis] / 2;
                        const double g = facet.area() * mobility[i] / distance;
                        boundary_conductance[t] = g;
                        diagonal[i] += g;
                        if (const auto port = pump_dof[tag]; port != no_pump) {
                            diagonal[port] += g;
                            entries.emplace_back(static_cast<SparseIndex>(i), static_cast<SparseIndex>(port), -g);
                            entries.emplace_back(static_cast<SparseIndex>(port), static_cast<SparseIndex>(i), -g);
                        }
                        else
                            rhs[i] += g * bc.pressure;
                    }
                },
                boundaries[tag].condition);
        }
        for (std::size_t tag = 0; tag < boundaries.size(); ++tag) {
            const auto port = pump_dof[tag];
            if (port == no_pump)
                continue;
            const auto& inlet = std::get<PressureInlet>(boundaries[tag].condition);
            const double g = 1.0 / inlet.pump_resistance;
            diagonal[port] += g;
            rhs[port] += g * inlet.pressure;
        }
        if (!reference)
            throw std::runtime_error(
                "fluid domain has no pressure reference; a pressure inlet or an outlet is required");
        for (std::size_t i = 0; i < system_size; ++i)
            entries.emplace_back(static_cast<SparseIndex>(i), static_cast<SparseIndex>(i), diagonal[i]);
        for (std::size_t id = 0; id < mesh.facets.size(); ++id)
            if (conductance[id] != 0) {
                const auto& f = mesh.facets[id];
                auto i = fluid.global_to_fluid[f.minus], j = fluid.global_to_fluid[f.plus];
                entries.emplace_back(static_cast<SparseIndex>(i), static_cast<SparseIndex>(j), -conductance[id]);
                entries.emplace_back(static_cast<SparseIndex>(j), static_cast<SparseIndex>(i), -conductance[id]);
            }
        auto matrix = mhs::core::build_csr(system_size, system_size, std::move(entries));
        auto solver = mhs::linear::create_solver();
        solver->compute(matrix.view());
        mhs::core::Vector<double> pressure(system_size, 0.0);
        const auto result = solver->solve(rhs, pressure);
        if (!result.success)
            throw std::runtime_error("fluid pressure solve failed");
        for (std::size_t t = 0; t < projected.size(); ++t) {
            const auto facet_id = projected[t];
            const auto& facet = mesh.facets[facet_id];
            const auto tag = facet_boundary_tags[facet_id];
            const auto i = fluid.global_to_fluid[facet.owner()];
            const auto& condition = boundaries[tag].condition;
            const double mass = std::visit(
                [&](const auto& bc) {
                    using T = std::decay_t<decltype(bc)>;
                    if constexpr (std::is_same_v<T, mhs::model::MassFlowInlet>)
                        return -bc.mass_flow_rate * facet.area() / total_area[tag];
                    else {
                        const auto port = pump_dof[tag];
                        const double boundary_pressure = port == no_pump ? bc.pressure : pressure[port];
                        return rho[i] * boundary_conductance[t] * (pressure[i] - boundary_pressure);
                    }
                },
                condition);
            if (mass < 0) {
                const double temperature = std::visit(
                    [&](const auto& bc) {
                        using T = std::decay_t<decltype(bc)>;
                        if constexpr (std::is_same_v<T, mhs::model::PressureOutlet>)
                            return model.reference_temperature;
                        else
                            return bc.temperature.value_or(model.reference_temperature);
                    },
                    condition);
                fluid.boundary_terms.push_back({facet_id, FluidInflow {-mass, temperature}});
            }
            else {
                fluid.boundary_terms.push_back({facet_id, FluidOutflow {mass}});
            }
        }
        for (std::size_t id = 0; id < mesh.facets.size(); ++id)
            if (conductance[id] != 0) {
                const auto& f = mesh.facets[id];
                auto i = fluid.global_to_fluid[f.minus], j = fluid.global_to_fluid[f.plus];
                fluid.facet_volume_flux[id] = (pressure[i] - pressure[j]) * conductance[id];
            }
    }
}
