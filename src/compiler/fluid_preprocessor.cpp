#include "compiler/fluid_preprocessor.hpp"
#include "compiler/fluid_physics.hpp"
#include "core/csr_ops.hpp"
#include "core/vector.hpp"
#include "linear/solver.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace mhs::compiler::fluid {
    void build_domain(mhs::model::Fom& model, const mhs::core::Vector<mhs::model::FluidBoundarySpec>& boundaries,
        const mhs::core::Vector<mhs::core::EntityIndex>& facet_boundary_tags,
        const mhs::core::Vector<double>& initial_viscosity)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        auto& fluid = model.fluid;
        const auto& mesh = model.mesh;
        fluid = {};
        fluid.global_to_fluid.assign(mesh.cell_count(), mhs::core::invalid_id);
        mhs::core::Vector<double> viscosity;
        for (std::size_t c = 0; c < mesh.cell_count(); ++c) {
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
        fluid.positive_volume_flux.resize(n);
        fluid.interface_heat_transfer_factor.resize(n);
        auto fluid_neighbor = [&](FluidCellId i, std::size_t axis, int sign) {
            const auto cell = mesh.neighbor(fluid.fluid_to_global[i], axis, sign);
            return cell == invalid_id ? invalid_id : fluid.global_to_fluid[cell];
        };
        auto for_each_positive_edge = [&](const auto& visit) {
            for (FluidCellId i = 0; i < n; ++i)
                for (std::size_t axis = 0; axis < 3; ++axis)
                    if (const auto j = fluid_neighbor(i, axis, 1); j != invalid_id)
                        visit(i, j, axis);
        };

        mhs::core::Vector<Vec3> extents(n);
        // A run begins at a fluid cell with no fluid neighbor in the negative direction.
        for (std::size_t axis = 0; axis < 3; ++axis)
            for (FluidCellId i = 0; i < n; ++i) {
                if (fluid_neighbor(i, axis, -1) != invalid_id)
                    continue;
                auto last = i;
                for (auto next = fluid_neighbor(last, axis, 1); next != invalid_id;
                    next = fluid_neighbor(last, axis, 1))
                    last = next;
                const double lo = mesh.cell(fluid.fluid_to_global[i]).min[axis];
                const double hi = mesh.cell(fluid.fluid_to_global[last]).max[axis];
                for (auto j = i; j != invalid_id; j = fluid_neighbor(j, axis, 1))
                    extents[j][axis] = hi - lo;
            }
        mhs::core::Vector<double> mobility(n), rho(n);
        for (std::size_t i = 0; i < n; ++i) {
            auto lengths = extents[i];
            std::sort(lengths.begin(), lengths.end());
            double w = lengths[0], h = lengths[1], diameter = 2 * w * h / (w + h);
            fluid.interface_heat_transfer_factor[i] = nusselt_rectangular(w, h) / diameter;
            mobility[i] = diameter * diameter / (2 * f_re_rectangular(w, h) * viscosity[i]);
            const auto c = fluid.fluid_to_global[i];
            const auto p = mesh.cell(c).center();
            rho[i] = model.material_table[model.cell_material[c]].rho.eval(
                {p[0], p[1], p[2], model.reference_temperature, 0});
        }
        // Use canonical exterior facets. The thermal and hydraulic laws share
        // their geometry but retain independent condition assignments.
        mhs::core::Vector<FacetId> projected;
        mhs::core::Vector<double> total_area(boundaries.size(), 0);
        for (std::size_t id = 0; id < mesh.exterior_facets.size(); ++id) {
            if (facet_boundary_tags[id] >= boundaries.size())
                continue;
            const auto cell = mesh.exterior_facets[id].owner;
            if (fluid.global_to_fluid[cell] == invalid_id)
                continue;
            projected.push_back(static_cast<FacetId>(id));
            total_area[facet_boundary_tags[id]] += mesh.exterior_facets[id].geometry.area();
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

        mhs::core::Vector<double> diagonal(system_size, 0);
        mhs::core::Vector<mhs::core::Triplet<double>> entries;
        // Reuse the flux slots for hydraulic conductances until pressures are available.
        for_each_positive_edge([&](FluidCellId i, FluidCellId j, std::size_t axis) {
            const auto left = fluid.fluid_to_global[i], right = fluid.fluid_to_global[j];
            const double g = mesh.cell(left).face_area(axis)
                / ((mesh.cell_width(left, axis) / 2) / mobility[i] + (mesh.cell_width(right, axis) / 2) / mobility[j]);
            fluid.positive_volume_flux[i][axis] = g;
            diagonal[i] += g;
            diagonal[j] += g;
            entries.emplace_back(static_cast<SparseIndex>(i), static_cast<SparseIndex>(j), -g);
            entries.emplace_back(static_cast<SparseIndex>(j), static_cast<SparseIndex>(i), -g);
        });
        bool reference = false;
        mhs::core::Vector<double> boundary_conductance(projected.size(), 0);
        mhs::core::Vector<double> rhs(system_size, 0.0);
        for (std::size_t t = 0; t < projected.size(); ++t) {
            const auto facet_id = projected[t];
            const auto& face = mesh.exterior_facets[facet_id];
            const auto& facet = face.geometry;
            const auto tag = facet_boundary_tags[facet_id];
            const auto i = fluid.global_to_fluid[face.owner];
            std::visit(
                [&](const auto& bc) {
                    using T = std::decay_t<decltype(bc)>;
                    if constexpr (std::is_same_v<T, mhs::model::MassFlowInlet>)
                        rhs[i] += bc.mass_flow_rate * facet.area() / total_area[tag] / rho[i];
                    else {
                        reference = true;
                        const auto axis = static_cast<std::size_t>(facet.axis);
                        const double distance = mesh.cell_width(face.owner, axis) / 2;
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
        auto matrix = mhs::core::build_csr(system_size, system_size, std::move(entries));
        auto solver = mhs::linear::create_solver();
        solver->compute(matrix.view());
        mhs::core::Vector<double> pressure(system_size, 0.0);
        const auto result = solver->solve(rhs, pressure);
        if (!result.success)
            throw std::runtime_error("fluid pressure solve failed");
        for (std::size_t t = 0; t < projected.size(); ++t) {
            const auto facet_id = projected[t];
            const auto& face = mesh.exterior_facets[facet_id];
            const auto& facet = face.geometry;
            const auto tag = facet_boundary_tags[facet_id];
            const auto i = fluid.global_to_fluid[face.owner];
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
            double temperature = model.reference_temperature;
            if (mass < 0)
                temperature = std::visit(
                    [&](const auto& bc) {
                        using T = std::decay_t<decltype(bc)>;
                        if constexpr (std::is_same_v<T, mhs::model::PressureOutlet>)
                            return model.reference_temperature;
                        else
                            return bc.temperature.value_or(model.reference_temperature);
                    },
                    condition);
            fluid.boundary_terms.push_back({face.owner, mass, temperature});
        }
        for_each_positive_edge([&](FluidCellId i, FluidCellId j, std::size_t axis) {
            fluid.positive_volume_flux[i][axis] *= pressure[i] - pressure[j];
        });
    }
}
