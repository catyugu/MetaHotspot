#include "solver/stokes_flow.hpp"
#include "expression/expression.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <variant>

namespace mhs::solver {
    namespace {
        using namespace mhs::model;
        double patch_area(const std::array<double, 4>& b) { return (b[1] - b[0]) * (b[3] - b[2]); }
        bool close_coordinate(double a, double b)
        {
            const double scale = std::max({std::abs(a), std::abs(b), 1e-3});
            return std::abs(a - b) <= 32 * std::numeric_limits<double>::epsilon() * scale;
        }
    }

    StokesAssembler::StokesAssembler(const model::CompiledModel& compiled, model::DomainId id)
        : domain_(compiled.fom(id))
        , materials_(compiled.library.material_table)
        , compiled_library_(compiled.library)
    {
        using namespace mhs::model;
        const auto& domain = domain_;
        const auto& layout = compiled.layout;
        const auto& mesh = domain.discrete;
        const auto& fluid = domain.fluid;
        if (domain.flow_mode != FlowMode::Stokes || fluid.fluid_to_global.empty())
            throw std::invalid_argument("StokesAssembler requires a nonempty staggered flow field");

        pressure_dof_.assign(mesh.cells.size(), core::invalid_id);
        const auto& pressure = layout.field(layout.find(id, FieldQuantity::Pressure));
        for (std::size_t i = 0; i < fluid.fluid_to_global.size(); ++i)
            pressure_dof_[fluid.fluid_to_global[i]] = pressure.dof(i);

        face_dof_.assign(mesh.faces.size(), core::invalid_id);
        const auto& velocity = layout.field(layout.find(id, FieldQuantity::Velocity));
        for (std::size_t i = 0; i < fluid.velocity_faces.size(); ++i)
            face_dof_[fluid.velocity_faces[i]] = velocity.dof(i);

        boundary_area_.assign(domain.fluid_assignments.size(), 0.0);
        for (const auto& boundary : domain.fluid_boundaries)
            boundary_area_.at(boundary.condition) += patch_area(boundary.bounds);

        // Include prescribed wall and mass-flow faces as zero/known-velocity
        // nodes. Their half-dual volumes provide consistent Dirichlet diffusion
        // contributions to neighbouring unknowns without adding wall DOFs.
        for (std::size_t f = 0; f < mesh.faces.size(); ++f) {
            const auto& face = mesh.faces[f];
            const bool minus = face.minus != core::invalid_id && fluid.global_to_fluid[face.minus] != core::invalid_id;
            const bool plus = face.plus != core::invalid_id && fluid.global_to_fluid[face.plus] != core::invalid_id;
            if (!minus && !plus)
                continue;
            Node node;
            node.face = static_cast<FaceId>(f);
            node.dof = face_dof_[f];
            node.minus = minus ? face.minus : core::invalid_id;
            node.plus = plus ? face.plus : core::invalid_id;
            node.condition = fluid.face_conditions.empty() ? core::invalid_id : fluid.face_conditions[f];
            node.component = static_cast<std::size_t>(face.axis);
            const auto axis = node.component;
            // Face plane, not the neighbouring cell's centre.
            node.center[axis]
                = face.minus != core::invalid_id ? mesh.cells[face.minus].max[axis] : mesh.cells[face.plus].min[axis];
            node.low[axis] = minus ? mesh.cells[face.minus].center()[axis] : node.center[axis];
            node.high[axis] = plus ? mesh.cells[face.plus].center()[axis] : node.center[axis];
            std::size_t t = 0;
            for (std::size_t j = 0; j < 3; ++j)
                if (j != axis) {
                    node.low[j] = face.bounds[2 * t];
                    node.high[j] = face.bounds[2 * t + 1];
                    node.center[j] = (node.low[j] + node.high[j]) / 2;
                    ++t;
                }
            if (node.dof != core::invalid_id
                && !((minus && plus) || (face.exterior() && node.condition != core::invalid_id)))
                throw std::invalid_argument("Stokes velocity DOF is not supported by an open or interior face");
            nodes_.push_back(node);
        }

        // Exact face/dual-volume adjacency on the Cartesian grid. Per-axis
        // sorted rows avoid quadratic neighbour searches, and no connection
        // can cross a gap or a solid cell.
        for (std::size_t component = 0; component < 3; ++component)
            for (std::size_t direction = 0; direction < 3; ++direction) {
                std::map<std::array<double, 2>, core::Vector<std::size_t>> lines;
                for (std::size_t i = 0; i < nodes_.size(); ++i) {
                    const auto& node = nodes_[i];
                    if (node.component != component)
                        continue;
                    std::array<double, 2> key {};
                    std::size_t t = 0;
                    for (std::size_t j = 0; j < 3; ++j)
                        if (j != direction)
                            key[t++] = node.center[j];
                    lines[key].push_back(i);
                }
                for (auto& [key, line] : lines) {
                    (void)key;
                    std::sort(line.begin(), line.end(), [&](std::size_t i, std::size_t j) {
                        return nodes_[i].center[direction] < nodes_[j].center[direction];
                    });
                    for (std::size_t q = 1; q < line.size(); ++q) {
                        auto& a = nodes_[line[q - 1]];
                        auto& b = nodes_[line[q]];
                        if (!close_coordinate(a.high[direction], b.low[direction]))
                            continue;
                        double area = 1;
                        for (std::size_t j = 0; j < 3; ++j)
                            if (j != direction)
                                area *= std::max(0.0, std::min(a.high[j], b.high[j]) - std::max(a.low[j], b.low[j]));
                        if (area <= 0)
                            continue;
                        const auto da = a.high[direction] - a.center[direction];
                        const auto db = b.center[direction] - b.low[direction];
                        if (da <= 0 || db <= 0)
                            throw std::invalid_argument("invalid staggered dual-volume separation");
                        a.joined[2 * direction + 1] = true;
                        b.joined[2 * direction] = true;
                        if (a.dof != core::invalid_id || b.dof != core::invalid_id)
                            links_.push_back({line[q - 1], line[q], area, da, db});
                    }
                }
            }

        // No-slip at solid/closed boundaries; natural viscous traction at
        // pressure openings. The normal half-dual ends at the cell centre,
        // so the distance to a missing boundary velocity is twice that span.
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            const auto& node = nodes_[i];
            if (node.dof == core::invalid_id)
                continue;
            for (std::size_t j = 0; j < 3; ++j)
                for (std::size_t side = 0; side < 2; ++side) {
                    if (node.joined[2 * j + side])
                        continue;
                    const double span = side ? node.high[j] - node.center[j] : node.center[j] - node.low[j];
                    if (span <= 0) // The velocity is itself on the open boundary.
                        continue;
                    const double distance = (j == node.component ? 2 : 1) * span;
                    double area = 1;
                    for (std::size_t k = 0; k < 3; ++k)
                        if (k != j)
                            area *= node.high[k] - node.low[k];
                    walls_.push_back({i, area, distance});
                }
        }
    }

    FrozenFlow StokesAssembler::assemble(std::span<const double> state, std::span<const double> temperature,
        double time, core::Vector<core::Triplet<double>>& stiffness, core::Vector<core::Triplet<double>>& capacity,
        core::Vector<double>& load) const
    {
        using namespace mhs::model;
        const auto& mesh = domain_.discrete;
        const auto& fluid = domain_.fluid;
        if (temperature.size() != mesh.cells.size())
            throw std::invalid_argument("Stokes temperature field dimension mismatch");
        core::Vector<double> density(mesh.cells.size(), 0.0), viscosity(mesh.cells.size(), 0.0);
        for (const auto cell : fluid.fluid_to_global) {
            const auto p = mesh.cells[cell].center();
            const auto& props = materials_[domain_.cell_material[cell]];
            const mhs::expression::FieldContext context {p[0], p[1], p[2], temperature[cell], time};
            density[cell] = props.rho.eval(context);
            viscosity[cell] = props.dynamic_viscosity->eval(context);
            if (!std::isfinite(density[cell]) || density[cell] <= 0 || !std::isfinite(viscosity[cell])
                || viscosity[cell] <= 0)
                throw std::invalid_argument("Stokes density and viscosity must be finite and strictly positive");
        }

        core::Vector<double> nodal_rho(nodes_.size()), nodal_mu(nodes_.size()), nodal_velocity(nodes_.size());
        FrozenFlow flow;
        flow.face_mass_flow.assign(mesh.faces.size(), 0.0);
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            const auto& n = nodes_[i];
            const bool two = n.minus != core::invalid_id && n.plus != core::invalid_id;
            const auto cell = n.minus != core::invalid_id ? n.minus : n.plus;
            nodal_rho[i] = two ? (density[n.minus] + density[n.plus]) / 2 : density[cell];
            nodal_mu[i] = two ? (viscosity[n.minus] + viscosity[n.plus]) / 2 : viscosity[cell];
            const auto& face = mesh.faces[n.face];
            const double area = face.area();
            double velocity = 0.0;
            if (n.dof != core::invalid_id) {
                velocity = state[n.dof];
            }
            else if (n.condition != core::invalid_id) {
                if (const auto* bc = std::get_if<MassFlowInlet>(&compiled_library_.fluid_conditions[domain_.fluid_assignments[n.condition]])) {
                    const double mass = bc->mass_flow_rate * area / boundary_area_[n.condition];
                    velocity = (n.minus != core::invalid_id ? -mass : mass) / (nodal_rho[i] * area);
                }
            }
            nodal_velocity[i] = velocity;
            const double mass = nodal_rho[i] * area * velocity;
            if (two)
                flow.face_mass_flow[n.face] = mass;
            // Each active velocity has its own inertial dual-volume capacity.
            if (n.dof != core::invalid_id) {
                double volume = 1.0;
                for (std::size_t a = 0; a < 3; ++a)
                    volume *= n.high[a] - n.low[a];
                capacity.emplace_back(static_cast<core::SparseIndex>(n.dof), static_cast<core::SparseIndex>(n.dof),
                    nodal_rho[i] * volume);
                const auto row = static_cast<core::SparseIndex>(n.dof);
                if (n.minus != core::invalid_id)
                    stiffness.emplace_back(row, static_cast<core::SparseIndex>(pressure_dof_[n.minus]), -area);
                if (n.plus != core::invalid_id)
                    stiffness.emplace_back(row, static_cast<core::SparseIndex>(pressure_dof_[n.plus]), area);
                if (n.condition != core::invalid_id) {
                    const auto& bc = compiled_library_.fluid_conditions[domain_.fluid_assignments[n.condition]];
                    if (const auto* inlet_pressure = std::get_if<PressureInlet>(&bc))
                        load[row] += (n.minus == core::invalid_id ? 1.0 : -1.0) * area * inlet_pressure->pressure;
                    else if (const auto* outlet_pressure = std::get_if<PressureOutlet>(&bc))
                        load[row] += (n.minus == core::invalid_id ? 1.0 : -1.0) * area * outlet_pressure->pressure;
                }
            }
            // Conservative cell continuity: sum of signed mass fluxes is zero.
            // Prescribed mass enters as RHS; pressure openings remain unknowns.
            if (n.dof != core::invalid_id) {
                if (n.minus != core::invalid_id)
                    stiffness.emplace_back(static_cast<core::SparseIndex>(pressure_dof_[n.minus]),
                        static_cast<core::SparseIndex>(n.dof), nodal_rho[i] * area);
                if (n.plus != core::invalid_id)
                    stiffness.emplace_back(static_cast<core::SparseIndex>(pressure_dof_[n.plus]),
                        static_cast<core::SparseIndex>(n.dof), -nodal_rho[i] * area);
            }
            else if (n.condition != core::invalid_id) {
                if (const auto* bc = std::get_if<MassFlowInlet>(&compiled_library_.fluid_conditions[domain_.fluid_assignments[n.condition]]))
                    load[pressure_dof_[cell]] += bc->mass_flow_rate * area / boundary_area_[n.condition];
            }

            if (n.condition != core::invalid_id && !two) {
                const auto& condition = compiled_library_.fluid_conditions[domain_.fluid_assignments[n.condition]];
                const double outward_mass = (n.minus != core::invalid_id ? 1.0 : -1.0) * mass;
                if (outward_mass < 0) {
                    const auto temperature_bc
                        = std::visit([](const auto& bc) -> std::optional<double> { return bc.temperature; }, condition);
                    if (!temperature_bc)
                        throw std::invalid_argument(
                            "incoming Stokes flow requires an explicit inlet/backflow temperature");
                    flow.boundary_terms.push_back({cell, FluidInflow {-outward_mass, *temperature_bc}});
                }
                else
                    flow.boundary_terms.push_back({cell, FluidOutflow {outward_mass}});
            }
        }
        // One physical viscous flux contributes opposing forces to both
        // velocity control volumes, independently of pressure/temperature.
        for (const auto& link : links_) {
            const auto& a = nodes_[link.a];
            const auto& b = nodes_[link.b];
            const double conductance = link.area / (link.da / nodal_mu[link.a] + link.db / nodal_mu[link.b]);
            if (a.dof != core::invalid_id) {
                const auto row = static_cast<core::SparseIndex>(a.dof);
                stiffness.emplace_back(row, row, conductance);
                if (b.dof != core::invalid_id)
                    stiffness.emplace_back(row, static_cast<core::SparseIndex>(b.dof), -conductance);
                else
                    load[row] += conductance * nodal_velocity[link.b];
            }
            if (b.dof != core::invalid_id) {
                const auto row = static_cast<core::SparseIndex>(b.dof);
                stiffness.emplace_back(row, row, conductance);
                if (a.dof != core::invalid_id)
                    stiffness.emplace_back(row, static_cast<core::SparseIndex>(a.dof), -conductance);
                else
                    load[row] += conductance * nodal_velocity[link.a];
            }
        }
        for (const auto& wall : walls_) {
            const auto row = static_cast<core::SparseIndex>(nodes_[wall.node].dof);
            stiffness.emplace_back(row, row, wall.area * nodal_mu[wall.node] / wall.distance);
        }
        return flow;
    }
} // namespace mhs::solver
