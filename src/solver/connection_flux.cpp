#include "solver/connection_flux.hpp"
#include "solver/thermal_reconstruction.hpp"
#include <cmath>
#include <stdexcept>

namespace mhs::solver {
    namespace {
        // Directional least squares over the domain's internal neighbours and
        // physical boundary traces. One-sided stencils reproduce linear fields.
        model::Vec3 gradient(const model::FomDomain& domain, std::span<const double> t, model::CellId cell,
            model::Axis normal, double time)
        {
            model::Vec3 sum {}, weights {};
            const auto center = domain.discrete.cells[cell].center();
            auto add = [&](std::size_t axis, double distance, double difference, double area) {
                if (distance != 0) {
                    sum[axis] += area * difference / distance;
                    weights[axis] += area;
                }
            };
            for (const auto id : domain.discrete.incident_faces(cell)) {
                const auto& face = domain.discrete.faces[id];
                if (face.exterior() || face.axis == normal)
                    continue;
                const auto other = face.minus == cell ? face.plus : face.minus;
                const auto axis = static_cast<std::size_t>(face.axis);
                add(axis, domain.discrete.cells[other].center()[axis] - center[axis], t[other] - t[cell], face.area());
            }
            const auto& props = domain.material_table[domain.cell_material[cell]];
            const expression::FieldContext ctx {center[0], center[1], center[2], t[cell], time};
            const model::Vec3 k {props.kx.eval(ctx), props.ky.eval(ctx), props.kz.eval(ctx)};
            for (const auto& boundary : boundary_faces_on_cell(domain, cell)) {
                if (boundary.axis == normal)
                    continue;
                const auto axis = static_cast<std::size_t>(boundary.axis);
                const auto point = boundary_center(domain.discrete.cells[cell], boundary);
                const double area
                    = (boundary.bounds[1] - boundary.bounds[0]) * (boundary.bounds[3] - boundary.bounds[2]);
                const double half = domain.discrete.cells[cell].size()[axis] / 2;
                const auto trace = evaluate_boundary(
                    domain.boundary_conditions[boundary.condition], point, area, half, t[cell], k[axis], time);
                add(axis, point[axis] - center[axis], trace.temperature - t[cell], area);
            }
            for (std::size_t axis = 0; axis < 3; ++axis)
                if (weights[axis] > 0)
                    sum[axis] /= weights[axis];
            return sum;
        }
    }
    core::Vector<ConnectionFlux> evaluate_connection_flux(const model::CompiledModel& compiled,
        const model::ThermalConnection& connection, std::span<const double> state, double time)
    {
        if (state.size() != compiled.layout.state_count)
            throw std::invalid_argument("connection flux requires a complete state");
        core::Vector<core::Vector<double>> temperatures;
        for (std::size_t d = 0; d < compiled.domains.size(); ++d) {
            auto& t = temperatures.emplace_back(compiled.domains[d].discrete.cells.size());
            const auto& field = compiled.layout.field(
                compiled.layout.find(static_cast<model::DomainId>(d), model::FieldQuantity::Temperature));
            for (std::size_t i = 0; i < field.dofs.size(); ++i)
                t.at(field.entities[i]) = state[field.dofs[i]];
        }
        return evaluate_connection_flux(compiled, connection, temperatures, time);
    }

    core::Vector<ConnectionFlux> evaluate_connection_flux(const model::CompiledModel& compiled,
        const model::ThermalConnection& connection, const core::Vector<core::Vector<double>>& t, double time)
    {
        const auto da = connection.a.domain, db = connection.b.domain;
        const auto& a = compiled.domains.at(da);
        const auto& b = compiled.domains.at(db);
        const auto axis = static_cast<std::size_t>(connection.axis);
        core::Vector<ConnectionFlux> result;
        result.reserve(connection.patches.size());
        auto conductivity = [&](const model::FomDomain& domain, model::CellId cell, double temperature) {
            const auto p = domain.discrete.cells[cell].center();
            const auto& mat = domain.material_table[domain.cell_material[cell]];
            const auto& law = axis == 0 ? mat.kx : axis == 1 ? mat.ky : mat.kz;
            const auto k = law.eval({p[0], p[1], p[2], temperature, time});
            if (!std::isfinite(k) || k <= 0)
                throw std::invalid_argument("connection requires finite positive normal conductivity");
            return k;
        };
        for (const auto& patch : connection.patches) {
            const auto& ca = a.discrete.cells[patch.a];
            const auto& cb = b.discrete.cells[patch.b];
            const double area = (patch.bounds[1] - patch.bounds[0]) * (patch.bounds[3] - patch.bounds[2]);
            const double g = area
                / (ca.size()[axis] / (2 * conductivity(a, patch.a, t[da][patch.a]))
                    + cb.size()[axis] / (2 * conductivity(b, patch.b, t[db][patch.b])));
            double correction = 0, offset_a = 0;
            if (connection.nonmatching) {
                const auto ga = gradient(a, t[da], patch.a, connection.axis, time);
                const auto gb = gradient(b, t[db], patch.b, connection.axis, time);
                const auto xa = ca.center(), xb = cb.center();
                std::size_t tangent = 0;
                for (std::size_t dim = 0; dim < 3; ++dim) {
                    if (dim == axis)
                        continue;
                    const double point = (patch.bounds[2 * tangent] + patch.bounds[2 * tangent + 1]) / 2;
                    offset_a += (point - xa[dim]) * ga[dim];
                    correction += g * ((point - xa[dim]) * ga[dim] - (point - xb[dim]) * gb[dim]);
                    ++tangent;
                }
            }
            const double q = g * (t[da][patch.a] - t[db][patch.b]) + correction;
            const double trace = t[da][patch.a] + offset_a
                - q * ca.size()[axis] / (2 * conductivity(a, patch.a, t[da][patch.a]) * area);
            result.push_back({g, correction, q, trace});
        }
        return result;
    }
}
