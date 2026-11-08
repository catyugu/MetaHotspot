#include "solver/connection_flux.hpp"
#include "solver/thermal_reconstruction.hpp"
#include <cmath>
#include <stdexcept>

namespace mhs::solver {
    namespace {
        // Directional least squares over internal neighbours and physical
        // boundary traces. One-sided stencils reproduce linear fields.
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

        double dot(std::span<const double> weights, std::span<const double> state, const model::FieldLayout& field)
        {
            double value = 0.0;
            for (std::size_t i = 0; i < weights.size(); ++i)
                value += weights[i] * state[field.dof(i)];
            return value;
        }

        struct Side {
            double temperature = 0.0;
            double resistance = 0.0; // Per unit area, m^2 K/W
            double offset = 0.0;
            std::array<double, 2> center {};
            std::array<double, 2> tangent {};
            core::Vector<WeightedDof> weights;
        };
    } // namespace

    core::Vector<ConnectionFlux> evaluate_connection_flux(const model::CompiledModel& compiled,
        const model::ThermalConnection& connection, std::span<const double> state, double time)
    {
        if (state.size() != compiled.layout.state_count)
            throw std::invalid_argument("connection flux requires a complete state");
        const auto nfom = compiled.domains.size();
        const auto axis = static_cast<std::size_t>(connection.axis);
        auto side = [&](model::PortReference ref, model::CellId index) -> Side {
            Side result;
            if (ref.domain < nfom) {
                const auto& domain = compiled.domains.at(ref.domain);
                const auto cell = index;
                const auto& box = domain.discrete.cells.at(cell);
                const auto center = box.center();
                const auto& mat = domain.material_table.at(domain.cell_material.at(cell));
                const auto& law = axis == 0 ? mat.kx : axis == 1 ? mat.ky : mat.kz;
                const auto& field
                    = compiled.layout.field(compiled.layout.find(ref.domain, model::FieldQuantity::Temperature));
                const auto temperatures = state.subspan(field.offset, field.count);
                const double temperature = temperatures[cell];
                const double k = law.eval({center[0], center[1], center[2], temperature, time});
                if (!std::isfinite(k) || k <= 0)
                    throw std::invalid_argument("connection requires finite positive normal conductivity");
                result.resistance = box.size()[axis] / (2 * k);
                result.temperature = temperature;
                result.weights.push_back({field.dof(cell), 1.0});
                const auto grad = gradient(domain, temperatures, cell, connection.axis, time);
                std::size_t t = 0;
                for (std::size_t a = 0; a < 3; ++a)
                    if (a != axis) {
                        result.center[t] = center[a];
                        result.tangent[t] = grad[a];
                        ++t;
                    }
            }
            else {
                const auto& rom = compiled.rom_domains.at(ref.domain - nfom);
                const auto& facet = rom.ports.at(ref.port).patches.at(index);
                const auto& field
                    = compiled.layout.field(compiled.layout.find(ref.domain, model::FieldQuantity::ReducedCoordinate));
                result.resistance = facet.half_resistance;
                result.offset = facet.offset;
                result.temperature = dot(facet.trace, state, field) + facet.offset;
                result.center = facet.center;
                for (std::size_t j = 0; j < rom.order; ++j)
                    if (facet.trace[j] != 0.0)
                        result.weights.push_back({field.dof(j), facet.trace[j]});
                for (std::size_t t = 0; t < 2; ++t)
                    if (!facet.tangential_gradient[t].empty())
                        result.tangent[t] = dot(facet.tangential_gradient[t], state, field);
            }
            return result;
        };
        core::Vector<ConnectionFlux> result;
        result.reserve(connection.patches.size());
        for (const auto& patch : connection.patches) {
            const auto a = side(connection.a, patch.a);
            const auto b = side(connection.b, patch.b);
            const double area = (patch.bounds[1] - patch.bounds[0]) * (patch.bounds[3] - patch.bounds[2]);
            const double g = area / (a.resistance + b.resistance);
            if (!std::isfinite(g) || g <= 0)
                throw std::invalid_argument("connection requires finite positive conductance");
            double correction = 0.0, offset_a = 0.0;
            if (connection.nonmatching) {
                for (std::size_t t = 0; t < 2; ++t) {
                    const auto x = (patch.bounds[2 * t] + patch.bounds[2 * t + 1]) / 2;
                    const double da = (x - a.center[t]) * a.tangent[t];
                    const double db = (x - b.center[t]) * b.tangent[t];
                    offset_a += da;
                    correction += g * (da - db);
                }
            }
            const double q = g * (a.temperature - b.temperature) + correction;
            const double trace = a.temperature + offset_a - q * a.resistance / area;
            result.push_back({g, correction, g * (a.offset - b.offset), q, trace, a.weights, b.weights});
        }
        return result;
    }
} // namespace mhs::solver
