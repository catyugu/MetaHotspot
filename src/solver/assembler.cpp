#include "solver/assembler.hpp"
#include "core/vector.hpp"
#include "numerics/sparse_ops.hpp"
#include "thermal_reconstruction.hpp"
#include <stdexcept>
#include <type_traits>

namespace mhs::sim {
    Operators assemble_thermal(const mhs::core::Model& model, std::span<const double> temperature, double time)
    {
        using namespace mhs::core;
        using namespace mhs::numerics;
        if (temperature.size() != model.discrete.cells.size())
            throw std::invalid_argument("assemble_thermal: state size differs from DOF count");
        const auto n = model.discrete.cells.size();
        mhs::core::Vector<mhs::numerics::Triplet> stiffness, capacity;
        stiffness.reserve(model.discrete.faces.size() * 4);
        capacity.reserve(model.discrete.cells.size());
        mhs::core::Vector<double> rhs(n, 0.0);
        mhs::core::Vector<Vec3> conductivity(model.discrete.cells.size());
        mhs::core::Vector<double> density(model.fluid.fluid_to_global.size()), specific_heat(density.size());
        for (Index c = 0; c < model.discrete.cells.size(); ++c) {
            const auto row = c;
            const auto center = model.discrete.cells[c].center();
            const FieldContext ctx {center[0], center[1], center[2], temperature[row], time};
            const auto& mat = model.material_table[model.cell_material[c]];
            conductivity[c] = {mat.kx.eval(ctx), mat.ky.eval(ctx), mat.kz.eval(ctx)};
            const auto volume = model.discrete.cells[c].volume();
            const double rho = mat.rho.eval(ctx), cp = mat.c.eval(ctx);
            capacity.emplace_back(row, row, rho * cp * volume);
            if (!density.empty()) {
                const auto fi = model.fluid.global_to_fluid[c];
                if (fi != invalidIndex) {
                    density[fi] = rho;
                    specific_heat[fi] = cp;
                }
            }
        }
        for (const auto& term : model.source_terms) {
            const auto row = term.cell.value;
            const auto& p = term.point;
            rhs[row]
                += term.weight * model.heat_source_table[term.source].eval({p[0], p[1], p[2], temperature[row], time});
        }
        auto fluid_index = [&](CellId c) {
            return c.value < model.fluid.global_to_fluid.size() ? model.fluid.global_to_fluid[c.value] : invalidIndex;
        };
        for (Index id = 0; id < model.discrete.faces.size(); ++id) {
            const auto& f = model.discrete.faces[id];
            const auto i = f.owner.value;
            const auto row = f.owner.value;
            const auto ki = conductivity[i][f.axis];
            if (f.neighbor.valid()) {
                const auto j = f.neighbor.value;
                const auto col = f.neighbor.value;
                const auto kj = conductivity[j][f.axis];
                auto resistance = f.owner_distance / ki + f.neighbor_distance / kj;
                const auto fi = fluid_index(f.owner), fj = fluid_index(f.neighbor);
                if ((fi == invalidIndex) != (fj == invalidIndex)) {
                    const auto fluid = fi != invalidIndex ? fi : fj;
                    const double kfluid = fi != invalidIndex ? ki : kj;
                    const double ksolid = fi != invalidIndex ? kj : ki;
                    const double distance = fi != invalidIndex ? f.neighbor_distance : f.owner_distance;
                    const double h = model.fluid.interface_heat_transfer_factor[fluid] * kfluid;
                    resistance = distance / ksolid + 1.0 / h;
                }
                const double g = f.area / resistance;
                stiffness.emplace_back(row, row, g);
                stiffness.emplace_back(row, col, -g);
                stiffness.emplace_back(col, col, g);
                stiffness.emplace_back(col, row, -g);
                if (fi != invalidIndex && fj != invalidIndex) {
                    const double mass = model.fluid.face_volume_flux[id] * 0.5 * (density[fi] + density[fj]);
                    if (mass != 0.0) {
                        const auto upstream = mass > 0 ? row : col;
                        const double enthalpy = mass * specific_heat[mass > 0 ? fi : fj];
                        stiffness.emplace_back(row, upstream, enthalpy);
                        stiffness.emplace_back(col, upstream, -enthalpy);
                    }
                }
            }
        }
        for (const auto& term : model.boundary_faces) {
            const auto f = model.discrete.faces[term.parent.value].project(term.bounds);
            const auto row = f.owner.value;
            const auto trace = evaluate_boundary(model.boundary_conditions[term.condition], f, temperature[row],
                conductivity[f.owner.value][f.axis], time);
            rhs[row] += trace.load;
            if (trace.conductance != 0)
                stiffness.emplace_back(row, row, trace.conductance);
        }
        for (const auto& term : model.fluid.boundary_terms) {
            const auto cell = model.discrete.faces[term.face.value].owner;
            const auto row = cell.value;
            const auto cp = specific_heat[model.fluid.global_to_fluid[cell.value]];
            std::visit(
                [&](const auto& flow) {
                    using T = std::decay_t<decltype(flow)>;
                    if constexpr (std::is_same_v<T, FluidInflow>)
                        rhs[row] += flow.mass_flow_rate * cp * flow.temperature;
                    else
                        stiffness.emplace_back(row, row, flow.mass_flow_rate * cp);
                },
                term.condition);
        }
        auto K = build_csr(n, n, std::move(stiffness));
        auto C = build_csr(n, n, std::move(capacity));
        return {std::move(K), std::move(C), std::move(rhs)};
    }
} // namespace mhs::sim
