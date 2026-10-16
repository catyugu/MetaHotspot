#include "solver/fluid_assembler.hpp"
#include "core/constants.hpp"
#include <cmath>

namespace mhs::sim::fluid {
    FluidAssemblyIncrement assemble_increment(const mhs::core::Model& model, std::span<const double> state, double time)
    {
        using namespace mhs::core;
        FluidAssemblyIncrement result;
        result.rhs = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(model.discrete.cells.size()));
        auto context = [&](CellId c) {
            auto p = model.discrete.cells[c.value].center();
            return FieldContext {p[0], p[1], p[2], state[c.value], time};
        };
        for (Index fi = 0; fi < model.fluid.fluid_to_global.size(); ++fi) {
            const CellId cell {model.fluid.fluid_to_global[fi]};
            const auto row = cell.value;
            const auto& mat = model.material_table[model.cell_material[cell.value]];
            const auto ctx = context(cell);
            const double rho = mat.rho.eval(ctx), cp = mat.c.eval(ctx);
            for (auto id : model.discrete.incident_faces(cell)) {
                const auto& f = model.discrete.faces[id.value];
                const auto neighbor = f.other(cell);
                if (!neighbor.valid() || model.fluid.global_to_fluid[neighbor.value] == invalidIndex)
                    continue;
                const auto col = neighbor.value;
                const auto& other = model.material_table[model.cell_material[neighbor.value]];
                const auto other_ctx = context(neighbor);
                const double q = model.fluid.face_volume_flux[id.value] * (f.owner == cell ? 1 : -1);
                if (std::abs(q) <= zero_guard)
                    continue;
                const double mass = q * 0.5 * (rho + other.rho.eval(other_ctx));
                if (std::abs(mass) <= zero_guard)
                    continue;
                if (mass > 0)
                    result.matrix_entries.emplace_back(row, row, mass * cp);
                else
                    result.matrix_entries.emplace_back(row, col, mass * other.c.eval(other_ctx));
            }
        }
        for (const auto& term : model.fluid.boundary_terms) {
            const auto cell = model.discrete.faces[term.face.value].owner;
            const auto row = cell.value;
            const auto& mat = model.material_table[model.cell_material[cell.value]];
            const auto cp = mat.c.eval(context(cell));
            if (term.mass_outflux > 0)
                result.matrix_entries.emplace_back(row, row, term.mass_outflux * cp);
            else
                result.rhs[row] -= term.mass_outflux * cp * term.temperature;
        }
        return result;
    }
}
