#include "core/boundary_geometry.hpp"
#include "solver/assembler.hpp"
#include "solver/fluid_assembler.hpp"
#include <Eigen/Sparse>
#include <stdexcept>
#include <vector>

namespace mhs::sim {
    Operators assemble_thermal(const mhs::core::Model& model, std::span<const double> temperature, double time)
    {
        using namespace mhs::core;
        if (temperature.size() != model.discrete.cells.size())
            throw std::invalid_argument("assemble_thermal: state size differs from DOF count");
        const auto n = static_cast<Eigen::Index>(model.discrete.cells.size());
        std::vector<Eigen::Triplet<double>> stiffness, capacity;
        stiffness.reserve(model.discrete.faces.size() * 4);
        capacity.reserve(model.discrete.cells.size());
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
        std::vector<Vec3> conductivity(model.discrete.cells.size());
        for (Index c = 0; c < model.discrete.cells.size(); ++c) {
            const auto row = c;
            const auto center = model.discrete.cells[c].center();
            const FieldContext ctx {center[0], center[1], center[2], temperature[row], time};
            const auto& mat = model.material_table[model.cell_material[c]];
            conductivity[c] = {mat.kx.eval(ctx), mat.ky.eval(ctx), mat.kz.eval(ctx)};
            const auto volume = model.discrete.cells[c].volume();
            capacity.emplace_back(row, row, mat.rho.eval(ctx) * mat.c.eval(ctx) * volume);
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
            }
        }
        for (const auto& term : model.boundary_terms) {
            const auto f = projected_face(model.discrete, term);
            const auto& bc = model.boundary_table[term.parameter];
            const auto row = f.owner.value;
            const auto ki = conductivity[f.owner.value][f.axis];
            const auto p = f.center();
            const FieldContext ctx {p[0], p[1], p[2], temperature[row], time};
            double g = 0;
            switch (bc.type) {
            case BcType::FirstType:
                g = ki * f.area / f.owner_distance;
                rhs[row] += g * bc.value.eval(ctx);
                break;
            case BcType::SecondType:
                rhs[row] += f.area * bc.value.eval(ctx);
                break;
            case BcType::ThirdType: {
                const double h = bc.value.eval(ctx);
                g = ki * h * f.area / (ki + h * f.owner_distance);
                rhs[row] += g * bc.ambient.eval(ctx);
                break;
            }
            default:
                break;
            }
            if (g != 0)
                stiffness.emplace_back(row, row, g);
        }
        const auto fluid = mhs::sim::fluid::assemble_increment(model, temperature, time);
        stiffness.insert(stiffness.end(), fluid.matrix_entries.begin(), fluid.matrix_entries.end());
        rhs += fluid.rhs;
        Eigen::SparseMatrix<double> K(n, n), C(n, n);
        K.setFromTriplets(stiffness.begin(), stiffness.end());
        C.setFromTriplets(capacity.begin(), capacity.end());
        return {std::move(K), std::move(C), std::move(rhs)};
    }
} // namespace mhs::sim
