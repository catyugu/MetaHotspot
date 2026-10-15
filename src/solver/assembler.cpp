#include "solver/assembler.hpp"
#include "solver/fluid_assembler.hpp"
#include <Eigen/Sparse>
#include <stdexcept>
#include <vector>

namespace mhs::sim {
    Operators assemble_thermal(const mhs::core::Model &model, std::span<const double> temperature, double time) {
        using namespace mhs::core;
        if (temperature.size() != model.dofs.size())
            throw std::invalid_argument("assemble_thermal: state size differs from DOF count");
        const auto n = static_cast<Eigen::Index>(model.dofs.size());
        std::vector<Eigen::Triplet<double>> stiffness, capacity;
        stiffness.reserve(model.discrete.faces.size() * 4);
        capacity.reserve(model.discrete.cells.size());
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
        std::vector<Vec3> conductivity(model.discrete.cells.size());
        for (Index c = 0; c < model.discrete.cells.size(); ++c) {
            const auto row = model.dofs.temperature(CellId{c}).value;
            const auto center = model.discrete.cells[c].center();
            const FieldContext ctx{center[0], center[1], center[2], temperature[row], time};
            const auto &mat = model.material_table[model.cells.material_id[c]];
            conductivity[c] = {mat.kx.eval(ctx), mat.ky.eval(ctx), mat.kz.eval(ctx)};
            const auto volume = model.discrete.cells[c].volume();
            capacity.emplace_back(row, row, mat.rho.eval(ctx) * mat.c.eval(ctx) * volume);
        }
        for (const auto &term : model.volume_projection) {
            const auto row = model.dofs.temperature(term.cell).value;
            const auto &p = term.centroid;
            rhs[row] +=
                term.volume * model.heat_source_table[term.source].eval({p[0], p[1], p[2], temperature[row], time});
        }
        auto fluid_index = [&](CellId c) {
            return c.value < model.fluid.global_to_fluid.size() ? model.fluid.global_to_fluid[c.value] : invalidIndex;
        };
        for (Index id = 0; id < model.discrete.faces.size(); ++id) {
            const auto &f = model.discrete.faces[id];
            const auto i = f.owner.value;
            const auto row = model.dofs.temperature(f.owner).value;
            const auto ki = conductivity[i][f.axis];
            if (f.neighbor.valid()) {
                const auto j = f.neighbor.value;
                const auto col = model.dofs.temperature(f.neighbor).value;
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
            } else {
                const auto &bc = model.patch_bcs[id];
                const auto p = f.center();
                const FieldContext ctx{p[0], p[1], p[2], temperature[row], time};
                double g = 0;
                switch (bc.type) {
                case BcType::FirstType:
                    g = ki * f.area / f.owner_distance;
                    rhs[row] += g * model.bc_params.dirichlet_T[bc.param_idx].eval(ctx);
                    break;
                case BcType::SecondType:
                    rhs[row] += f.area * model.bc_params.neumann_q[bc.param_idx].eval(ctx);
                    break;
                case BcType::ThirdType: {
                    const double h = model.bc_params.cauchy_h[bc.param_idx].eval(ctx);
                    g = ki * h * f.area / (ki + h * f.owner_distance);
                    rhs[row] += g * model.bc_params.cauchy_T_inf[bc.param_idx].eval(ctx);
                    break;
                }
                default:
                    break;
                }
                if (g != 0)
                    stiffness.emplace_back(row, row, g);
            }
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
