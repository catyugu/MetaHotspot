#include "solver/assembler.hpp"
#include "core/csr_ops.hpp"
#include "core/vector.hpp"
#include "thermal_reconstruction.hpp"
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace mhs::solver {
    namespace {
        std::shared_ptr<const core::CSRPattern> build_thermal_pattern(const model::DiscreteMesh& mesh)
        {
            const auto n = mesh.cells.size();
            core::detail::checked_sparse_size(n);
            core::Vector<core::SparseIndex> ptr(n + 1, 0), columns, row;
            for (std::size_t i = 0; i < n; ++i) {
                row.clear();
                row.push_back(static_cast<core::SparseIndex>(i));
                for (const auto id : mesh.incident_faces(static_cast<model::CellId>(i))) {
                    const auto other = mesh.faces[id].other(static_cast<model::CellId>(i));
                    if (other != core::invalid_id)
                        row.push_back(static_cast<core::SparseIndex>(other));
                }
                std::sort(row.begin(), row.end());
                row.erase(std::unique(row.begin(), row.end()), row.end());
                core::detail::checked_sparse_size(columns.size() + row.size());
                columns.insert(columns.end(), row.begin(), row.end());
                ptr[i + 1] = static_cast<core::SparseIndex>(columns.size());
            }
            return std::make_shared<const core::CSRPattern>(n, n, std::move(ptr), std::move(columns));
        }
        core::CSRMatrix<double> build_capacity(std::size_t n)
        {
            core::Vector<core::SparseIndex> ptr(n + 1), columns(n);
            std::iota(ptr.begin(), ptr.end(), core::SparseIndex {0});
            std::iota(columns.begin(), columns.end(), core::SparseIndex {0});
            return {n, n, std::move(ptr), std::move(columns), core::Vector<double>(n)};
        }
    }

    ThermalAssembler::ThermalAssembler(const model::Model& model)
        : model_(model)
        , operators_ {core::CSRMatrix<double>(build_thermal_pattern(model.discrete)),
              build_capacity(model.discrete.cells.size()), core::Vector<double>(model.discrete.cells.size())}
        , diagonal_positions_(operators_.K.rows())
        , conductivity_(operators_.K.rows())
        , density_(model.fluid.fluid_to_global.size())
        , specific_heat_(density_.size())
    {
        const auto& A = operators_.K;
        const auto columns = A.col_idx();
        for (std::size_t i = 0; i < A.rows(); ++i)
            diagonal_positions_[i] = static_cast<core::SparseIndex>(
                std::lower_bound(columns.begin() + A.row_ptr()[i], columns.begin() + A.row_ptr()[i + 1], i)
                - columns.begin());
    }

    const Operators& ThermalAssembler::assemble(std::span<const double> temperature, double time)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        using namespace mhs::expression;
        const auto& model = model_;
        if (temperature.size() != model.discrete.cells.size())
            throw std::invalid_argument("assemble_thermal: state size differs from DOF count");
        auto& A = operators_.K;
        auto& rhs = operators_.f;
        auto values = A.values();
        std::fill(values.begin(), values.end(), 0.0);
        std::fill(rhs.begin(), rhs.end(), 0.0);
        auto slot = [&](SparseIndex row, SparseIndex col) {
            const auto columns = A.col_idx();
            return std::lower_bound(columns.begin() + A.row_ptr()[row], columns.begin() + A.row_ptr()[row + 1], col)
                - columns.begin();
        };
        for (std::size_t c = 0; c < model.discrete.cells.size(); ++c) {
            const auto row = static_cast<SparseIndex>(c);
            const auto center = model.discrete.cells[c].center();
            const FieldContext ctx {center[0], center[1], center[2], temperature[row], time};
            const auto& mat = model.material_table[model.cell_material[c]];
            conductivity_[c] = {mat.kx.eval(ctx), mat.ky.eval(ctx), mat.kz.eval(ctx)};
            const auto volume = model.discrete.cells[c].volume();
            const double rho = mat.rho.eval(ctx), cp = mat.c.eval(ctx);
            operators_.C.values()[c] = rho * cp * volume;
            if (!density_.empty()) {
                const auto fi = model.fluid.global_to_fluid[c];
                if (fi != mhs::core::invalid_id) {
                    density_[fi] = rho;
                    specific_heat_[fi] = cp;
                }
            }
        }
        for (const auto& term : model.source_terms) {
            const auto row = term.cell;
            const auto& p = term.point;
            rhs[row]
                += term.weight * model.heat_source_table[term.source].eval({p[0], p[1], p[2], temperature[row], time});
        }
        auto fluid_index = [&](CellId c) {
            return c < model.fluid.global_to_fluid.size() ? model.fluid.global_to_fluid[c] : mhs::core::invalid_id;
        };
        for (std::size_t id = 0; id < model.discrete.faces.size(); ++id) {
            const auto& f = model.discrete.faces[id];
            if (f.exterior())
                continue;
            const auto i = f.minus;
            const auto row = static_cast<SparseIndex>(f.minus);
            const auto ki = conductivity_[i][static_cast<std::size_t>(f.axis)];
            const auto j = f.plus;
            const auto col = static_cast<SparseIndex>(f.plus);
            const auto kj = conductivity_[j][static_cast<std::size_t>(f.axis)];
            auto resistance
                = f.distance(model.discrete.cells[f.minus]) / ki + f.distance(model.discrete.cells[f.plus]) / kj;
            const auto fi = fluid_index(f.minus), fj = fluid_index(f.plus);
            if ((fi == mhs::core::invalid_id) != (fj == mhs::core::invalid_id)) {
                const auto fluid = (fi != mhs::core::invalid_id) ? fi : fj;
                const double kfluid = (fi != mhs::core::invalid_id) ? ki : kj;
                const double ksolid = (fi != mhs::core::invalid_id) ? kj : ki;
                const double distance = (fi != mhs::core::invalid_id) ? f.distance(model.discrete.cells[f.plus])
                                                                      : f.distance(model.discrete.cells[f.minus]);
                const double h = model.fluid.interface_heat_transfer_factor[fluid] * kfluid;
                resistance = distance / ksolid + 1.0 / h;
            }
            const double g = f.area() / resistance;
            const auto ij = slot(row, col), ji = slot(col, row);
            values[diagonal_positions_[i]] += g;
            values[ij] -= g;
            values[diagonal_positions_[j]] += g;
            values[ji] -= g;
            if ((fi != mhs::core::invalid_id) && (fj != mhs::core::invalid_id)) {
                const double mass = model.fluid.face_volume_flux[id] * 0.5 * (density_[fi] + density_[fj]);
                if (mass != 0.0) {
                    const double enthalpy = mass * specific_heat_[(mass > 0 ? fi : fj)];
                    values[mass > 0 ? diagonal_positions_[i] : ij] += enthalpy;
                    values[mass > 0 ? ji : diagonal_positions_[j]] -= enthalpy;
                }
            }
        }
        for (const auto& term : model.boundary_faces) {
            const auto& f = model.discrete.faces[term.parent];
            const auto row = static_cast<SparseIndex>(f.cell());
            const auto trace = evaluate_boundary(model.boundary_conditions[term.condition],
                f.center(model.discrete.cells, term.bounds), f.area(term.bounds),
                f.distance(model.discrete.cells[f.cell()]), temperature[row],
                conductivity_[f.cell()][static_cast<std::size_t>(f.axis)], time);
            rhs[row] += trace.load;
            values[diagonal_positions_[row]] += trace.conductance;
        }
        for (const auto& term : model.fluid.boundary_terms) {
            const auto cell = model.discrete.faces[term.face].cell();
            const auto row = static_cast<SparseIndex>(cell);
            const auto cp = specific_heat_[model.fluid.global_to_fluid[cell]];
            std::visit(
                [&](const auto& flow) {
                    using T = std::decay_t<decltype(flow)>;
                    if constexpr (std::is_same_v<T, FluidInflow>)
                        rhs[row] += flow.mass_flow_rate * cp * flow.temperature;
                    else
                        values[diagonal_positions_[row]] += flow.mass_flow_rate * cp;
                },
                term.condition);
        }
        return operators_;
    }

    Operators assemble_thermal(const model::Model& model, std::span<const double> temperature, double time)
    {
        ThermalAssembler workspace(model);
        workspace.assemble(temperature, time);
        return std::move(workspace).take_operators();
    }
} // namespace mhs::solver
