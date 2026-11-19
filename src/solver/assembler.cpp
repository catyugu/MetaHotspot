#include "solver/assembler.hpp"
#include "core/vector.hpp"
#include "thermal_physics.hpp"
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
                for (const auto id : mesh.incident_facets(static_cast<model::CellId>(i))) {
                    const auto other
                        = (mesh.cell_facets[id].minus == i ? mesh.cell_facets[id].plus : mesh.cell_facets[id].minus);
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

    ThermalAssembler::ThermalAssembler(const model::Fom& model)
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

    const Operators& ThermalAssembler::assemble_internal(std::span<const double> temperature, double time)
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
        for (std::size_t id = 0; id < model.discrete.facets.size(); ++id) {
            const auto& f = model.discrete.facets[id];
            const auto& cells = model.discrete.cell_facets[id];
            if (cells.exterior())
                continue;
            const auto i = cells.minus;
            const auto row = static_cast<SparseIndex>(cells.minus);
            const auto axis = static_cast<std::size_t>(f.axis);
            const auto ki = conductivity_[i][axis];
            const auto j = cells.plus;
            const auto col = static_cast<SparseIndex>(cells.plus);
            const auto kj = conductivity_[j][axis];
            const auto fi = fluid_index(i), fj = fluid_index(j);
            const auto [r_minus, r_plus] = interior_facet_resistances(model, static_cast<FacetId>(id), ki, kj);
            const double g = f.area() / (r_minus + r_plus);
            const auto ij = slot(row, col), ji = slot(col, row);
            values[diagonal_positions_[i]] += g;
            values[ij] -= g;
            values[diagonal_positions_[j]] += g;
            values[ji] -= g;
            if ((fi != mhs::core::invalid_id) && (fj != mhs::core::invalid_id)) {
                const double mass = model.fluid.facet_volume_flux[id] * 0.5 * (density_[fi] + density_[fj]);
                if (mass != 0.0) {
                    const double enthalpy = mass * specific_heat_[(mass > 0 ? fi : fj)];
                    values[mass > 0 ? diagonal_positions_[i] : ij] += enthalpy;
                    values[mass > 0 ? ji : diagonal_positions_[j]] -= enthalpy;
                }
            }
        }
        for (const auto& term : model.fluid.boundary_terms) {
            const auto cell = model.discrete.cell_facets[term.facet].owner();
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

    const Operators& ThermalAssembler::assemble(std::span<const double> temperature, double time,
        const model::ExposedFacets& exposed, const BoundaryOverrides& overrides)
    {
        assemble_internal(temperature, time);
        for (std::size_t i = 0; i < model_.facet_thermal_condition.size(); ++i) {
            const auto condition = model_.facet_thermal_condition[i];
            if (condition == core::invalid_id)
                continue;
            const auto& law = boundary_override(overrides, condition, model_.boundary_conditions[condition]);
            if (is_adiabatic(law))
                continue;
            const auto id = static_cast<model::FacetId>(i);
            const auto cell = model_.discrete.cell_facets[id].owner();
            const auto r = resistance(id);
            for (const auto& patch : exposed_patches(exposed, id, model_.discrete.facets[id])) {
                const auto trace = evaluate_boundary(law, patch, temperature[cell], time, r);
                operators_.K.values()[diagonal_positions_[cell]] += trace.conductance;
                operators_.f[cell] += trace.load;
            }
        }
        return operators_;
    }

    double ThermalAssembler::resistance(model::FacetId id) const
    {
        const auto& facet = model_.discrete.facets[id];
        const auto cell = model_.discrete.cell_facets[id].owner();
        const auto axis = static_cast<std::size_t>(facet.axis);
        return side_thermal_resistance(model_.discrete.cells[cell].size()[axis] / 2, conductivity_[cell][axis]);
    }

    Operators assemble_thermal(const model::Fom& model, std::span<const double> temperature, double time)
    {
        ThermalAssembler assembler(model);
        assembler.assemble(temperature, time);
        return std::move(assembler).take_operators();
    }

} // namespace mhs::solver
