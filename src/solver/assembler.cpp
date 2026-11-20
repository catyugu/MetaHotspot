#include "solver/assembler.hpp"
#include "core/vector.hpp"
#include "thermal_physics.hpp"
#include <numeric>
#include <stdexcept>

namespace mhs::solver {
    namespace {
        std::shared_ptr<const core::CSRPattern> build_thermal_pattern(const model::ConformalMesh& mesh)
        {
            const auto n = mesh.cell_count();
            core::detail::checked_sparse_size(n);
            core::Vector<core::SparseIndex> ptr(n + 1, 0), columns;
            for (std::size_t i = 0; i < n; ++i) {
                std::array<core::SparseIndex, 7> row {static_cast<core::SparseIndex>(i)};
                auto end = row.begin() + 1;
                for (std::size_t axis = 0; axis < 3; ++axis)
                    for (int sign : {-1, 1})
                        if (const auto j = mesh.neighbor(i, axis, sign); j != core::invalid_id)
                            *end++ = static_cast<core::SparseIndex>(j);
                std::sort(row.begin(), end);
                core::detail::checked_sparse_size(columns.size() + (end - row.begin()));
                columns.insert(columns.end(), row.begin(), end);
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
        , operators_ {core::CSRMatrix<double>(build_thermal_pattern(model.mesh)),
              build_capacity(model.mesh.cell_count()), core::Vector<double>(model.mesh.cell_count())}
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
        if (temperature.size() != model.mesh.cell_count())
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
        const auto& mesh = model.mesh;
        for (std::size_t c = 0; c < mesh.cell_count(); ++c) {
            const auto row = static_cast<SparseIndex>(c);
            const auto geometry = mesh.cell(c);
            const auto center = geometry.center();
            const FieldContext ctx {center[0], center[1], center[2], temperature[row], time};
            const auto& mat = model.material_table[model.cell_material[c]];
            conductivity_[c] = {mat.kx.eval(ctx), mat.ky.eval(ctx), mat.kz.eval(ctx)};
            const auto volume = geometry.volume();
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
        for (const auto& term : model.source_terms)
            rhs[term.cell] += source_power(model, term, temperature[term.cell], time);
        auto fluid_index = [&](CellId c) {
            return c < model.fluid.global_to_fluid.size() ? model.fluid.global_to_fluid[c] : mhs::core::invalid_id;
        };
        for (CellId i = 0; i < mesh.cell_count(); ++i) {
            const auto row = static_cast<SparseIndex>(i);
            const auto cell = mesh.cell(i);
            const auto fi = fluid_index(i);
            for (std::size_t axis = 0; axis < 3; ++axis) {
                const auto j = mesh.neighbor(i, axis, 1);
                if (j == invalid_id)
                    continue;
                const auto col = static_cast<SparseIndex>(j);
                const auto fj = fluid_index(j);
                const auto [r_i, r_j]
                    = interior_resistances(model, i, j, axis, conductivity_[i][axis], conductivity_[j][axis]);
                const double g = cell.face_area(axis) / (r_i + r_j);
                const auto ij = slot(row, col), ji = slot(col, row);
                values[diagonal_positions_[i]] += g;
                values[ij] -= g;
                values[diagonal_positions_[j]] += g;
                values[ji] -= g;
                if ((fi != mhs::core::invalid_id) && (fj != mhs::core::invalid_id)) {
                    const double mass
                        = model.fluid.positive_volume_flux[fi][axis] * 0.5 * (density_[fi] + density_[fj]);
                    if (mass != 0.0) {
                        const double enthalpy = mass * specific_heat_[(mass > 0 ? fi : fj)];
                        values[mass > 0 ? diagonal_positions_[i] : ij] += enthalpy;
                        values[mass > 0 ? ji : diagonal_positions_[j]] -= enthalpy;
                    }
                }
            }
        }
        for (const auto& term : model.fluid.boundary_terms) {
            const auto cell = term.cell;
            const auto cp = specific_heat_[model.fluid.global_to_fluid[cell]];
            const auto mass = term.mass_flow_rate;
            if (mass < 0)
                rhs[cell] -= mass * cp * term.inlet_temperature;
            else
                values[diagonal_positions_[cell]] += mass * cp;
        }
        return operators_;
    }

    const Operators& ThermalAssembler::assemble(
        std::span<const double> temperature, double time, const model::Project* project, std::size_t component)
    {
        assemble_internal(temperature, time);
        const auto* overrides = project ? &project->components()[component].boundaries : nullptr;
        for (model::FacetId id = 0; id < model_.mesh.exterior_facets.size(); ++id) {
            const auto& facet = model_.mesh.exterior_facets[id];
            const auto surface = facet.boundary_group;
            const auto& fallback = model_.boundary_groups[surface].condition;
            const auto& law = overrides ? *boundary_override(*overrides, surface, &fallback) : fallback;
            if (is_adiabatic(law))
                continue;
            const auto cell = facet.owner;
            const auto r = resistance(id);
            const auto [conductance, load] = boundary_terms(facet.geometry,
                project ? project->face_partition(component, id) : nullptr,
                [&](const model::Facet& patch) { return evaluate_boundary(law, patch, temperature[cell], time, r); });
            operators_.K.values()[diagonal_positions_[cell]] += conductance;
            operators_.f[cell] += load;
        }
        return operators_;
    }

    double ThermalAssembler::resistance(model::FacetId id) const
    {
        const auto& facet = model_.mesh.exterior_facets[id];
        const auto axis = static_cast<std::size_t>(facet.geometry.axis);
        return boundary_resistance(model_, id, conductivity_[facet.owner][axis]);
    }

    Operators assemble_thermal(const model::Fom& model, std::span<const double> temperature, double time)
    {
        ThermalAssembler assembler(model);
        assembler.assemble(temperature, time);
        return std::move(assembler).take_operators();
    }

} // namespace mhs::solver
