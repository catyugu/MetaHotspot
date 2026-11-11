#include "solver/assembler.hpp"
#include "core/vector.hpp"
#include "thermal_physics.hpp"
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace mhs::solver {
    namespace {
        std::shared_ptr<const core::CSRPattern> build_thermal_pattern(const model::Mesh& mesh)
        {
            const auto n = model::cell_count(mesh);
            core::detail::checked_sparse_size(n);
            core::Vector<std::size_t> counts(n + 1, 1);
            counts[0] = 0;
            model::for_each_connection(mesh, [&](model::Connection c) {
                ++counts[c.a + 1];
                ++counts[c.b + 1];
            });
            std::partial_sum(counts.begin(), counts.end(), counts.begin());
            core::detail::checked_sparse_size(counts.back());
            core::Vector<core::SparseIndex> ptr(counts.begin(), counts.end());
            core::Vector<core::SparseIndex> columns(static_cast<std::size_t>(ptr.back()));
            auto next = ptr;
            for (std::size_t i = 0; i < n; ++i)
                columns[next[i]++] = static_cast<core::SparseIndex>(i);
            model::for_each_connection(mesh, [&](model::Connection c) {
                columns[next[c.a]++] = static_cast<core::SparseIndex>(c.b);
                columns[next[c.b]++] = static_cast<core::SparseIndex>(c.a);
            });
            for (std::size_t i = 0; i < n; ++i)
                std::sort(columns.begin() + ptr[i], columns.begin() + ptr[i + 1]);
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
        , operators_ {core::CSRMatrix<double>(build_thermal_pattern(model.mesh)), build_capacity(model.cells.size()),
              core::Vector<double>(model.cells.size())}
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
        if (temperature.size() != model.cells.size())
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
        std::visit(
            [&](const auto& mesh) {
                for (std::size_t c = 0; c < model.cells.size(); ++c) {
                    const auto row = static_cast<SparseIndex>(c);
                    const auto center = cell_geometry(mesh, static_cast<CellId>(c)).center();
                    const FieldContext ctx {center[0], center[1], center[2], temperature[row], time};
                    const auto& mat = model.material_table[model.cells[c].material];
                    conductivity_[c] = {mat.kx.eval(ctx), mat.ky.eval(ctx), mat.kz.eval(ctx)};
                    const auto volume = cell_geometry(mesh, static_cast<CellId>(c)).volume();
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
            },
            model.mesh);
        for (const auto& term : model.source_terms) {
            const auto row = term.cell;
            const auto& p = term.point;
            rhs[row]
                += term.weight * model.heat_source_table[term.source].eval({p[0], p[1], p[2], temperature[row], time});
        }
        auto add_conduction
            = [&](Connection c, const FaceGeometry& geometry, const CellGeometry& a, const CellGeometry& b) {
                  const auto axis = static_cast<std::size_t>(geometry.axis);
                  const auto [ra, rb] = connection_resistances(
                      model, c, a, b, geometry.axis, conductivity_[c.a][axis], conductivity_[c.b][axis]);
                  const double g = geometry.area() / (ra + rb);
                  values[diagonal_positions_[c.a]] += g;
                  values[slot(static_cast<SparseIndex>(c.a), static_cast<SparseIndex>(c.b))] -= g;
                  values[diagonal_positions_[c.b]] += g;
                  values[slot(static_cast<SparseIndex>(c.b), static_cast<SparseIndex>(c.a))] -= g;
              };
        std::visit(
            [&](const auto& mesh) {
                for_each_connection(mesh, [&](Connection c) {
                    const auto a = cell_geometry(mesh, c.a), b = cell_geometry(mesh, c.b);
                    add_conduction(c, connection_geometry(a, b), a, b);
                });
            },
            model.mesh);
        for (const auto& flow : model.fluid.connections) {
            const auto [i, j] = flow.cells;
            const auto fi = model.fluid.global_to_fluid[i], fj = model.fluid.global_to_fluid[j];
            const double mass = flow.volume_flux * 0.5 * (density_[fi] + density_[fj]);
            if (mass == 0.0)
                continue;
            const double enthalpy = mass * specific_heat_[mass > 0 ? fi : fj];
            const auto ij = slot(static_cast<SparseIndex>(i), static_cast<SparseIndex>(j));
            const auto ji = slot(static_cast<SparseIndex>(j), static_cast<SparseIndex>(i));
            values[mass > 0 ? diagonal_positions_[i] : ij] += enthalpy;
            values[mass > 0 ? ji : diagonal_positions_[j]] -= enthalpy;
        }
        for (std::size_t id = 0; id < model.boundary_facets.size(); ++id) {
            const auto& facet = model.boundary_facets[id];
            if (model.boundary_thermal_condition[id] == invalid_id)
                continue;
            const auto cell = facet.owner();
            const auto axis = static_cast<std::size_t>(facet.axis);
            const auto row = static_cast<SparseIndex>(cell);
            const auto center = facet.center();
            const mhs::expression::FieldContext context {center[0], center[1], center[2], temperature[row], time};
            const double d = cell_geometry(model.mesh, cell).size()[axis] / 2;
            const double g = facet.area() / side_thermal_resistance(d, conductivity_[cell][axis]);
            const auto& condition = model.boundary_conditions[model.boundary_thermal_condition[id]];
            const auto trace = evaluate_boundary(condition, context, facet.area(), g);
            rhs[row] += trace.load;
            values[diagonal_positions_[row]] += trace.conductance;
        }
        for (const auto& term : model.fluid.boundary_terms) {
            const auto cell = model.boundary_facets[term.facet].owner();
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
