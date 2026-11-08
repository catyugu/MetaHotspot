#include "solver/assembler.hpp"
#include "core/csr_ops.hpp"
#include "core/vector.hpp"
#include "solver/connection_flux.hpp"
#include "solver/frozen_flow.hpp"
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
                    const auto other = (mesh.faces[id].minus == i ? mesh.faces[id].plus : mesh.faces[id].minus);
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
        // Galerkin scatter for a physical trace q = g * (B*x) - rhs,
        // where B is a signed map from global states to interface temperature.
        void scatter_trace(core::Vector<core::Triplet<double>>& stiffness, core::Vector<double>& load,
            std::span<const WeightedDof> weights, double conductance, double rhs)
        {
            for (const auto& row : weights) {
                load[row.dof] += row.weight * rhs;
                for (const auto& col : weights)
                    stiffness.emplace_back(static_cast<core::SparseIndex>(row.dof),
                        static_cast<core::SparseIndex>(col.dof), conductance * row.weight * col.weight);
            }
        }
        // Reduced local operators are constant for the compiled component.
        Operators build_reduced_operators(const model::RomDomain& rom)
        {
            core::Vector<core::Triplet<double>> stiffness, capacity;
            core::Vector<double> load(rom.order);
            for (std::size_t i = 0; i < rom.order; ++i) {
                const auto row = static_cast<core::SparseIndex>(i);
                double rhs = rom.f[i];
                for (const auto& term : rom.affine_terms)
                    rhs += term.value * term.f[i];
                load[i] = rhs;
                for (std::size_t j = 0; j < rom.order; ++j) {
                    const auto col = static_cast<core::SparseIndex>(j);
                    const auto index = i * rom.order + j;
                    double k = rom.K[index];
                    for (const auto& term : rom.affine_terms)
                        k += term.value * term.K[index];
                    if (k != 0.0)
                        stiffness.emplace_back(row, col, k);
                    if (rom.C[index] != 0.0)
                        capacity.emplace_back(row, col, rom.C[index]);
                }
            }
            return {core::build_csr(rom.order, rom.order, std::move(stiffness)),
                core::build_csr(rom.order, rom.order, std::move(capacity)), std::move(load)};
        }
        core::CSRMatrix<double> build_capacity(std::size_t n)
        {
            core::Vector<core::SparseIndex> ptr(n + 1), columns(n);
            std::iota(ptr.begin(), ptr.end(), core::SparseIndex {0});
            std::iota(columns.begin(), columns.end(), core::SparseIndex {0});
            return {n, n, std::move(ptr), std::move(columns), core::Vector<double>(n)};
        }
    }

    ThermalAssembler::ThermalAssembler(const model::FomDomain& model, const model::CompiledModel& compiled, FrozenFlow flow)
        : model_(model)
        , compiled_(compiled)
        , flow_(std::move(flow))
        , operators_ {core::CSRMatrix<double>(build_thermal_pattern(model.discrete)),
              build_capacity(model.discrete.cells.size()), core::Vector<double>(model.discrete.cells.size())}
        , diagonal_positions_(operators_.K.rows())
        , conductivity_(operators_.K.rows())
        , specific_heat_(model.fluid.fluid_to_global.size())
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
            const auto& mat = compiled_.library->material_table[model.cell_material[c]];
            conductivity_[c] = {mat.kx.eval(ctx), mat.ky.eval(ctx), mat.kz.eval(ctx)};
            const auto volume = model.discrete.cells[c].volume();
            const double rho = mat.rho.eval(ctx), cp = mat.c.eval(ctx);
            operators_.C.values()[c] = rho * cp * volume;
            if (!specific_heat_.empty()) {
                const auto fi = model.fluid.global_to_fluid[c];
                if (fi != mhs::core::invalid_id) {
                    specific_heat_[fi] = cp;
                }
            }
        }
        for (const auto& term : model.source_terms) {
            const auto row = term.cell;
            const auto& p = term.point;
            rhs[row]
                += term.weight * compiled_.library->heat_source_table[term.source].eval({p[0], p[1], p[2], temperature[row], time});
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
            const auto axis = static_cast<std::size_t>(f.axis);
            const double d_minus = model.discrete.cells[f.minus].size()[axis] / 2;
            const double d_plus = model.discrete.cells[f.plus].size()[axis] / 2;
            auto resistance = d_minus / ki + d_plus / kj;
            const auto fi = fluid_index(f.minus), fj = fluid_index(f.plus);
            if (model.flow_mode == FlowMode::Frozen
                && ((fi == mhs::core::invalid_id) != (fj == mhs::core::invalid_id))) {
                const auto fluid = (fi != mhs::core::invalid_id) ? fi : fj;
                const double kfluid = (fi != mhs::core::invalid_id) ? ki : kj;
                const double ksolid = (fi != mhs::core::invalid_id) ? kj : ki;
                const double distance = (fi != mhs::core::invalid_id) ? d_plus : d_minus;
                const double h = flow_.interface_heat_transfer_factor[fluid] * kfluid;
                resistance = distance / ksolid + 1.0 / h;
            }
            const double g = f.area() / resistance;
            const auto ij = slot(row, col), ji = slot(col, row);
            values[diagonal_positions_[i]] += g;
            values[ij] -= g;
            values[diagonal_positions_[j]] += g;
            values[ji] -= g;
            if ((fi != mhs::core::invalid_id) && (fj != mhs::core::invalid_id)) {
                const double mass = flow_.face_mass_flow[id];
                if (mass != 0.0) {
                    const double enthalpy = mass * specific_heat_[(mass > 0 ? fi : fj)];
                    values[mass > 0 ? diagonal_positions_[i] : ij] += enthalpy;
                    values[mass > 0 ? ji : diagonal_positions_[j]] -= enthalpy;
                }
            }
        }
        for (const auto& term : model.boundary_faces) {
            const auto axis = static_cast<std::size_t>(term.axis);
            const double area = (term.bounds[1] - term.bounds[0]) * (term.bounds[3] - term.bounds[2]);
            const auto row = static_cast<SparseIndex>(term.cell);
            const auto trace = evaluate_boundary(compiled_.library->boundary_conditions[term.condition],
                boundary_center(model.discrete.cells[term.cell], term), area,
                (model.discrete.cells[term.cell].size()[axis] / 2), temperature[row], conductivity_[term.cell][axis],
                time);
            rhs[row] += trace.load;
            values[diagonal_positions_[row]] += trace.conductance;
        }
        for (const auto& term : flow_.boundary_terms) {
            const auto cell = term.cell;
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

    ModelAssembler::ModelAssembler(const model::CompiledModel& model, std::span<const double> initial, double time)
        : model_(model)
    {
        if (initial.size() != model.layout.state_count || model.components.empty())
            throw std::invalid_argument("thermal assembly requires a complete state and a physical domain");
        components_.resize(model.components.size());
        for (const auto& field : model.layout.fields) {
            auto& workspace = components_[field.domain];
            if (field.quantity == model::FieldQuantity::ReducedCoordinate) {
                workspace.reduced = build_reduced_operators(model.rom(field.domain));
                continue;
            }
            if (field.quantity != model::FieldQuantity::Temperature)
                continue;
            const auto& domain = model.fom(field.domain);
            const auto temperature = initial.subspan(field.offset, field.count);
            if (domain.flow_mode == model::FlowMode::Stokes && !domain.fluid.fluid_to_global.empty()) {
                workspace.stokes = std::make_unique<StokesAssembler>(model, field.domain);
                workspace.thermal = std::make_unique<ThermalAssembler>(domain, model, FrozenFlow {});
            }
            else
                workspace.thermal
                    = std::make_unique<ThermalAssembler>(
                        domain, model, initialize_frozen_flow(domain, model.library->material_table, model.library->fluid_conditions, temperature, time));
        }
    }

    const Operators& ModelAssembler::assemble(std::span<const double> state, double time)
    {
        if (state.size() != model_.layout.state_count)
            throw std::invalid_argument("thermal assembly state size mismatch");

        // When the global layout is exactly one uncoupled local block, reuse
        // its CSR directly. This applies to Cartesian and reduced components
        // alike, avoiding a triplet rebuild on every nonlinear iteration.
        if (model_.components.size() == 1 && model_.connections.empty() && model_.rom_boundary_terms.empty()
            && model_.layout.fields.size() == 1) {
            auto& block = components_.front();
            if (block.reduced)
                return *block.reduced;
            if (block.thermal && !block.stokes)
                return block.thermal->assemble(state, time);
        }

        core::Vector<core::Triplet<double>> stiffness, capacity;
        core::Vector<double> load(state.size(), 0.0);
        for (const auto& field : model_.layout.fields) {
            auto& block = components_[field.domain];
            const Operators* ops = nullptr;
            if (field.quantity == model::FieldQuantity::Temperature) {
                const auto temperature = state.subspan(field.offset, field.count);
                if (block.stokes)
                    block.thermal->set_flow(
                        block.stokes->assemble(state, temperature, time, stiffness, capacity, load));
                ops = &block.thermal->assemble(temperature, time);
            }
            else if (field.quantity == model::FieldQuantity::ReducedCoordinate)
                ops = &*block.reduced;
            else
                continue;
            auto append = [&](const core::CSRMatrix<double>& matrix, auto& entries) {
                for (std::size_t row = 0; row < matrix.rows(); ++row)
                    for (auto j = matrix.row_ptr()[row]; j < matrix.row_ptr()[row + 1]; ++j)
                        entries.emplace_back(static_cast<core::SparseIndex>(field.dof(row)),
                            static_cast<core::SparseIndex>(field.dof(matrix.col_idx()[j])), matrix.values()[j]);
            };
            append(ops->K, stiffness);
            append(ops->C, capacity);
            for (std::size_t c = 0; c < ops->f.size(); ++c)
                load[field.dof(c)] += ops->f[c];
        }

        // An exposed ROM patch is a one-sided thermal connection. Reuse the
        // FOM boundary law's local trace elimination, then distribute its
        // physical conductance and heat load by the Galerkin transpose map.
        for (const auto& term : model_.rom_boundary_terms) {
            const auto& rom = model_.rom(term.domain);
            const auto& port = model_.components.at(term.domain).ports.at(term.port);
            const auto& facet = port.patches.at(term.patch);
            const auto& field
                = model_.layout.field(model_.layout.find(term.domain, model::FieldQuantity::ReducedCoordinate));
            double temperature = facet.offset;
            for (std::size_t i = 0; i < rom.order; ++i)
                temperature += facet.trace[i] * state[field.dof(i)];
            const double area = (term.bounds[1] - term.bounds[0]) * (term.bounds[3] - term.bounds[2]);
            model::Vec3 center {};
            const auto axis = static_cast<std::size_t>(port.region.axis);
            center[axis] = port.region.coordinate;
            std::size_t tangent = 0;
            for (std::size_t d = 0; d < 3; ++d)
                if (d != axis) {
                    center[d] = (term.bounds[2 * tangent] + term.bounds[2 * tangent + 1]) / 2;
                    ++tangent;
                }
            const auto& condition = model_.library->boundary_conditions.at(term.condition);
            const auto trace
                = evaluate_boundary(condition, center, area, facet.half_resistance, temperature, 1.0, time);
            core::Vector<WeightedDof> weights;
            for (std::size_t i = 0; i < rom.order; ++i)
                if (facet.trace[i] != 0.0)
                    weights.push_back({field.dof(i), facet.trace[i]});
            scatter_trace(stiffness, load, weights, trace.conductance, trace.load - trace.conductance * facet.offset);
        }

        for (const auto& connection : model_.connections) {
            const auto flux = evaluate_connection_flux(model_, connection, state, time);
            for (const auto& patch_flux : flux) {
                scatter_trace(stiffness, load, patch_flux.weights, patch_flux.conductance,
                    -patch_flux.correction - patch_flux.bias_flux);
            }
        }
        operators_ = {core::build_csr(state.size(), state.size(), std::move(stiffness)),
            core::build_csr(state.size(), state.size(), std::move(capacity)), std::move(load)};
        return operators_;
    }

    Operators assemble_thermal(const model::CompiledModel& compiled, std::span<const double> state, double time)
    {
        ModelAssembler workspace(compiled, state, time);
        return workspace.assemble(state, time);
    }

} // namespace mhs::solver
