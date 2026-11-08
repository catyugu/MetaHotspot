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
        core::CSRMatrix<double> build_capacity(std::size_t n)
        {
            core::Vector<core::SparseIndex> ptr(n + 1), columns(n);
            std::iota(ptr.begin(), ptr.end(), core::SparseIndex {0});
            std::iota(columns.begin(), columns.end(), core::SparseIndex {0});
            return {n, n, std::move(ptr), std::move(columns), core::Vector<double>(n)};
        }
    }

    ThermalAssembler::ThermalAssembler(const model::FomDomain& model, FrozenFlow flow)
        : model_(model)
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
            const auto& mat = model.material_table[model.cell_material[c]];
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
            const auto trace = evaluate_boundary(model.boundary_conditions[term.condition],
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
        if (initial.size() != model.layout.state_count || (model.domains.empty() && model.rom_domains.empty()))
            throw std::invalid_argument("thermal assembly requires a complete state and a physical domain");
        core::Vector<bool> seen(initial.size(), false);
        for (std::size_t d = 0; d < model.domains.size(); ++d) {
            const auto& domain = model.domains[d];
            const auto& field = model.layout.field(
                model.layout.find(static_cast<model::DomainId>(d), model::FieldQuantity::Temperature));
            const auto n = domain.discrete.cells.size();
            auto& map = dofs_.emplace_back(n, core::invalid_id);
            if (field.entities.size() != n || field.dofs.size() != n)
                throw std::invalid_argument("invalid thermal state layout");
            for (std::size_t i = 0; i < n; ++i) {
                const auto c = field.entities[i], dof = field.dofs[i];
                if (c >= n || dof >= seen.size() || seen[dof] || map[c] != core::invalid_id)
                    throw std::invalid_argument("invalid thermal state mapping");
                seen[dof] = true;
                map[c] = dof;
            }
            temperatures_.emplace_back(n);
        }
        // Nonthermal fields are independent state entries, not cell temperatures.
        // Validate the full injective map before constructing workspaces.
        for (const auto& field : model.layout.fields)
            if (field.quantity == model::FieldQuantity::Pressure
                || field.quantity == model::FieldQuantity::Velocity) {
                if (field.domain >= model.domains.size() || field.entities.size() != field.dofs.size())
                    throw std::invalid_argument("invalid fluid state layout");
                if (model.domains[field.domain].flow_mode != model::FlowMode::Stokes)
                    throw std::invalid_argument("fluid state field on a frozen-flow domain");
                for (const auto dof : field.dofs) {
                    if (dof >= seen.size() || seen[dof])
                        throw std::invalid_argument("invalid or repeated fluid DOF");
                    seen[dof] = true;
                }
            }
        for (std::size_t r = 0; r < model.rom_domains.size(); ++r) {
            const auto domain = static_cast<model::DomainId>(model.domains.size() + r);
            const auto& field = model.layout.field(model.layout.find(domain, model::FieldQuantity::ReducedCoordinate));
            const auto n = model.rom_domains[r].order;
            if (field.entities.size() != n || field.dofs.size() != n)
                throw std::invalid_argument("ROM field does not cover modal coordinates");
            core::Vector<bool> local_seen(n, false);
            for (std::size_t i = 0; i < n; ++i) {
                const auto entity = field.entities[i], dof = field.dofs[i];
                if (entity >= n || dof >= seen.size() || seen[dof] || local_seen[entity])
                    throw std::invalid_argument("invalid ROM state mapping");
                seen[dof] = true;
                local_seen[entity] = true;
            }
        }
        if (std::find(seen.begin(), seen.end(), false) != seen.end())
            throw std::invalid_argument("fields do not cover entire system state");
        gather(initial);
        for (std::size_t d = 0; d < model.domains.size(); ++d) {
            const auto& domain = model.domains[d];
            if (domain.flow_mode == model::FlowMode::Stokes && !domain.fluid.fluid_to_global.empty()) {
                stokes_.push_back(std::make_unique<StokesAssembler>(
                    domain, model.layout, static_cast<model::DomainId>(d)));
                domains_.push_back(std::make_unique<ThermalAssembler>(domain, FrozenFlow {}));
            }
            else {
                stokes_.push_back(nullptr);
                domains_.push_back(std::make_unique<ThermalAssembler>(
                    domain, initialize_frozen_flow(domain, temperatures_[d], time)));
            }
        }
        identity_ = model.domains.size() == 1 && model.rom_domains.empty() && model.connections.empty()
            && !stokes_.front();
        for (std::size_t c = 0; identity_ && c < dofs_[0].size(); ++c)
            identity_ = dofs_[0][c] == c;
    }

    void ModelAssembler::gather(std::span<const double> state)
    {
        if (state.size() != model_.layout.state_count)
            throw std::invalid_argument("thermal assembly state size mismatch");
        for (std::size_t d = 0; d < temperatures_.size(); ++d)
            for (std::size_t c = 0; c < temperatures_[d].size(); ++c)
                temperatures_[d][c] = state[dofs_[d][c]];
    }

    const Operators& ModelAssembler::assemble(std::span<const double> state, double time)
    {
        gather(state);
        if (identity_)
            return domains_[0]->assemble(temperatures_[0], time);
        core::Vector<core::Triplet<double>> stiffness, capacity;
        core::Vector<double> load(state.size(), 0.0);
        for (std::size_t d = 0; d < domains_.size(); ++d) {
            if (stokes_[d])
                domains_[d]->set_flow(stokes_[d]->assemble(state, temperatures_[d], time,
                    stiffness, capacity, load));
            const auto& ops = domains_[d]->assemble(temperatures_[d], time);
            const auto& map = dofs_[d];
            auto append = [&](const core::CSRMatrix<double>& matrix, auto& entries) {
                for (std::size_t row = 0; row < matrix.rows(); ++row)
                    for (auto j = matrix.row_ptr()[row]; j < matrix.row_ptr()[row + 1]; ++j)
                        entries.emplace_back(static_cast<core::SparseIndex>(map[row]),
                            static_cast<core::SparseIndex>(map[matrix.col_idx()[j]]), matrix.values()[j]);
            };
            append(ops.K, stiffness);
            append(ops.C, capacity);
            for (std::size_t c = 0; c < ops.f.size(); ++c)
                load[map[c]] += ops.f[c];
        }
        for (std::size_t r = 0; r < model_.rom_domains.size(); ++r) {
            const auto& rom = model_.rom_domains[r];
            const auto domain = static_cast<model::DomainId>(model_.domains.size() + r);
            const auto& field
                = model_.layout.field(model_.layout.find(domain, model::FieldQuantity::ReducedCoordinate));
            for (std::size_t i = 0; i < rom.order; ++i) {
                const auto row = static_cast<core::SparseIndex>(field.dofs[i]);
                double rhs = rom.f[i];
                for (const auto& term : rom.affine_terms)
                    rhs += term.value * term.f[i];
                load[row] += rhs;
                for (std::size_t j = 0; j < rom.order; ++j) {
                    const auto col = static_cast<core::SparseIndex>(field.dofs[j]);
                    double stiffness_value = rom.K[i * rom.order + j];
                    for (const auto& term : rom.affine_terms)
                        stiffness_value += term.value * term.K[i * rom.order + j];
                    if (stiffness_value != 0.0)
                        stiffness.emplace_back(row, col, stiffness_value);
                    if (rom.C[i * rom.order + j] != 0.0)
                        capacity.emplace_back(row, col, rom.C[i * rom.order + j]);
                }
            }
        }

        // An exposed ROM patch is a one-sided thermal connection. Reuse the
        // FOM boundary law's local trace elimination, then distribute its
        // physical conductance and heat load by the Galerkin transpose map.
        for (const auto& term : model_.rom_boundary_terms) {
            const auto& rom = model_.rom_domains.at(term.domain - model_.domains.size());
            const auto& port = rom.ports.at(term.port);
            const auto& facet = port.patches.at(term.patch);
            const auto& field
                = model_.layout.field(model_.layout.find(term.domain, model::FieldQuantity::ReducedCoordinate));
            double temperature = facet.offset;
            for (std::size_t i = 0; i < rom.order; ++i)
                temperature += facet.trace[i] * state[field.dofs[i]];
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
            // The equivalent normal conductance is area/half_resistance.
            const auto& condition = model_.rom_boundary_conditions.at(term.condition);
            const auto normal_resistance = facet.half_resistance;
            const auto trace = evaluate_boundary(condition, center, area, normal_resistance, temperature, 1.0, time);
            // q_out = g*(V*z + offset) - load; ROM contribution V^T*q_out.
            const double rhs = trace.load - trace.conductance * facet.offset;
            for (std::size_t i = 0; i < rom.order; ++i) {
                if (facet.trace[i] == 0.0)
                    continue;
                const auto row = static_cast<core::SparseIndex>(field.dofs[i]);
                load[row] += facet.trace[i] * rhs;
                for (std::size_t j = 0; j < rom.order; ++j)
                    if (facet.trace[j] != 0.0 && trace.conductance != 0.0)
                        stiffness.emplace_back(row, static_cast<core::SparseIndex>(field.dofs[j]),
                            facet.trace[i] * trace.conductance * facet.trace[j]);
            }
        }
        for (const auto& connection : model_.connections) {
            const auto flux = evaluate_connection_flux(model_, connection, state, time);
            for (const auto& patch_flux : flux) {
                const auto g = patch_flux.conductance;
                const auto forcing = patch_flux.correction + patch_flux.bias_flux;
                // Galerkin transpose scattering: q = g*(P_a*x-P_b*x)+forcing.
                // Inject P_a^T*q and -P_b^T*q so both sides use one flux.
                for (const auto& wa : patch_flux.a) {
                    const auto row = static_cast<core::SparseIndex>(wa.dof);
                    load[row] -= wa.weight * forcing;
                    for (const auto& xa : patch_flux.a)
                        stiffness.emplace_back(row, static_cast<core::SparseIndex>(xa.dof), g * wa.weight * xa.weight);
                    for (const auto& xb : patch_flux.b)
                        stiffness.emplace_back(row, static_cast<core::SparseIndex>(xb.dof), -g * wa.weight * xb.weight);
                }
                for (const auto& wb : patch_flux.b) {
                    const auto row = static_cast<core::SparseIndex>(wb.dof);
                    load[row] += wb.weight * forcing;
                    for (const auto& xa : patch_flux.a)
                        stiffness.emplace_back(row, static_cast<core::SparseIndex>(xa.dof), -g * wb.weight * xa.weight);
                    for (const auto& xb : patch_flux.b)
                        stiffness.emplace_back(row, static_cast<core::SparseIndex>(xb.dof), g * wb.weight * xb.weight);
                }
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

    Operators assemble_thermal(const model::FomDomain& model, std::span<const double> temperature, double time)
    {
        if (model.flow_mode == model::FlowMode::Stokes)
            throw std::invalid_argument("standalone FOM thermal assembly requires frozen flow; use a state layout for Stokes");
        ThermalAssembler workspace(model, initialize_frozen_flow(model, temperature, time));
        workspace.assemble(temperature, time);
        return std::move(workspace).take_operators();
    }
} // namespace mhs::solver
