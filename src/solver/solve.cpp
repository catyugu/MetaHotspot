#include "solver/solve.hpp"
#include "core/csr_ops.hpp"
#include "core/vector.hpp"
#include "logging/logger.hpp"
#include "solver/assembler.hpp"
#include "solver/connection_flux.hpp"
#include "solver/frozen_flow.hpp"
#include "solver/nonlinear_solver.hpp"
#include "solver/probe_recorder.hpp"
#include "solver/solver.hpp"
#include "solver/stokes_flow.hpp"
#include "solver/thermal_reconstruction.hpp"
#include <atstep/stepper.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace mhs::solver {

    namespace {

        // Obtain a physically consistent initial flow for the default state.
        // In a pure Neumann thermal channel, starting with u=0 would make the
        // first thermal Picard matrix singular before pressure-driven inflow
        // can supply the required temperature reference.
        void initialize_stokes_state(const model::CompiledModel& compiled, core::Vector<double>& state)
        {
            for (std::size_t domain_id = 0; domain_id < compiled.domains.size(); ++domain_id) {
                const auto& domain = compiled.domains[domain_id];
                if (domain.flow_mode != model::FlowMode::Stokes || domain.fluid.fluid_to_global.empty())
                    continue;
                const auto id = static_cast<model::DomainId>(domain_id);
                const auto& thermal
                    = compiled.layout.field(compiled.layout.find(id, model::FieldQuantity::Temperature));
                const auto temperatures = std::span<const double>(state).subspan(thermal.offset, thermal.count);
                const auto& pressure = compiled.layout.field(compiled.layout.find(id, model::FieldQuantity::Pressure));
                const auto& velocity = compiled.layout.field(compiled.layout.find(id, model::FieldQuantity::Velocity));
                const auto first = static_cast<core::SparseIndex>(pressure.offset);
                const auto count = pressure.count + velocity.count;
                const auto last = first + static_cast<core::SparseIndex>(count);

                StokesAssembler stokes(domain, compiled.layout, id);
                core::Vector<core::Triplet<double>> entries, capacity;
                core::Vector<double> load(state.size(), 0.0);
                stokes.assemble(state, temperatures, 0.0, entries, capacity, load);
                core::Vector<core::Triplet<double>> block;
                for (const auto& e : entries)
                    if (e.row >= first && e.row < last && e.col >= first && e.col < last)
                        block.emplace_back(e.row - first, e.col - first, e.value);
                const auto matrix = core::build_csr(count, count, std::move(block));
                auto solver = linear::create_solver({linear::SolverType::SparseLu, {}});
                solver->compute(matrix.view());
                core::Vector<double> rhs(count), solution(count, 0.0);
                for (std::size_t j = 0; j < count; ++j)
                    rhs[j] = load[first + static_cast<core::SparseIndex>(j)];
                if (!solver->solve(rhs, solution).success)
                    throw std::runtime_error("initial Stokes hydraulic solve failed");
                for (std::size_t j = 0; j < count; ++j)
                    state[first + static_cast<core::SparseIndex>(j)] = solution[j];
            }
        }

        void validate_operator_dimensions(const Operators& ops, std::size_t state_count)
        {
            if (ops.K.rows() != state_count || ops.K.cols() != state_count || ops.C.rows() != state_count
                || ops.C.cols() != state_count || ops.f.size() != state_count) {
                throw std::invalid_argument(
                    "solve_system: SystemAssembler returned Operators with mismatched K/C/f dimensions (expected "
                    + std::to_string(state_count) + ")");
            }
        }

        mhs::linear::SolverSpec build_solver_spec(const SolveOptions& opts)
        {
            mhs::linear::SolverType type;
            switch (opts.linear_solver) {
            case SolveOptions::LinearSolverType::Pardiso:
                type = mhs::linear::SolverType::Pardiso;
                break;
            case SolveOptions::LinearSolverType::Amg:
                type = mhs::linear::SolverType::Amg;
                break;
            case SolveOptions::LinearSolverType::SparseLu:
                type = mhs::linear::SolverType::SparseLu;
                break;
            default:
                throw std::invalid_argument("build_solver_spec: unknown linear_solver");
            }
            return {type, {opts.linear_tolerance, opts.linear_max_iterations}};
        }

        NonLinearConfig build_nonlinear_config(const SolveOptions& opts)
        {
            return {opts.underrelaxation, opts.nonlinear_max_iterations, opts.nonlinear_relative_tolerance,
                opts.nonlinear_absolute_tolerance};
        }

        mhs::solver::Solution solve_steady(const SystemAssembler& assemble, mhs::core::Vector<double> state,
            mhs::linear::SolverPtr& solver, const NonLinearConfig& nl_config, const StateObserver& observe)
        {
            const auto state_count = state.size();
            mhs::solver::Solution result;
            LinearSystemProvider build_ls = [&](std::span<const double> current_state) -> LinearSystemView {
                const auto& ops = assemble(current_state, 0.0);
                validate_operator_dimensions(ops, state_count);
                return {ops.K.view(), ops.f};
            };
            auto nonlinear_result = nonlinear_solve(build_ls, state, solver, nl_config);
            result.state = std::move(state);
            result.time = 0.0;
            result.converged = nonlinear_result;
            result.snapshot_times.push_back(0.0);
            result.snapshot_states = result.state;
            if (observe)
                observe(0.0, result.state);
            return result;
        }

        mhs::solver::Solution solve_transient(const mhs::model::Study& study, const SystemAssembler& assemble,
            mhs::core::Vector<double> state, mhs::linear::SolverPtr& solver, const SolveOptions& opts,
            const NonLinearConfig& nl_config, const StateObserver& observe)
        {
            using State = core::Vector<double>;
            const auto state_count = state.size();
            if (!std::isfinite(study.duration) || study.duration <= 0.0 || !std::isfinite(study.output_interval)
                || study.output_interval < 0.0)
                throw std::invalid_argument("transient duration must be positive and output interval nonnegative");

            atstep::Settings settings;
            auto& steps = std::get<atstep::AdaptiveSteps>(settings.steps);
            steps.max_order = 2;
            steps.interpolate_end_time = false;
            if (opts.max_time_step)
                steps.max_step = atstep::ConstantMaxStep {*opts.max_time_step};
            settings.error.relative = opts.time_tolerance;
            settings.error.scaled = false;
            settings.storage = study.output_interval > 0.0 ? atstep::Storage::interpolate : atstep::Storage::steps;

            const auto& initial_ops = assemble(state, 0.0);
            validate_operator_dimensions(initial_ops, state_count);
            // Establish the union once. K, C and A may share structure, while
            // retaining independent numerical storage and arbitrary sparse C.
            auto A = core::add_scaled(initial_ops.K, 0.0, initial_ops.C);
            State rhs(state_count);
            State derivative(state_count);
            bool positive_diagonal = true;
            const auto C = initial_ops.C.view();
            for (std::size_t i = 0; i < state_count; ++i) {
                double capacity = 0.0;
                for (auto j = C.row_ptr[i]; j < C.row_ptr[i + 1]; ++j) {
                    if (static_cast<std::size_t>(C.col_idx[j]) == i)
                        capacity = C.values[j];
                    else if (C.values[j] != 0.0)
                        positive_diagonal = false;
                }
                if (!std::isfinite(capacity) || capacity <= 0.0)
                    positive_diagonal = false;
                derivative[i] = capacity;
            }

            atstep::TimeStepper<State> stepper(settings, [&](const atstep::StepRequest<State>& request, State& trial) {
                LinearSystemProvider provider = [&](std::span<const double> iterate) -> LinearSystemView {
                    const auto& ops = assemble(iterate, request.time);
                    validate_operator_dimensions(ops, state_count);
                    core::add_scaled_into(ops.K, request.derivative_weight(), ops.C, A);
                    std::copy(ops.f.begin(), ops.f.end(), rhs.begin());
                    core::spmv(ops.C.view(), request.derivative_history(), rhs, -1.0, 1.0);
                    return {A.view(), rhs};
                };
                return nonlinear_solve(provider, trial, solver, nl_config);
            });

            // Preserve the supplied T(0) for positive diagonal capacity.
            // Algebraic or coupled capacity retains consistent initialization.
            if (positive_diagonal) {
                std::copy(initial_ops.f.begin(), initial_ops.f.end(), rhs.begin());
                core::spmv(initial_ops.K.view(), state, rhs, -1.0, 1.0);
                for (std::size_t i = 0; i < state_count; ++i)
                    derivative[i] = rhs[i] / derivative[i];
                stepper.start(0.0, state, derivative, study.duration);
            }
            else
                stepper.start(0.0, state, study.duration);

            State output_times {0.0};
            if (study.output_interval > 0.0) {
                for (std::size_t i = 1;; ++i) {
                    const double time = static_cast<double>(i) * study.output_interval;
                    if (time >= study.duration
                        || study.duration - time <= 8.0 * std::numeric_limits<double>::epsilon() * study.duration)
                        break;
                    if (!(time > output_times.back()))
                        throw std::length_error("output interval cannot advance the output timeline");
                    output_times.push_back(time);
                }
            }
            output_times.push_back(study.duration);
            mhs::solver::Solution result;
            atstep::RunHooks<State> hooks;
            hooks.on_store = [&](double time, const State& value, bool) {
                const auto& stored = time == study.duration ? stepper.state() : value;
                result.snapshot_times.push_back(time);
                result.snapshot_states.insert(result.snapshot_states.end(), stored.begin(), stored.end());
                if (observe)
                    observe(time, stored);
            };
            hooks.on_step = [](const atstep::StepRecord& record, const State&) {
                MHS_LOG_DEBUG("Time: {} solved (dt={}, order={})", record.time, record.step, record.order);
            };
            stepper.run(output_times, std::move(hooks));
            result.state = stepper.state();
            result.time = stepper.time();
            result.converged = stepper.finished();
            return result;
        }

    } // namespace

    Solution solve_system(const mhs::model::Study& study, const SystemAssembler& assemble, core::Vector<double> state,
        const SolveOptions& opts, const StateObserver& observe)
    {
        if (!assemble)
            throw std::invalid_argument("solve_system: assembler is empty");
        if (state.empty())
            throw std::invalid_argument("solve_system: initial_state is empty");
        auto solver = linear::create_solver(build_solver_spec(opts));
        const auto nl_config = build_nonlinear_config(opts);
        if (study.type == mhs::model::StudyType::Steady)
            return solve_steady(assemble, std::move(state), solver, nl_config, observe);
        return solve_transient(study, assemble, std::move(state), solver, opts, nl_config, observe);
    }

    mhs::solver::Solution solve(const mhs::model::CompiledModel& compiled, const mhs::model::Study& study,
        std::span<const double> initial_state, const SolveOptions& options)
    {
        const auto state_count = compiled.layout.state_count;
        if (state_count == 0)
            throw std::invalid_argument("solve: model has no states");
        if (!std::isfinite(study.initial_temperature))
            throw std::invalid_argument("solve: initial temperature must be finite");

        core::Vector<double> state;
        if (initial_state.empty()) {
            state.assign(state_count, 0.0);
            for (const auto& field : compiled.layout.fields)
                if (field.quantity == model::FieldQuantity::Temperature)
                    std::fill_n(state.begin() + field.offset, field.count, study.initial_temperature);
            for (std::size_t r = 0; r < compiled.rom_domains.size(); ++r) {
                const auto domain = static_cast<model::DomainId>(compiled.domains.size() + r);
                const auto& field
                    = compiled.layout.field(compiled.layout.find(domain, model::FieldQuantity::ReducedCoordinate));
                const auto& initial = compiled.rom_domains[r].initial_state;
                std::copy(initial.begin(), initial.end(), state.begin() + field.offset);
            }
        }
        else {
            if (initial_state.size() != state_count)
                throw std::invalid_argument("solve: initial_state size must equal state_count");
            state.assign(initial_state.begin(), initial_state.end());
        }
        if (!std::all_of(state.begin(), state.end(), [](double v) { return std::isfinite(v); }))
            throw std::invalid_argument("initial state must be finite");

        if (initial_state.empty())
            initialize_stokes_state(compiled, state);
        ModelAssembler assembler(compiled, state);
        SystemAssembler assemble = [&](std::span<const double> current, double time) -> const Operators& {
            return assembler.assemble(current, time);
        };

        // Resolve physical probes against either FOM geometry or explicitly
        // declared ROM observations. A point cannot belong to both.
        core::Vector<model::DomainId> probe_domains;
        core::Vector<std::unique_ptr<ProbeRecorder>> recorders;
        core::Vector<const model::RomObservation*> rom_observations;
        core::Vector<ProbeTrace> rom_traces;
        for (const auto& probe : compiled.point_probes) {
            model::DomainId owner = core::invalid_id;
            const model::RomObservation* observation = nullptr;
            for (std::size_t d = 0; d < compiled.domains.size(); ++d) {
                const auto cell = locate_cell(compiled.domains[d].discrete, {probe.x, probe.y, probe.z});
                if (cell == core::invalid_id)
                    continue;
                if (owner != core::invalid_id)
                    throw std::invalid_argument("probe belongs to multiple domains");
                owner = static_cast<model::DomainId>(d);
            }
            for (std::size_t r = 0; r < compiled.rom_domains.size(); ++r) {
                for (const auto& value : compiled.rom_domains[r].observations) {
                    if (probe.name != value.point.name || probe.x != value.point.x || probe.y != value.point.y
                        || probe.z != value.point.z)
                        continue;
                    if (owner != core::invalid_id)
                        throw std::invalid_argument("probe belongs to multiple domains");
                    owner = static_cast<model::DomainId>(compiled.domains.size() + r);
                    observation = &value;
                }
            }
            if (owner == core::invalid_id)
                throw std::invalid_argument("probe is outside all domains");
            probe_domains.push_back(owner);
            rom_observations.push_back(observation);
            rom_traces.push_back({probe.name, {}});
            if (observation)
                recorders.push_back(nullptr);
            else
                recorders.push_back(std::make_unique<ProbeRecorder>(compiled.domains[owner], std::span(&probe, 1)));
        }

        StateObserver observe = [&](double t, std::span<const double> accepted_state) {
            core::Vector<core::Vector<SurfaceTrace>> traces(compiled.domains.size());
            for (const auto& connection : compiled.connections) {
                if (connection.a.domain >= compiled.domains.size() && connection.b.domain >= compiled.domains.size())
                    continue;
                const auto flux = evaluate_connection_flux(compiled, connection, accepted_state, t);
                for (std::size_t p = 0; p < connection.patches.size(); ++p) {
                    const auto& patch = connection.patches[p];
                    if (connection.a.domain < compiled.domains.size())
                        traces[connection.a.domain].push_back(
                            {{patch.a, connection.axis, connection.sign_a, core::invalid_id, patch.bounds},
                                flux[p].interface_temperature});
                    if (connection.b.domain < compiled.domains.size())
                        traces[connection.b.domain].push_back(
                            {{patch.b, connection.axis, static_cast<std::int8_t>(-connection.sign_a), core::invalid_id,
                                 patch.bounds},
                                flux[p].interface_temperature});
                }
            }
            for (std::size_t p = 0; p < recorders.size(); ++p) {
                if (recorders[p]) {
                    const auto& field = compiled.layout.field(
                        compiled.layout.find(probe_domains[p], model::FieldQuantity::Temperature));
                    recorders[p]->record(
                        t, accepted_state.subspan(field.offset, field.count), traces[probe_domains[p]]);
                    continue;
                }
                const auto* observation = rom_observations[p];
                const auto& field = compiled.layout.field(
                    compiled.layout.find(probe_domains[p], model::FieldQuantity::ReducedCoordinate));
                double temperature = observation->offset;
                for (std::size_t j = 0; j < observation->weights.size(); ++j)
                    temperature += observation->weights[j] * accepted_state[field.dof(j)];
                rom_traces[p].values.push_back(temperature);
            }
        };

        // The assembled mixed velocity-pressure block is indefinite and has
        // algebraic pressure rows. The thermal AMG default is not a Stokes
        // preconditioner; use SparseLU unless a direct solver was requested.
        auto effective_options = options;
        if (effective_options.linear_solver == SolveOptions::LinearSolverType::Amg)
            for (const auto& domain : compiled.domains)
                if (domain.flow_mode == model::FlowMode::Stokes && !domain.fluid.fluid_to_global.empty()) {
                    effective_options.linear_solver = SolveOptions::LinearSolverType::SparseLu;
                    break;
                }
        auto sol = solve_system(study, assemble, std::move(state), effective_options, observe);
        for (std::size_t p = 0; p < recorders.size(); ++p) {
            if (recorders[p])
                sol.probe_traces.push_back(std::move(*recorders[p]).take_traces().front());
            else
                sol.probe_traces.push_back(std::move(rom_traces[p]));
        }
        return sol;
    }

} // namespace mhs::solver
