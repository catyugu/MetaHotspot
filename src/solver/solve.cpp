#include "solver/solve.hpp"
#include "core/csr_ops.hpp"
#include "core/vector.hpp"
#include "logging/logger.hpp"
#include "solver/assembler.hpp"
#include "solver/nonlinear_solver.hpp"
#include "solver/solver.hpp"
#include <atstep/settings.hpp>
#include <atstep/stepper.hpp>

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace mhs::solver {

    namespace {

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
            settings.initialization = atstep::Initialization::off;

            const auto& initial_ops = assemble(state, 0.0);
            validate_operator_dimensions(initial_ops, state_count);
            // Establish the union once. K, C and A may share structure, while
            // retaining independent numerical storage and arbitrary sparse C.
            auto A = core::add_scaled(initial_ops.K, 0.0, initial_ops.C);
            State rhs(state_count);

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

    mhs::solver::Solution solve(const mhs::model::Model& model, const mhs::model::Study& study,
        std::span<const double> initial_state, const SolveOptions& options)
    {
        return solve(single_model_project(model), study, initial_state, options);
    }

    Solution solve(const model::Project& project, const model::Study& study, std::span<const double> initial_state,
        const SolveOptions& options)
    {
        const auto count = project.dof_count();
        if (count == 0)
            throw std::invalid_argument("solve: project has no degrees of freedom");
        if (!std::isfinite(study.initial_temperature))
            throw std::invalid_argument("solve: initial temperature must be finite");
        if (!initial_state.empty() && initial_state.size() != count)
            throw std::invalid_argument("solve: initial_state size must match the project DOF count");

        core::Vector<double> state(count, study.initial_temperature);
        core::Vector<ProbeTrace> traces;
        const auto components = project.components();
        for (std::size_t i = 0; i < components.size(); ++i) {
            const auto& component = components[i];
            if (const auto* model = std::get_if<std::shared_ptr<const model::Model>>(&component.asset)) {
                for (std::size_t p = 0; p < (*model)->point_probes.size(); ++p)
                    traces.push_back(
                        {(*model)->point_probes[p].name, {}, component.id, static_cast<model::OutputId>(p)});
            }
            else {
                const auto& erom = *std::get<std::shared_ptr<const model::Erom>>(component.asset);
                for (const auto& output : erom.outputs)
                    traces.push_back({output.name, {}, component.id, output.id});
                if (initial_state.empty()) {
                    const auto offset = project.dof_offset(i);
                    const double delta = study.initial_temperature - erom.initial_temperature;
                    if (delta != 0 && erom.uniform_temperature_mode.empty()
                        && study.type == model::StudyType::Transient)
                        throw std::invalid_argument(
                            "EROM requires an explicit modal initial state for a different initial temperature");
                    std::ranges::copy(erom.initial_state, state.begin() + offset);
                    for (std::size_t j = 0; j < erom.uniform_temperature_mode.size(); ++j)
                        state[offset + j] += delta * erom.uniform_temperature_mode[j];
                }
            }
        }
        if (!initial_state.empty())
            std::ranges::copy(initial_state, state.begin());

        ProjectAssembler assembler(project);
        auto assemble = [&](std::span<const double> current, double time) -> const Operators& {
            return assembler.assemble(current, time);
        };
        auto observe = [&](double time, std::span<const double> accepted) {
            std::size_t trace = 0;
            for (std::size_t i = 0; i < components.size(); ++i) {
                const auto values = accepted.subspan(project.dof_offset(i), project.dof_size(i));
                if (const auto* fom = std::get_if<std::shared_ptr<const model::Model>>(&components[i].asset)) {
                    for (const auto& p : (*fom)->point_probes)
                        traces[trace++].values.push_back(assembler.sample_fom(i, p, accepted, time));
                }
                else {
                    const auto& erom = *std::get<std::shared_ptr<const model::Erom>>(components[i].asset);
                    for (std::size_t o = 0; o < erom.outputs.size(); ++o) {
                        const auto row = erom.L.row(o);
                        traces[trace++].values.push_back(
                            std::inner_product(row.begin(), row.end(), values.begin(), erom.outputs[o].offset));
                    }
                }
            }
        };
        auto solution = solve_system(study, assemble, std::move(state), options, observe);
        solution.probe_traces = std::move(traces);
        return solution;
    }

} // namespace mhs::solver
