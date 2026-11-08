#include "solver/solve.hpp"
#include "core/csr_ops.hpp"
#include "core/vector.hpp"
#include "logging/logger.hpp"
#include "solver/assembler.hpp"
#include "solver/nonlinear_solver.hpp"
#include "solver/probe_recorder.hpp"
#include "solver/solver.hpp"
#include <atstep/stepper.hpp>

#include <cmath>
#include <limits>
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

    mhs::solver::Solution solve(const mhs::model::Model& model, const mhs::model::Study& study,
        std::span<const double> initial_state, const SolveOptions& options)
    {
        const auto cell_count = model.discrete.cells.size();
        if (cell_count == 0) {
            throw std::invalid_argument("solve: model has zero cells");
        }

        if (!std::isfinite(study.initial_temperature))
            throw std::invalid_argument("solve: initial temperature must be finite");

        mhs::core::Vector<double> state;
        if (initial_state.empty()) {
            state.assign(cell_count, study.initial_temperature);
        }
        else {
            if (initial_state.size() != cell_count) {
                throw std::invalid_argument("solve: initial_state size (" + std::to_string(initial_state.size())
                    + ") must equal cell_count (" + std::to_string(cell_count) + ")");
            }
            state.assign(initial_state.begin(), initial_state.end());
        }

        ThermalAssembler assembler(model);
        SystemAssembler assemble = [&](std::span<const double> current, double time) -> const Operators& {
            return assembler.assemble(current, time);
        };

        ProbeRecorder probe_recorder(model);
        StateObserver observe
            = [&](double t, std::span<const double> accepted_state) { probe_recorder.record(t, accepted_state); };

        auto sol = solve_system(study, assemble, std::move(state), options, observe);
        sol.probe_traces = std::move(probe_recorder).take_traces();
        return sol;
    }

} // namespace mhs::solver
