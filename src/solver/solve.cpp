#include "solver/solve.hpp"
#include "logging/logger.hpp"
#include "common/solver.hpp"
#include "common/constants.hpp"
#include "solver/assembler.hpp"
#include "solver/nonlinear_solver.hpp"
#include "solver/probe_recorder.hpp"
#include "solver/solution_history.hpp"
#include "solver/time_integration.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace mhs::sim {

    namespace {

        void validate_operator_dimensions(const Operators& ops, std::size_t state_count)
        {
            if (static_cast<std::size_t>(ops.K.rows()) != state_count
                || static_cast<std::size_t>(ops.K.cols()) != state_count
                || static_cast<std::size_t>(ops.C.rows()) != state_count
                || static_cast<std::size_t>(ops.C.cols()) != state_count
                || static_cast<std::size_t>(ops.f.size()) != state_count) {
                throw std::invalid_argument(
                    "solve_system: SystemAssembler returned Operators with mismatched K/C/f dimensions (expected "
                    + std::to_string(state_count) + ")");
            }
        }

        SolverSpec build_solver_spec(const SolveOptions& opts)
        {
            SolverType type;
            switch (opts.linear_solver) {
            case SolveOptions::LinearSolverType::Pardiso:
                type = SolverType::Pardiso;
                break;
            case SolveOptions::LinearSolverType::EigenSparseLU:
                type = SolverType::EigenSparseLU;
                break;
            case SolveOptions::LinearSolverType::EigenBiCGSTAB:
                type = SolverType::EigenBiCGSTAB;
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

        time_scheme::IntegratorKind build_integrator(const SolveOptions& opts)
        {
            switch (opts.integrator) {
            case SolveOptions::Integrator::Bdf1:
                return time_scheme::IntegratorKind::Bdf1;
            case SolveOptions::Integrator::Bdf2:
                return time_scheme::IntegratorKind::Bdf2;
            default:
                throw std::invalid_argument("build_integrator: unknown integrator");
            }
        }

        time_scheme::StepStrategy build_strategy(const SolveOptions& opts)
        {
            switch (opts.step_strategy) {
            case SolveOptions::StepStrategy::Adaptive:
                return time_scheme::StepStrategy::Adaptive;
            case SolveOptions::StepStrategy::Fixed:
                return time_scheme::StepStrategy::Fixed;
            default:
                throw std::invalid_argument("build_strategy: unknown step_strategy");
            }
        }

    } // namespace

    // ---- Internal callback-driven solver ----

    mhs::core::Solution solve_system(const Study& study, const SystemAssembler& assemble,
        std::span<const double> initial_state, const SolveOptions& opts, const StateObserver& observe)
    {
        if (!assemble) {
            throw std::invalid_argument("solve_system: assembler is empty");
        }
        if (initial_state.empty()) {
            throw std::invalid_argument("solve_system: initial_state is empty");
        }

        auto solver_spec = build_solver_spec(opts);
        auto nl_config = build_nonlinear_config(opts);
        auto integrator = build_integrator(opts);
        auto step_strategy = build_strategy(opts);

        auto solver = LinearSolver::create(solver_spec);
        std::vector<double> state(initial_state.begin(), initial_state.end());
        const auto state_count = state.size();
        double current_time = 0.0;
        mhs::core::SolutionHistory accepted {state_count, 2};

        // Steady: single non-linear solve, then output.
        if (study.type == mhs::core::StudyType::Steady) {
            LinearSystemProvider build_ls = [&](std::span<const double> s) -> LinearSystem {
                auto ops = assemble(s, 0.0);
                validate_operator_dimensions(ops, state_count);
                return {std::move(ops.K), std::move(ops.f)};
            };
            auto nl_result = nonlinear_solve(build_ls, state, *solver, nl_config);
            if (observe)
                observe(0.0, state);
            return {std::move(state), state_count, current_time, nl_result.converged, {}};
        }

        // Transient.
        const double duration = study.duration;
        const double output_dt = study.output_interval;
        const double min_dt = opts.min_dt;
        const double max_dt = opts.max_dt;
        time_scheme::StepController step_ctrl {step_strategy, min_dt, max_dt, duration, output_dt, opts.fixed_dt};

        accepted.initialize(state, current_time);
        if (observe)
            observe(current_time, state);

        double dt_sug = std::clamp(output_dt, min_dt, max_dt);

        while (current_time < duration - mhs::core::zero_guard) {
            double dt = step_ctrl.prepare(dt_sug, current_time);
            if (dt <= 0.0)
                break;

            // Build the linearised system at (state, time + dt)
            LinearSystemProvider ls_provider = [&](std::span<const double> iter_state) -> LinearSystem {
                auto ops = assemble(iter_state, current_time + dt);
                validate_operator_dimensions(ops, state_count);
                return time_scheme::build_system(integrator, ops, accepted, dt);
            };

            // Save state before trial so we can restore on rejection
            auto saved_state = state;

            // Non-linear solve (Picard/Anderson)
            auto nl = nonlinear_solve(ls_provider, state, *solver, nl_config);

            if (!nl.converged) {
                state = std::move(saved_state);
                dt_sug = dt * 0.5;
                MHS_LOG_DEBUG("Step rejected at t={} (nonlinear), retry dt={}", current_time, dt_sug);

                if (dt <= min_dt * 1.0001) {
                    MHS_LOG_WARN("Nonlinear solver diverged at minimum dt t={}", current_time);
                    return {std::move(state), state_count, current_time, false, {}};
                }
                continue;
            }

            bool accepted_step = true;
            double suggested_dt_factor = 1.0;

            if (step_strategy == time_scheme::StepStrategy::Fixed) {
                accepted.accept(state, current_time + dt);
            }
            else {
                auto est = time_scheme::estimate_error(accepted, state, dt, {opts.error_abs_tol, opts.error_safety});
                suggested_dt_factor = est.suggested_factor;
                accepted_step = (est.error_ratio <= 1.0) || (dt <= min_dt * 1.0001);

                if (accepted_step) {
                    accepted.accept(state, current_time + dt);
                }
            }

            if (accepted_step) {
                current_time += dt;
                MHS_LOG_DEBUG("Time: {} solved (dt={})", current_time, dt);

                if (observe && step_ctrl.output_due(current_time))
                    observe(current_time, state);

                dt_sug = (step_strategy == time_scheme::StepStrategy::Fixed)
                    ? opts.fixed_dt
                    : std::clamp(dt * suggested_dt_factor, min_dt, max_dt);
            }
            else {
                state = std::move(saved_state);
                dt_sug = dt * 0.5;
                MHS_LOG_DEBUG("Step rejected at t={} (LTE), retry dt={}", current_time, dt_sug);
            }
        }

        return {std::move(state), state_count, current_time, true, {}};
    }

    // ---- Public solve entry point ----

    mhs::core::Solution solve(
        const mhs::core::Model& model, std::span<const double> initial_state, const SolveOptions& options)
    {
        const auto fvm_count = model.cells.cell_to_grid.size();
        if (fvm_count == 0) {
            throw std::invalid_argument("solve: model has zero cells");
        }

        std::vector<double> state;
        if (initial_state.empty()) {
            state.assign(fvm_count, model.initial_temperature);
        }
        else {
            if (initial_state.size() != fvm_count) {
                throw std::invalid_argument("solve: initial_state size (" + std::to_string(initial_state.size())
                    + ") must equal fvm_count (" + std::to_string(fvm_count) + ")");
            }
            state.assign(initial_state.begin(), initial_state.end());
        }

        Study study {model.study_type, model.transient_duration, model.transient_time_step};
        SystemAssembler assemble
            = [&](std::span<const double> s, double t) { return mhs::sim::assemble_thermal(model, s, t); };

        ProbeRecorder probe_recorder;
        probe_recorder.initialize(model);
        StateObserver observe
            = [&](double t, std::span<const double> accepted_state) { probe_recorder.record(t, accepted_state); };

        auto sol = solve_system(study, assemble, state, options, observe);
        sol.fvm_count = fvm_count;
        sol.probe_traces = probe_recorder.traces();
        return sol;
    }

} // namespace mhs::sim
