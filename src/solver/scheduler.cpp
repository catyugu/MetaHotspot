#include "logging/logger.hpp"
#include "runtime/constants.hpp"
#include "solver/assembler.hpp"
#include "solver/nonlinear_solver.hpp"
#include "solver/probe_recorder.hpp"
#include "solver/scheduler.hpp"
#include "solver/solution_history.hpp"
#include "solver/time_integration.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <cstddef>

namespace mhs::sim {

    namespace {

        struct StepState {
            double current_time = 0.0;
            int time_step = 0;
            double dt = 0.0;
            mhs::core::SolutionHistory accepted {0, 1};
            std::vector<double> state;
        };

        std::vector<double> extract_cell_temperature(const mhs::core::Model& model, std::span<const double> state)
        {
            const auto& cells = model.dofs.cell_states;
            return {state.begin() + static_cast<std::ptrdiff_t>(cells.begin),
                state.begin() + static_cast<std::ptrdiff_t>(cells.end())};
        }

        /// Linear interpolation of the state between two snapshots.
        inline std::vector<double> interpolate_state(
            double t0, std::span<const double> x0, double t1, std::span<const double> x1, double t)
        {
            const double dt = t1 - t0;
            if (dt <= 0.0)
                return std::vector<double>(x0.begin(), x0.end());
            const double s = (t - t0) / dt; // ∈ [0, 1]
            std::vector<double> out(x0.size());
            for (std::size_t i = 0; i < x0.size(); ++i)
                out[i] = x0[i] + s * (x1[i] - x0[i]);
            return out;
        }

    } // anonymous namespace

    // -----------------------------------------------------------------------
    //  take_step — shared kernel for single transient step
    // -----------------------------------------------------------------------
    StepResult take_step(Assembler& assembler, LinearSolver& solver, mhs::core::SolutionHistory& history,
        std::vector<double>& state, double current_time, double dt, const NonLinearConfig& nonlinear_cfg,
        time_scheme::IntegratorKind integrator)
    {
        StepResult result {};

        // Build the linearised system at (state, time + dt)
        LinearSystemProvider provider = [&](std::vector<double>& iter_state) -> LinearSystem {
            AssembleContext ctx {iter_state, current_time + dt};
            auto ops = assembler.assemble(ctx);
            return time_scheme::build_system(integrator, ops, history, dt);
        };

        // Non-linear solve (Picard/Anderson)
        auto nl = nonlinear_solve(provider, state, solver, nonlinear_cfg);
        result.nonlinear_converged = nl.converged;
        result.nonlinear_iterations = nl.iterations;

        if (!nl.converged) {
            result.accepted = false;
            result.error_ratio = 1.0;
            result.suggested_dt_factor = 0.5;
            return result;
        }

        // Error estimation (LTE check for adaptive stepping)
        auto est = time_scheme::estimate_error(history, state, dt, {});
        result.error_ratio = est.error_ratio;
        result.suggested_dt_factor = est.suggested_factor;

        // Acceptance criterion
        double min_dt = 1e-12;
        result.accepted = (est.error_ratio <= 1.0) || (dt <= min_dt * 1.0001);

        if (result.accepted) {
            history.accept(state, current_time + dt);
        }

        return result;
    }

    mhs::core::Solution solve(
        const mhs::core::Model& model, const SolveOptions& options, std::span<const double> initial_state)
    {
        auto solver = LinearSolver::create(options.solver);
        ProbeRecorder probe_recorder;
        probe_recorder.initialize(model);
        StepState step;

        const mhs::core::Index state_count = model.dofs.total_count;
        if (!initial_state.empty()) {
            step.state.assign(initial_state.begin(), initial_state.end());
        }
        else {
            step.state.assign(static_cast<std::size_t>(state_count), model.initial_temperature);
        }
        assert(step.state.size() == state_count);
        step.current_time = 0.0;
        step.time_step = 0;

        Assembler assembler(model);

        // Steady: single non-linear solve, then output.
        if (model.study_type == mhs::core::StudyType::Steady) {
            LinearSystemProvider build_ls = [&](std::vector<double>& state) -> LinearSystem {
                AssembleContext ctx {state, 0.0};
                auto ops = assembler.assemble(ctx);
                return {std::move(ops.K), std::move(ops.f)};
            };
            nonlinear_solve(build_ls, step.state, *solver, options.nonlinear);
            auto cell_temperature = extract_cell_temperature(model, step.state);
            probe_recorder.record(0.0, cell_temperature);
            return {std::move(step.state), std::move(cell_temperature), step.current_time, probe_recorder.traces()};
        }

        // Transient.
        const double duration = model.transient_duration;
        const double output_dt = model.transient_time_step;

        time_scheme::StepController step_ctrl {
            time_scheme::StepStrategy::Free, /*min_dt=*/1e-12, /*max_dt=*/duration, /*fixed_dt=*/output_dt};
        const double min_dt = step_ctrl.min_dt();
        const double max_dt = step_ctrl.max_dt();

        step.accepted = mhs::core::SolutionHistory(static_cast<std::size_t>(state_count), 2);
        step_ctrl.rebuild(duration, output_dt);

        step.accepted.initialize(step.state, step.current_time);
        probe_recorder.record(step.current_time, extract_cell_temperature(model, step.state));

        double dt_sug = std::clamp(output_dt, min_dt, max_dt);
        double dt = dt_sug;

        while (step.current_time < duration - mhs::core::zero_guard) {
            dt = step_ctrl.prepare(dt_sug, step.current_time, duration);
            if (dt <= 0.0)
                break;
            step.dt = dt;

            // Use shared take_step kernel.
            auto result
                = take_step(assembler, *solver, step.accepted, step.state, step.current_time, dt, options.nonlinear);

            if (result.accepted) {
                step.current_time += dt;
                step.time_step++;

                MHS_LOG_DEBUG("Time: {} solved (dt={})", step.current_time, dt);

                // Free mode: step end may overshoot output time → interpolate.
                auto out = step_ctrl.flush_outputs(step.current_time);
                for (double t_out : out) {
                    auto state_at_output = interpolate_state(
                        /* t0 = */ step.accepted.time_at(1),
                        /* x0 = */ step.accepted.at(1),
                        /* t1 = */ step.current_time,
                        /* x1 = */ step.state, t_out);
                    probe_recorder.record(t_out, extract_cell_temperature(model, state_at_output));
                }

                // Manual mode → keep fixed dt; otherwise adapt from error estimate.
                dt_sug = (step_ctrl.strategy() == time_scheme::StepStrategy::Manual)
                    ? output_dt
                    : std::clamp(dt * result.suggested_dt_factor, min_dt, max_dt);
            }
            else {
                dt_sug = dt * 0.5;
                MHS_LOG_DEBUG("Step rejected at t={}, retry dt={}", step.current_time, dt_sug);
            }
        }

        auto cell_temperature = extract_cell_temperature(model, step.state);
        return {std::move(step.state), std::move(cell_temperature), step.current_time, probe_recorder.traces()};
    }

} // namespace mhs::sim
