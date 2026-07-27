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
        std::vector<double>& state, double current_time, double dt, const SolverOpts& opts)
    {
        StepResult result {};

        // Build the linearised system at (state, time + dt)
        LinearSystemProvider provider = [&](std::vector<double>& iter_state) -> LinearSystem {
            AssembleContext ctx {iter_state, current_time + dt};
            auto ops = assembler.assemble(ctx);
            return time_scheme::build_system(opts.integrator, ops, history, dt);
        };

        // Non-linear solve (Picard/Anderson)
        auto nl = nonlinear_solve(provider, state, solver, opts.nonlinear);
        result.nonlinear_converged = nl.converged;
        result.nonlinear_iterations = nl.iterations;

        if (!nl.converged) {
            result.accepted = false;
            result.error_ratio = 1.0;
            result.suggested_dt_factor = 0.5;
            return result;
        }

        // Fixed strategy: skip LTE-based rejection entirely
        if (opts.step_strategy == time_scheme::StepStrategy::Fixed) {
            history.accept(state, current_time + dt);
            result.accepted = true;
            result.error_ratio = 0.0;
            result.suggested_dt_factor = 1.0;
            return result;
        }

        // Error estimation (LTE check for adaptive stepping)
        auto est = time_scheme::estimate_error(history, state, dt, {opts.error_abs_tol, opts.error_safety});
        result.error_ratio = est.error_ratio;
        result.suggested_dt_factor = est.suggested_factor;

        // Acceptance criterion
        double min_dt = opts.min_dt;
        result.accepted = (est.error_ratio <= 1.0) || (dt <= min_dt * 1.0001);

        if (result.accepted) {
            history.accept(state, current_time + dt);
        }

        return result;
    }

    mhs::core::Solution solve(
        const mhs::core::Model& model, const SolverOpts& opts, std::span<const double> initial_state)
    {
        auto solver = LinearSolver::create(opts.solver);
        ProbeRecorder probe_recorder;
        probe_recorder.initialize(model);

        const mhs::core::Index state_count = model.cells.cell_to_grid.size();
        std::vector<double> state;
        if (!initial_state.empty()) {
            state.assign(initial_state.begin(), initial_state.end());
        }
        else {
            state.assign(static_cast<std::size_t>(state_count), model.initial_temperature);
        }
        assert(static_cast<mhs::core::Index>(state.size()) == state_count);

        double current_time = 0.0;
        double dt = 0.0;

        mhs::core::SolutionHistory accepted {static_cast<std::size_t>(state_count), 2};

        Assembler assembler(model);

        // Steady: single non-linear solve, then output.
        if (model.study_type == mhs::core::StudyType::Steady) {
            LinearSystemProvider build_ls = [&](std::vector<double>& state) -> LinearSystem {
                AssembleContext ctx {state, 0.0};
                auto ops = assembler.assemble(ctx);
                return {std::move(ops.K), std::move(ops.f)};
            };
            nonlinear_solve(build_ls, state, *solver, opts.nonlinear);
            probe_recorder.record(0.0, state);
            return {std::move(state), current_time, probe_recorder.traces()};
        }

        // Transient.
        const double duration = model.transient_duration;
        const double output_dt = model.transient_time_step;

        const double min_dt = opts.min_dt;
        const double max_dt = std::max(opts.max_dt, duration);

        time_scheme::StepController step_ctrl {opts.step_strategy, min_dt, max_dt, duration, output_dt, opts.fixed_dt};

        accepted.initialize(state, current_time);
        probe_recorder.record(current_time, state);

        double dt_sug = std::clamp(output_dt, min_dt, max_dt);

        while (current_time < duration - mhs::core::zero_guard) {
            dt = step_ctrl.prepare(dt_sug, current_time, duration);
            if (dt <= 0.0)
                break;

            // Save state before trial so we can restore on rejection
            auto saved_state = state;
            auto result = take_step(assembler, *solver, accepted, state, current_time, dt, opts);

            if (result.accepted) {
                current_time += dt;
                MHS_LOG_DEBUG("Time: {} solved (dt={})", current_time, dt);

                // AdaptiveFree mode: step end may overshoot output time → interpolate.
                auto out = step_ctrl.flush_outputs(current_time);
                for (double t_out : out) {
                    auto state_at_output = interpolate_state(
                        /* t0 = */ accepted.time_at(1),
                        /* x0 = */ accepted.at(1),
                        /* t1 = */ current_time,
                        /* x1 = */ state, t_out);
                    probe_recorder.record(t_out, state_at_output);
                }

                // Fixed mode → keep fixed dt; otherwise adapt from error estimate.
                dt_sug = (opts.step_strategy == time_scheme::StepStrategy::Fixed)
                    ? opts.fixed_dt
                    : std::clamp(dt * result.suggested_dt_factor, min_dt, max_dt);
            }
            else {
                // Restore clean state before retrying
                state = std::move(saved_state);
                dt_sug = dt * 0.5;
                MHS_LOG_DEBUG("Step rejected at t={}, retry dt={}", current_time, dt_sug);

                // Check for fatal: nonlinear divergence at minimum dt
                if (!result.nonlinear_converged && dt <= min_dt * 1.0001) {
                    MHS_LOG_WARN("Nonlinear solver diverged at minimum dt t={}", current_time);
                    // Flush any remaining outputs
                    auto final_out = step_ctrl.flush_outputs(current_time);
                    for (double t_out : final_out)
                        probe_recorder.record(t_out, state);
                    return {std::move(state), current_time, probe_recorder.traces(), false};
                }
            }
        }

        // Final flush — ensure last output times are recorded
        auto final_out = step_ctrl.flush_outputs(current_time);
        for (double t_out : final_out)
            probe_recorder.record(t_out, state);

        return {std::move(state), current_time, probe_recorder.traces()};
    }

} // namespace mhs::sim
