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
    //  solve_system — generic solver (full state, any Assemble callback)
    // -----------------------------------------------------------------------
    mhs::core::SolveResult solve_system(Assemble provider, std::span<const double> initial_state,
        mhs::core::StudyType study_type, double transient_duration, double transient_time_step, const SolverOpts& opts,
        OutputCallback on_output)
    {
        // Validate K/C/f dimension consistency at entry.
        {
            const auto n = initial_state.size();
            auto ops = provider(initial_state, 0.0);
            if (static_cast<std::size_t>(ops.K.rows()) != n || static_cast<std::size_t>(ops.K.cols()) != n
                || static_cast<std::size_t>(ops.C.rows()) != n || static_cast<std::size_t>(ops.C.cols()) != n
                || static_cast<std::size_t>(ops.f.size()) != n) {
                throw std::invalid_argument(
                    "solve_system: provider returned mismatched K/C/f dimensions (expected " + std::to_string(n) + ")");
            }
        }

        auto solver = LinearSolver::create(opts.solver);

        const auto state_count = initial_state.size();
        std::vector<double> state(initial_state.begin(), initial_state.end());

        double current_time = 0.0;

        mhs::core::SolutionHistory accepted {state_count, 2};

        // Steady: single non-linear solve, then output.
        if (study_type == mhs::core::StudyType::Steady) {
            LinearSystemProvider build_ls = [&](std::span<const double> s) -> LinearSystem {
                auto ops = provider(s, 0.0);
                return {std::move(ops.K), std::move(ops.f)};
            };
            auto nl_result = nonlinear_solve(build_ls, state, *solver, opts.nonlinear);
            if (on_output)
                on_output(0.0, state);
            return {std::move(state), current_time, nl_result.converged};
        }

        // Transient.
        const double duration = transient_duration;
        const double output_dt = transient_time_step;

        const double min_dt = opts.min_dt;
        const double max_dt = opts.max_dt;

        time_scheme::StepController step_ctrl {opts.step_strategy, min_dt, max_dt, duration, output_dt, opts.fixed_dt};

        accepted.initialize(state, current_time);
        if (on_output)
            on_output(current_time, state);

        double dt_sug = std::clamp(output_dt, min_dt, max_dt);

        while (current_time < duration - mhs::core::zero_guard) {
            double dt = step_ctrl.prepare(dt_sug, current_time, duration);
            if (dt <= 0.0)
                break;

            // Build the linearised system at (state, time + dt)
            LinearSystemProvider ls_provider = [&](std::span<const double> iter_state) -> LinearSystem {
                auto ops = provider(iter_state, current_time + dt);
                return time_scheme::build_system(opts.integrator, ops, accepted, dt);
            };

            // Save state before trial so we can restore on rejection
            auto saved_state = state;

            // Non-linear solve (Picard/Anderson)
            auto nl = nonlinear_solve(ls_provider, state, *solver, opts.nonlinear);

            if (!nl.converged) {
                // Restore clean state before retrying
                state = std::move(saved_state);
                dt_sug = dt * 0.5;
                MHS_LOG_DEBUG("Step rejected at t={} (nonlinear), retry dt={}", current_time, dt_sug);

                // Fatal: nonlinear divergence at minimum dt
                if (dt <= min_dt * 1.0001) {
                    MHS_LOG_WARN("Nonlinear solver diverged at minimum dt t={}", current_time);
                    auto final_out = step_ctrl.flush_outputs(current_time);
                    for (double t_out : final_out) {
                        if (on_output)
                            on_output(t_out, state);
                    }
                    return {std::move(state), current_time, false};
                }
                continue;
            }

            bool accepted_step = true;
            double suggested_dt_factor = 1.0;

            // Fixed strategy: skip LTE-based rejection entirely
            if (opts.step_strategy == time_scheme::StepStrategy::Fixed) {
                accepted.accept(state, current_time + dt);
            }
            else {
                // Error estimation (LTE check for adaptive stepping)
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

                // AdaptiveFree mode: step end may overshoot output time → interpolate.
                auto out = step_ctrl.flush_outputs(current_time);
                for (double t_out : out) {
                    auto state_at_output = interpolate_state(
                        /* t0 = */ accepted.time_at(1),
                        /* x0 = */ accepted.at(1),
                        /* t1 = */ current_time,
                        /* x1 = */ state, t_out);
                    if (on_output)
                        on_output(t_out, state_at_output);
                }

                // Fixed mode → keep fixed dt; otherwise adapt from error estimate.
                dt_sug = (opts.step_strategy == time_scheme::StepStrategy::Fixed)
                    ? opts.fixed_dt
                    : std::clamp(dt * suggested_dt_factor, min_dt, max_dt);
            }
            else {
                // Restore clean state before retrying
                state = std::move(saved_state);
                dt_sug = dt * 0.5;
                MHS_LOG_DEBUG("Step rejected at t={} (LTE), retry dt={}", current_time, dt_sug);
            }
        }

        // Final flush — ensure last output times are recorded
        auto final_out = step_ctrl.flush_outputs(current_time);
        for (double t_out : final_out) {
            if (on_output)
                on_output(t_out, state);
        }

        return {std::move(state), current_time, true};
    }

    // -----------------------------------------------------------------------
    //  solve_thermal — thermal convenience wrapper
    // -----------------------------------------------------------------------
    mhs::core::ThermalSolution solve_thermal(
        const mhs::core::Model& model, const SolverOpts& opts, std::span<const double> initial_state)
    {
        const auto cell_count = static_cast<std::size_t>(model.cells.cell_to_grid.size());

        // Validate initial_state size if provided
        if (!initial_state.empty() && initial_state.size() != cell_count) {
            throw std::invalid_argument("solve_thermal: initial_state.size() = " + std::to_string(initial_state.size())
                + " != cell_count = " + std::to_string(cell_count));
        }

        ProbeRecorder probe_recorder;
        probe_recorder.initialize(model);

        // Build provider: (state, time) -> assemble_thermal(model, state, time)
        Assemble provider = [&model](std::span<const double> s, double t) { return assemble_thermal(model, s, t); };

        // Build initial state if not provided
        std::vector<double> default_state;
        if (initial_state.empty()) {
            default_state.assign(cell_count, model.initial_temperature);
            initial_state = default_state;
        }

        // Output callback captures probe recordings
        auto on_output = [&](double t, std::span<const double> state) { probe_recorder.record(t, state); };

        auto sol = solve_system(provider, initial_state, model.study_type, model.transient_duration,
            model.transient_time_step, opts, on_output);

        mhs::core::ThermalSolution result;
        result.temperature = std::move(sol.state);
        result.time = sol.time;
        result.converged = sol.converged;
        result.probe_traces = probe_recorder.traces();
        return result;
    }

} // namespace mhs::sim
