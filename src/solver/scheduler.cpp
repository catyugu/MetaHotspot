#include "logging/logger.hpp"
#include "runtime/constants.hpp"
#include "solver/assembler.hpp"
#include "solver/nonlinear_solver.hpp"
#include "solver/probe_recorder.hpp"
#include "solver/scheduler.hpp"
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
                throw std::invalid_argument("solve_system: SystemAssembler returned Operators with mismatched K/C/f dimensions (expected "
                    + std::to_string(state_count) + ")");
            }
        }

    } // namespace

    mhs::core::Solution solve_system(const Study& study, const SystemAssembler& assemble,
        std::span<const double> initial_state, const SolveOptions& opts, const StateObserver& observe)
    {
        if (!assemble) {
            throw std::invalid_argument("solve_system: assembler is empty");
        }
        if (initial_state.empty()) {
            throw std::invalid_argument("solve_system: initial_state is empty");
        }

        auto solver = LinearSolver::create(opts.solver);
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
            auto nl_result = nonlinear_solve(build_ls, state, *solver, opts.nonlinear);
            if (observe)
                observe(0.0, state);
            return {std::move(state), state_count, current_time, nl_result.converged, {}};
        }

        // Transient.
        const double duration = study.duration;
        const double output_dt = study.output_interval;
        const double min_dt = opts.min_dt;
        const double max_dt = opts.max_dt;
        time_scheme::StepController step_ctrl {opts.step_strategy, min_dt, max_dt, duration, output_dt, opts.fixed_dt};

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
                    return {std::move(state), state_count, current_time, false, {}};
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

                if (observe && step_ctrl.output_due(current_time))
                    observe(current_time, state);

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

        return {std::move(state), state_count, current_time, true, {}};
    }

} // namespace mhs::sim
