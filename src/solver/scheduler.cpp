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

        std::vector<double> extract_cell_temperature(const mhs::core::Model& model, const std::vector<double>& state)
        {
            const auto& cells = model.dofs.cell_states;
            return {state.begin() + static_cast<std::ptrdiff_t>(cells.begin),
                state.begin() + static_cast<std::ptrdiff_t>(cells.end())};
        }

        /// Linear interpolation of the state between two snapshots.
        inline std::vector<double> interpolate_state(
            double t0, const std::vector<double>& x0, double t1, const std::vector<double>& x1, double t)
        {
            const double dt = t1 - t0;
            if (dt <= 0.0)
                return x0;
            const double s = (t - t0) / dt; // ∈ [0, 1]
            std::vector<double> out(x0.size());
            for (std::size_t i = 0; i < x0.size(); ++i)
                out[i] = x0[i] + s * (x1[i] - x0[i]);
            return out;
        }

    } // anonymous namespace

    mhs::core::Solution solve(const mhs::core::Model& model, const SolveOptions& options)
    {
        auto solver = LinearSolver::create(options.solver);
        ProbeRecorder probe_recorder;
        probe_recorder.initialize(model);
        StepState step;

        const mhs::core::Index state_count = model.dofs.total_count;
        step.state = model.initial_state;
        assert(step.state.size() == state_count);
        step.current_time = 0.0;
        step.time_step = 0;

        Assembler assembler(model);

        // Steady: single non-linear solve, then output.
        if (model.study_type == mhs::core::StudyType::Steady) {
            LinearSystemProvider build_ls = [&](std::vector<double>& state) -> LinearSystem {
                AssembleContext ctx {state, step.current_time};
                auto ops = assembler.assemble(ctx);
                return {std::move(ops.K), std::move(ops.f)};
            };
            nonlinear_solve(build_ls, step.state, *solver, options.nonlinear);
            auto cell_temperature = extract_cell_temperature(model, step.state);
            probe_recorder.record(step.current_time, cell_temperature);
            return {std::move(step.state), std::move(cell_temperature), step.current_time, probe_recorder.traces()};
        }

        // Transient.
        const double duration = model.transient_duration;
        const double output_dt = model.transient_time_step;

        time_scheme::StepController step_ctrl {
            time_scheme::StepStrategy::Free, /*min_dt=*/1e-12, /*max_dt=*/duration, /*fixed_dt=*/output_dt};
        const double min_dt = step_ctrl.min_dt();
        const double max_dt = step_ctrl.max_dt();

        // Solution-history ring buffer: capacity 2 (one for current, one for
        // the previous step — enough for BDF2's startup sequence).
        step.accepted = mhs::core::SolutionHistory(static_cast<std::size_t>(state_count), 2);
        step_ctrl.rebuild(duration, output_dt);

        step.accepted.initialize(step.state, step.current_time);
        probe_recorder.record(step.current_time, extract_cell_temperature(model, step.state));

        double dt_sug = std::clamp(output_dt / 10.0, min_dt, max_dt);
        double dt = dt_sug;

        while (step.current_time < duration - mhs::core::zero_guard) {
            dt = step_ctrl.prepare(dt_sug, step.current_time, duration);
            if (dt <= 0.0)
                break;
            step.dt = dt;

            // Re-assemble state-dependent operators inside each non-linear iteration.
            LinearSystemProvider provider = [&](std::vector<double>& state) -> LinearSystem {
                AssembleContext ctx {state, step.current_time};
                return time_scheme::build_system(
                    time_scheme::IntegratorKind::Bdf1, assembler.assemble(ctx), step.accepted, step.dt);
            };
            auto nl = nonlinear_solve(provider, step.state, *solver, options.nonlinear);
            if (!nl.converged) {
                MHS_LOG_WARN("Non-linear iteration did not converge at step {}", step.time_step);
            }

            auto est = time_scheme::estimate_error(step.accepted, step.state, dt, /*err_cfg=*/ {});
            dt_sug = std::clamp(dt * est.suggested_factor, min_dt, max_dt);
            const bool accepted_step = (est.error_ratio <= 1.0) || (dt <= min_dt * 1.0001);

            if (accepted_step) {
                step.current_time += dt;
                step.time_step++;
                step.accepted.accept(step.state, step.current_time);

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
