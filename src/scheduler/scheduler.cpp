#include "assembler/assembler.hpp"
#include "data/solution_history.hpp"
#include "data/tolerance_config.hpp"
#include "logger/logger.hpp"
#include "nonlinear/nonlinear_solver.hpp"
#include "scheduler/probe_recorder.hpp"
#include "scheduler/scheduler.hpp"
#include "time_scheme/error_controller.hpp"
#include "time_scheme/integrator.hpp"
#include "time_scheme/step_controller.hpp"

#include <Eigen/Core>
#include <algorithm>

namespace mhs::sim {

    namespace {

        struct StepState {
            double current_time = 0.0;
            int time_step = 0;
            double dt = 0.0;
            mhs::core::SolutionHistory accepted {0, 1};
            std::vector<double> T;
        };

        /// Linear interpolation of the temperature field between two snapshots.
        inline std::vector<double> lerp_T(
            double t0, const std::vector<double>& T0, double t1, const std::vector<double>& T1, double t)
        {
            const double dt = t1 - t0;
            if (dt <= 0.0)
                return T0;
            const double s = (t - t0) / dt; // ∈ [0, 1]
            const std::size_t N = T0.size();
            std::vector<double> out(N);
            for (std::size_t i = 0; i < N; ++i)
                out[i] = T0[i] + s * (T1[i] - T0[i]);
            return out;
        }

    } // anonymous namespace

    mhs::core::Solution solve(const mhs::core::Model& model, const SolveOptions& options)
    {
        auto solver = LinearSolver::create(options.solver);
        ProbeRecorder probe_recorder;
        probe_recorder.initialize(model);
        StepState step;

        const mhs::Index N = static_cast<mhs::Index>(model.cells.material_id.size());
        step.T.resize(N);

        std::fill_n(step.T.data(), N, model.initial_temperature);
        step.current_time = 0.0;
        step.time_step = 0;

        Assembler assembler(model);

        // Steady: single non-linear solve, then output.
        if (model.study_type == mhs::core::StudyType::Steady) {
            LinearSystemProvider build_ls = [&](Eigen::Ref<const Eigen::VectorXd> T_in) -> LinearSystem {
                AssembleContext ctx {T_in, step.current_time};
                auto ops = assembler.assemble(ctx);
                return {std::move(ops.K), std::move(ops.f)};
            };
            Eigen::Map<Eigen::VectorXd> T_map(step.T.data(), static_cast<Eigen::Index>(N));
            nonlinear_solve(build_ls, T_map, *solver, options.nonlinear);
            probe_recorder.record(step.current_time, step.T);
            return {std::move(step.T), step.current_time, probe_recorder.traces()};
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
        step.accepted = mhs::core::SolutionHistory(static_cast<std::size_t>(N), 2);
        step_ctrl.rebuild(duration, output_dt);

        step.accepted.initialize(step.T, step.current_time);
        probe_recorder.record(step.current_time, step.T);

        double dt_sug = std::clamp(output_dt / 10.0, min_dt, max_dt);
        double dt = dt_sug;

        Eigen::Map<Eigen::VectorXd> T_map(step.T.data(), static_cast<Eigen::Index>(N));

        while (step.current_time < duration - mhs::core::zero_guard) {
            dt = step_ctrl.prepare(dt_sug, step.current_time, duration);
            if (dt <= 0.0)
                break;
            step.dt = dt;

            // Re-assembles K, f inside each non-linear iteration; M_diag is
            // frozen at accepted.current() via build_system's BDF stencil logic.
            LinearSystemProvider provider = [&](Eigen::Ref<const Eigen::VectorXd> T_in) -> LinearSystem {
                AssembleContext ctx {T_in, step.current_time};
                return time_scheme::build_system(
                    time_scheme::IntegratorKind::Bdf1, assembler.assemble(ctx), step.accepted, step.dt);
            };
            auto nl = nonlinear_solve(provider, T_map, *solver, options.nonlinear);
            if (!nl.converged) {
                MHS_LOG_WARN("Non-linear iteration did not converge at step {}", step.time_step);
            }

            auto est = time_scheme::estimate_error(step.accepted, step.T, dt, /*err_cfg=*/ {});
            dt_sug = std::clamp(dt * est.suggested_factor, min_dt, max_dt);
            const bool accepted_step = (est.error_ratio <= 1.0) || (dt <= min_dt * 1.0001);

            if (accepted_step) {
                step.current_time += dt;
                step.time_step++;
                step.accepted.accept(step.T, step.current_time);

                MHS_LOG_DEBUG("Time: {} solved (dt={})", step.current_time, dt);

                // Free mode: step end may overshoot output time → interpolate.
                auto out = step_ctrl.flush_outputs(step.current_time);
                for (double t_out : out) {
                    auto T_interp = lerp_T(
                        /* t0 = */ step.accepted.time_at(1),
                        /* T0 = */ step.accepted.at(1),
                        /* t1 = */ step.current_time,
                        /* T1 = */ step.T, t_out);
                    probe_recorder.record(t_out, T_interp);
                }
            }
            else {
                dt_sug = dt * 0.5;
                MHS_LOG_DEBUG("Step rejected at t={}, retry dt={}", step.current_time, dt_sug);
            }
        }

        return {std::move(step.T), step.current_time, probe_recorder.traces()};
    }

} // namespace mhs::sim
