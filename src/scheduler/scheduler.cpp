#include "assembler/assembler.hpp"
#include "common/logger.hpp"
#include "nonlinear/nonlinear_solver.hpp"
#include "preprocessor/fluid_preprocessor.hpp"
#include "scheduler/scheduler.hpp"
#include "time_scheme/error_controller.hpp"
#include "time_scheme/integrator.hpp"
#include "time_scheme/step_controller.hpp"

#include <Eigen/Core>
#include <algorithm>

namespace mhs::sim {

    void Scheduler::setModel(mhs::core::InternalModel* model)
    {
        model_ = model;
        if (model_)
            probe_recorder_.initialize(*model);
    }

    void Scheduler::setSolver(std::unique_ptr<LinearSolver> solver) { solver_ = std::move(solver); }

    namespace {

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

    void Scheduler::run()
    {
        if (!model_ || !solver_) {
            MHS_FATAL("Scheduler: model or solver not set");
        }

        const std::size_t N = static_cast<std::size_t>(model_->cells.cell_bcs.size());
        state_.T.assign(N, model_->initial_temperature);
        state_.current_time = 0.0;
        state_.time_step = 0;

        Assembler assembler(*model_);
        solveFluidFlow(*model_);

        // Steady: single non-linear solve, then output.
        if (model_->study_type == mhs::core::StudyType::Steady) {
            LinearSystemProvider build_ls = [&](const mhs::core::GlobalState& s) -> LinearSystem {
                auto ops = assembler.assemble(s);
                return {std::move(ops.K), std::move(ops.f)};
            };
            nonlinear_solve(build_ls, state_, *solver_);
            solution_ = state_.T;
            probe_recorder_.record(state_.current_time, state_.T);
            return;
        }

        // Transient.
        const double duration = model_->transient_duration;
        const double output_dt = model_->transient_time_step;

        time_scheme::StepController step_ctrl {
            time_scheme::StepStrategy::Free, /*min_dt=*/1e-12, /*max_dt=*/duration, /*fixed_dt=*/output_dt};
        const double min_dt = step_ctrl.min_dt();
        const double max_dt = step_ctrl.max_dt();

        // Solution-history ring buffer: capacity 2 (one for current, one for
        // the previous step — enough for BDF2's startup sequence).
        state_.accepted = mhs::core::SolutionHistory(N, 2);
        step_ctrl.rebuild(duration, output_dt);

        state_.accepted.initialize(state_.T, state_.current_time);
        probe_recorder_.record(state_.current_time, state_.T);

        double dt_sug = std::clamp(output_dt / 10.0, min_dt, max_dt);
        double dt = dt_sug;

        while (state_.current_time < duration - 1e-14) {
            dt = step_ctrl.prepare(dt_sug, state_.current_time, duration);
            if (dt <= 0.0)
                break;
            state_.dt = dt;

            // Re-assembles K, f inside each non-linear iteration; M_diag is
            // frozen at accepted.current() via build_system's BDF stencil logic.
            LinearSystemProvider provider = [&](const mhs::core::GlobalState& s) -> LinearSystem {
                return time_scheme::build_system(
                    time_scheme::IntegratorKind::Bdf1, assembler.assemble(s), s.accepted, s.dt);
            };
            auto nl = nonlinear_solve(provider, state_, *solver_);
            if (!nl.converged) {
                MHS_LOG_WARN("Non-linear iteration did not converge at step {}", state_.time_step);
            }

            auto est = time_scheme::estimate_error(state_.accepted, state_.T, dt, /*err_cfg=*/ {});
            dt_sug = std::clamp(dt * est.suggested_factor, min_dt, max_dt);
            const bool accepted_step = (est.error_ratio <= 1.0) || (dt <= min_dt * 1.0001);

            if (accepted_step) {
                state_.current_time += dt;
                state_.time_step++;
                state_.accepted.accept(state_.T, state_.current_time);

                MHS_LOG_DEBUG("Time: {} solved (dt={})", state_.current_time, dt);

                // Free mode: step end may overshoot output time → interpolate.
                auto out = step_ctrl.flush_outputs(state_.current_time);
                for (double t_out : out) {
                    auto T_interp = lerp_T(
                        /* t0 = */ state_.accepted.time_at(1),
                        /* T0 = */ state_.accepted.at(1),
                        /* t1 = */ state_.current_time,
                        /* T1 = */ state_.T, t_out);
                    probe_recorder_.record(t_out, T_interp);
                }
            }
            else {
                dt_sug = dt * 0.5;
                MHS_LOG_DEBUG("Step rejected at t={}, retry dt={}", state_.current_time, dt_sug);
            }
        }

        solution_ = state_.T;
    }

} // namespace mhs::sim
