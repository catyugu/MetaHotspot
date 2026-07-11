#include "assembler/assembler.hpp"
#include "common/logger.hpp"
#include "data/tolerance_config.hpp"
#include "nonlinear/nonlinear_solver.hpp"
#include "scheduler/scheduler.hpp"
#include "time_scheme/error_controller.hpp"
#include "time_scheme/integrator.hpp"
#include "time_scheme/step_controller.hpp"

#include <Eigen/Core>
#include <algorithm>

namespace mhs::sim {

    void Scheduler::setModel(mhs::core::Model* model)
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

        const std::size_t N = static_cast<std::size_t>(model_->cells.material_id.size());
        step_.T.assign(N, model_->initial_temperature);
        step_.current_time = 0.0;
        step_.time_step = 0;

        Assembler assembler(*model_);

        // Steady: single non-linear solve, then output.
        if (model_->study_type == mhs::core::StudyType::Steady) {
            LinearSystemProvider build_ls = [&](Eigen::Ref<const Eigen::VectorXd> T_in) -> LinearSystem {
                AssembleContext ctx {T_in, step_.current_time};
                auto ops = assembler.assemble(ctx);
                return {std::move(ops.K), std::move(ops.f)};
            };
            Eigen::Map<Eigen::VectorXd> T_map(step_.T.data(), static_cast<Eigen::Index>(N));
            nonlinear_solve(build_ls, T_map, *solver_);
            solution_ = step_.T;
            probe_recorder_.record(step_.current_time, step_.T);
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
        step_.accepted = mhs::core::SolutionHistory(N, 2);
        step_ctrl.rebuild(duration, output_dt);

        step_.accepted.initialize(step_.T, step_.current_time);
        probe_recorder_.record(step_.current_time, step_.T);

        double dt_sug = std::clamp(output_dt / 10.0, min_dt, max_dt);
        double dt = dt_sug;

        Eigen::Map<Eigen::VectorXd> T_map(step_.T.data(), static_cast<Eigen::Index>(N));

        while (step_.current_time < duration - mhs::core::zero_guard) {
            dt = step_ctrl.prepare(dt_sug, step_.current_time, duration);
            if (dt <= 0.0)
                break;
            step_.dt = dt;

            // Re-assembles K, f inside each non-linear iteration; M_diag is
            // frozen at accepted.current() via build_system's BDF stencil logic.
            LinearSystemProvider provider = [&](Eigen::Ref<const Eigen::VectorXd> T_in) -> LinearSystem {
                AssembleContext ctx {T_in, step_.current_time};
                return time_scheme::build_system(
                    time_scheme::IntegratorKind::Bdf1, assembler.assemble(ctx), step_.accepted, step_.dt);
            };
            auto nl = nonlinear_solve(provider, T_map, *solver_);
            if (!nl.converged) {
                MHS_LOG_WARN("Non-linear iteration did not converge at step {}", step_.time_step);
            }

            auto est = time_scheme::estimate_error(step_.accepted, step_.T, dt, /*err_cfg=*/ {});
            dt_sug = std::clamp(dt * est.suggested_factor, min_dt, max_dt);
            const bool accepted_step = (est.error_ratio <= 1.0) || (dt <= min_dt * 1.0001);

            if (accepted_step) {
                step_.current_time += dt;
                step_.time_step++;
                step_.accepted.accept(step_.T, step_.current_time);

                MHS_LOG_DEBUG("Time: {} solved (dt={})", step_.current_time, dt);

                // Free mode: step end may overshoot output time → interpolate.
                auto out = step_ctrl.flush_outputs(step_.current_time);
                for (double t_out : out) {
                    auto T_interp = lerp_T(
                        /* t0 = */ step_.accepted.time_at(1),
                        /* T0 = */ step_.accepted.at(1),
                        /* t1 = */ step_.current_time,
                        /* T1 = */ step_.T, t_out);
                    probe_recorder_.record(t_out, T_interp);
                }
            }
            else {
                dt_sug = dt * 0.5;
                MHS_LOG_DEBUG("Step rejected at t={}, retry dt={}", step_.current_time, dt_sug);
            }
        }

        solution_ = step_.T;
    }

} // namespace mhs::sim
