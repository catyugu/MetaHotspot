#include "assembler/assembler.hpp"
#include "common/logger.hpp"
#include "nonlinear/nonlinear_solver.hpp"
#include "preprocessor/fluid_preprocessor.hpp"
#include "scheduler/scheduler.hpp"
#include "time_scheme/error_controller.hpp"

#include <Eigen/Core>
#include <algorithm>

namespace mhs::sim {

    // ======================================================================
    // Setup helpers
    // ======================================================================

    void Scheduler::setModel(mhs::core::InternalModel* model)
    {
        model_ = model;
        if (model_)
            probe_recorder_.initialize(*model);
    }

    void Scheduler::setSolver(std::unique_ptr<LinearSolver> solver) { solver_ = std::move(solver); }

    void Scheduler::setStepStrategy(time_scheme::StepStrategy strategy)
    {
        step_ctrl_ = time_scheme::StepController {
            strategy, model_ ? model_->transient_time_step : 1.0, min_dt_, max_dt_, fixed_dt_};
    }

    void Scheduler::setStepBounds(double min_dt, double max_dt)
    {
        min_dt_ = min_dt;
        max_dt_ = max_dt;
    }

    // ======================================================================
    // Interpolation helper (Free-mode output)
    // ======================================================================

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

    // ======================================================================
    // Main solve
    // ======================================================================

    void Scheduler::run()
    {
        if (!model_ || !solver_) {
            MHS_FATAL("Scheduler: model or solver not set");
        }

        const std::size_t N = static_cast<std::size_t>(model_->cells.cell_bcs.size());
        state_.T.assign(N, model_->initial_temperature);
        state_.current_time = 0.0;
        state_.time_step = 0;
        state_.dt = model_->transient_time_step;

        Assembler assembler(*model_);
        solveFluidFlow(*model_);

        // ================================================================
        // Steady — single non-linear solve, then output
        // ================================================================
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

        // ================================================================
        // Transient
        // ================================================================
        const double duration = model_->transient_duration;
        const double output_dt = model_->transient_time_step;

        // Reconstruct step controller with the actual output_dt from model
        // (in case setStepStrategy was called before setModel or the model
        // config changed between setup and run).
        step_ctrl_ = time_scheme::StepController {step_ctrl_.strategy(), output_dt, min_dt_, max_dt_, fixed_dt_};

        // Solution-history ring buffer: capacity 2 (one for current, one for
        // the previous step — enough for BDF2's startup sequence).
        state_.accepted = mhs::core::SolutionHistory(N, 2);

        // Initialise step controller with the full output-time grid.
        step_ctrl_.rebuild(duration);

        // Record t=0 probe and initialise BDF history.
        state_.accepted.initialize(state_.T, state_.current_time);
        probe_recorder_.record(state_.current_time, state_.T);

        // Suggested dt starts conservatively at 1/10 of the output interval.
        double dt_sug = std::clamp(output_dt / 10.0, min_dt_, max_dt_);
        double dt = dt_sug;

        while (state_.current_time < duration - 1e-14) {

            // ---- 1. Strategy adjustment  --------------------------------
            dt = step_ctrl_.prepare(dt_sug, state_.current_time, duration);
            if (dt <= 0.0)
                break;
            state_.dt = dt;

            // ---- 2. Non-linear solve with time-integrated system --------
            // The provider re-assembles K, f inside each non-linear
            // iteration; M_diag is frozen at accepted.current() via
            // build_system's internal BDF stencil logic.
            LinearSystemProvider provider = [&](const mhs::core::GlobalState& s) -> LinearSystem {
                return time_scheme::build_system(integrator_, assembler.assemble(s), s.accepted, s.dt);
            };
            auto nl = nonlinear_solve(provider, state_, *solver_);
            if (!nl.converged) {
                MHS_LOG_WARN("Non-linear iteration did not converge at step {}", state_.time_step);
            }

            // ---- 3. Error control (skipped in Manual mode) --------------
            bool accepted = true;
            if (step_ctrl_.strategy() != time_scheme::StepStrategy::Manual) {
                auto est = time_scheme::estimate_error(state_.accepted, state_.T, dt, err_cfg_);

                dt_sug = std::clamp(dt * est.suggested_factor, min_dt_, max_dt_);
                accepted = (est.error_ratio <= 1.0) || (dt <= min_dt_ * 1.0001);
            }

            // ---- 4. Accept / reject ------------------------------------
            if (accepted) {
                state_.current_time += dt;
                state_.time_step++;
                state_.accepted.accept(state_.T, state_.current_time);

                MHS_LOG_DEBUG("Time: {} solved (dt={})", state_.current_time, dt);

                // ---- 5. Output -----------------------------------------
                auto out = step_ctrl_.flush_outputs(state_.current_time);
                for (double t_out : out) {
                    if (step_ctrl_.strategy() == time_scheme::StepStrategy::Free) {
                        // Step end overshoots output time → interpolate.
                        auto T_interp = lerp_T(
                            /* t0 = */ state_.accepted.time_at(1),
                            /* T0 = */ state_.accepted.at(1),
                            /* t1 = */ state_.current_time,
                            /* T1 = */ state_.T, t_out);
                        probe_recorder_.record(t_out, T_interp);
                    }
                    else {
                        probe_recorder_.record(t_out, state_.T);
                    }
                }
            }
            else {
                // Step rejected: shrink and retry.
                dt_sug = dt * 0.5;
                MHS_LOG_DEBUG("Step rejected at t={}, retry dt={}", state_.current_time, dt_sug);
            }
        }

        solution_ = state_.T;
    }

} // namespace mhs::sim
