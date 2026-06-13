#include "common/logger.hpp"
#include "nonlinear/nonlinear_solver.hpp"
#include "scheduler.hpp"
#include "time_scheme/time_scheme.hpp"

#include <algorithm>
#include <cmath>

namespace mhs::sim {

    void Scheduler::setModel(mhs::core::InternalModel* model)
    {
        model_ = model;
        // 同步初始化探针记录器：空观察点时 recorder 内部 record() 是 no-op。
        if (model) {
            probe_recorder_.initialize(*model);
        }
    }

    void Scheduler::setSolver(std::unique_ptr<LinearSolver> solver) { solver_ = std::move(solver); }

    void Scheduler::setTimeSchemeConfig(time_scheme::TimeSchemeConfig cfg) { scheme_cfg_ = std::move(cfg); }

    void Scheduler::run()
    {
        if (!model_ || !solver_) {
            MHS_LOG_ERROR("Scheduler: model or solver not set");
            return;
        }

        // Initialize state
        int N = static_cast<int>(model_->cells.cell_bcs.size());
        state_.T.resize(N, model_->initial_temperature);
        state_.T_prev.resize(N, model_->initial_temperature);
        state_.residual.resize(N, 0.0);
        state_.current_time = 0.0;
        state_.time_step    = 0;
        state_.dt           = model_->transient_time_step;
        state_.output_step  = 0;

        // Resolve the effective TimeSchemeConfig: explicit setTimeSchemeConfig
        // call wins; otherwise pull from InternalModel (which the preprocessor
        // populated from IOStructure's <TimeScheme> block, defaulting to Bdf1
        // when the XML omits the block).
        time_scheme::TimeSchemeConfig effective_cfg = scheme_cfg_;
        // "Was the user explicit?"  We detect by checking initial_dt: the
        // struct's default is 1.0; if it equals the default and the model has
        // a different initial_dt, prefer the model's value.  (This is a
        // pragmatic sentinel; a future API could split out a has-been-set bit.)
        if (effective_cfg.initial_dt == 1.0 && model_->ts_initial_dt > 0.0
            && std::abs(effective_cfg.initial_dt - model_->ts_initial_dt) > 1e-12) {
            effective_cfg.kind        = model_->time_scheme_kind;
            effective_cfg.initial_dt = model_->ts_initial_dt;
            effective_cfg.min_dt     = model_->ts_min_dt;
            effective_cfg.max_dt     = model_->ts_max_dt;
            effective_cfg.abs_tol    = model_->ts_abs_tol;
            effective_cfg.rel_tol    = model_->ts_rel_tol;
            effective_cfg.max_order  = model_->ts_max_order;
            effective_cfg.output_dt  = model_->ts_output_dt;
        }

        // TimeStepBuffer capacity matches the time scheme's max_order so BDFk
        // windows fit.  For BDF1 / AdaptiveBdf(max_order=1) this is 2.
        std::size_t history_cap = std::max<std::size_t>(1, effective_cfg.max_order + 1);
        state_.history = mhs::core::TimeStepBuffer(static_cast<std::size_t>(N), history_cap);

        if (model_->study_type == mhs::core::StudyType::Steady) {
            // Steady: no time loop.  Delegate to legacy nonlinear_solve.
            nonlinear_solve(*model_, state_, *solver_);
            solution_ = state_.T;
            probe_recorder_.record(state_.current_time, state_.T);
            return;
        }

        // --- Transient main loop driven by the time scheme ---
        auto scheme = time_scheme::create_scheme(effective_cfg);

        const double duration = model_->transient_duration;
        const double dt_init  = model_->transient_time_step;

        // 1) Initialize history with the initial state at t=0.
        state_.history.reset(state_.T);
        scheme->initialize(state_.history, state_);

        // 2) Record probe at t=0.
        probe_recorder_.record(state_.current_time, state_.T);

        // 3) Time loop.
        while (state_.current_time < duration) {
            auto step = scheme->select_step(state_.history, state_.current_time);

            // Clamp dt to remaining duration.
            double dt = step.dt;
            if (dt <= 0.0) dt = dt_init;
            double remaining = duration - state_.current_time;
            if (dt > remaining)
                dt = remaining;

            state_.dt   = dt;
            std::size_t order = step.order;

            // For slice 3 we keep the legacy nonlinear_solve() signature; it
            // builds the LinearSystem internally from state.dt (transient BDF1
            // glue from slice 1).  This preserves the BDF1 numerics on existing
            // cases.  Slice 4/6 will route the build through scheme->build_system.
            (void)order; // reserved for future slices
            auto result = nonlinear_solve(*model_, state_, *solver_);
            if (!result.converged) {
                MHS_LOG_WARN("Non-Linear iteration did not converge at time step {}", state_.time_step);
            }

            // Advance clock, push to history.
            state_.current_time += dt;
            state_.time_step++;
            state_.T_prev = state_.T;
            state_.history.push(state_.T, state_.current_time);
            MHS_LOG_INFO("Time: {} solved (dt={})", state_.current_time, dt);
            probe_recorder_.record(state_.current_time, state_.T);
        }

        solution_ = state_.T;
    }

    const std::vector<double>& Scheduler::solution() const { return solution_; }

} // namespace mhs::sim
