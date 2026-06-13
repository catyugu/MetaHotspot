#include "common/logger.hpp"
#include "nonlinear/nonlinear_solver.hpp"
#include "scheduler.hpp"
#include "time_scheme/time_scheme.hpp"

#include <Eigen/Core>

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
        state_.residual.resize(N, 0.0);
        state_.current_time = 0.0;
        state_.time_step = 0;
        state_.dt = model_->transient_time_step;
        state_.output_step = 0;

        // Resolve the effective TimeSchemeConfig: explicit setTimeSchemeConfig
        // call wins; otherwise pull from InternalModel.
        time_scheme::TimeSchemeConfig effective_cfg = scheme_cfg_;
        if (effective_cfg.initial_dt == 1.0 && model_->ts_initial_dt > 0.0
            && std::abs(effective_cfg.initial_dt - model_->ts_initial_dt) > 1e-12) {
            effective_cfg.kind = model_->time_scheme_kind;
            effective_cfg.initial_dt = model_->ts_initial_dt;
            effective_cfg.min_dt = model_->ts_min_dt;
            effective_cfg.max_dt = model_->ts_max_dt;
            effective_cfg.abs_tol = model_->ts_abs_tol;
            effective_cfg.rel_tol = model_->ts_rel_tol;
            effective_cfg.max_order = model_->ts_max_order;
            effective_cfg.output_dt = model_->ts_output_dt;
        }

        std::size_t history_cap = std::max<std::size_t>(1, effective_cfg.max_order + 1);
        state_.history = mhs::core::TimeStepBuffer(static_cast<std::size_t>(N), history_cap);

        Assembler assembler(*model_);

        if (model_->study_type == mhs::core::StudyType::Steady) {
            auto build_ls = [&]() -> LinearSystem {
                auto sops = assembler.assemble_static(state_);
                Eigen::VectorXd T_vec(static_cast<Eigen::Index>(state_.T.size()));
                for (int i = 0; i < static_cast<int>(state_.T.size()); ++i)
                    T_vec(i) = state_.T[i];
                return {sops.K, sops.f_static, sops.f_static - sops.K * T_vec};
            };
            nonlinear_solve(build_ls, state_, *solver_);
            solution_ = state_.T;
            probe_recorder_.record(state_.current_time, state_.T);
            return;
        }

        // --- Transient main loop driven by the time scheme ---
        auto scheme = time_scheme::create_scheme(effective_cfg);

        const double duration = model_->transient_duration;
        const double dt_init = model_->transient_time_step;
        const double output_dt = effective_cfg.output_dt;

        // 1) Initialize history with the initial state at t=0.
        state_.history.reset(state_.T);
        scheme->initialize(state_.history, state_);

        // 2) Record probe at t=0.
        probe_recorder_.record(state_.current_time, state_.T);

        // 3) Track the *previous* T snapshot for linear interpolation when an
        //    output time falls strictly inside a step.
        double t_last = state_.current_time;
        std::vector<double> T_at_t_last = state_.T;

        while (state_.current_time < duration) {
            auto step = scheme->select_step(state_.history, state_.current_time);

            double dt = step.dt;
            if (dt <= 0.0)
                dt = dt_init;
            double remaining = duration - state_.current_time;

            // 4) Output-time alignment: clamp dt so the next internal step
            //    boundary lands exactly on t_out = output_step * output_dt.
            if (output_dt > 0.0) {
                double t_next_out = (state_.output_step + 1) * output_dt;
                if (t_next_out > state_.current_time && t_next_out < state_.current_time + dt) {
                    dt = t_next_out - state_.current_time;
                }
            }
            if (dt > remaining)
                dt = remaining;

            state_.dt = dt;
            std::size_t order = step.order;

            auto build_ls = [&]() -> LinearSystem {
                auto sops = assembler.assemble_static(state_);
                auto mops = assembler.assemble_mass(state_);
                return scheme->build_system(sops, mops, state_.history, order, dt);
            };
            auto result = nonlinear_solve(build_ls, state_, *solver_);
            if (!result.converged) {
                MHS_LOG_WARN("Non-Linear iteration did not converge at time step {}", state_.time_step);
            }

            // Advance clock, push to history.
            double t_new = state_.current_time + dt;
            state_.current_time = t_new;
            state_.time_step++;
            state_.history.push(state_.T, t_new);

            MHS_LOG_INFO("Time: {} solved (dt={})", t_new, dt);
            probe_recorder_.record(t_new, state_.T);

            if (output_dt > 0.0) {
                double t_next_out = (state_.output_step + 1) * output_dt;
                if (std::abs(t_new - t_next_out) <= 1e-9 * std::max(1.0, t_new)) {
                    state_.output_step++;
                }
            }

            (void)t_last;
            (void)T_at_t_last;
        }

        solution_ = state_.T;
    }

    const std::vector<double>& Scheduler::solution() const { return solution_; }

} // namespace mhs::sim
