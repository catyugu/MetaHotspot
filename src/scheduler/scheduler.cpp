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

        time_scheme::TimeSchemeConfig time_scheme_config;

        // Use adaptive BDF with fine-grained startup step.
        // The scheme owns all output-time alignment internally; the scheduler
        // just records probes whenever the scheme says we're at a boundary.
        time_scheme_config.kind = time_scheme::TimeSchemeKind::AdaptiveBdf;
        time_scheme_config.initial_dt = model_->transient_time_step / 10.0;
        time_scheme_config.max_dt = model_->transient_time_step;
        time_scheme_config.output_dt = model_->transient_time_step;

        std::size_t history_cap = std::max<std::size_t>(1, time_scheme_config.max_order + 1);
        state_.history = mhs::core::TimeStepBuffer(static_cast<std::size_t>(N), history_cap);

        Assembler assembler(*model_);

        if (model_->study_type == mhs::core::StudyType::Steady) {
            auto build_ls = [&]() -> LinearSystem {
                auto sops = assembler.assemble_static(state_);
                Eigen::Map<const Eigen::VectorXd> T_map(state_.T.data(), static_cast<Eigen::Index>(state_.T.size()));
                return {sops.K, sops.f_static, sops.f_static - sops.K * T_map};
            };
            nonlinear_solve(build_ls, state_, *solver_);
            solution_ = state_.T;
            probe_recorder_.record(state_.current_time, state_.T);
            return;
        }

        // --- Transient main loop driven by the time scheme ---
        auto scheme = time_scheme::create_scheme(time_scheme_config);

        const double duration = model_->transient_duration;

        // 1) Initialize history with the initial state at t=0.
        scheme->initialize(state_.history, state_);

        // 2) Record probe at t=0.
        probe_recorder_.record(state_.current_time, state_.T);

        while (state_.current_time < duration) {
            // The scheme returns a dt that already accounts for output-time
            // alignment and remaining-duration clamping — apply it directly.
            auto step = scheme->select_step(state_.history, state_.current_time, duration);
            state_.dt = step.dt;

            auto build_ls = [&]() -> LinearSystem {
                auto sops = assembler.assemble_static(state_);
                auto mops = assembler.assemble_mass(state_);
                return scheme->build_system(sops, mops, state_.history, step.order, state_.dt);
            };
            auto result = nonlinear_solve(build_ls, state_, *solver_);
            if (!result.converged) {
                MHS_LOG_WARN("Non-Linear iteration did not converge at time step {}", state_.time_step);
            }

            std::vector<double> error_estimate(static_cast<std::size_t>(N));
            const auto& T_prev = state_.history.latest();
            for (int i = 0; i < N; ++i) {
                double denom = std::max(1.0, std::abs(T_prev[i]));
                error_estimate[static_cast<std::size_t>(i)] = std::abs(state_.T[i] - T_prev[i]) / denom;
            }
            scheme->accept_or_reject(state_.history, state_.T, error_estimate);

            // Advance clock, push to history.
            double t_new = state_.current_time + state_.dt;
            state_.current_time = t_new;
            state_.time_step++;
            state_.history.push(state_.T, t_new);

            MHS_LOG_INFO("Time: {} solved (dt={})", t_new, state_.dt);

            // Let the scheme decide whether this step lands on an output
            // boundary (the scheme owns output_dt alignment internally).
            if (scheme->is_output_boundary(t_new)) {
                probe_recorder_.record(t_new, state_.T);
            }
        }

        solution_ = state_.T;
    }

    const std::vector<double>& Scheduler::solution() const { return solution_; }

} // namespace mhs::sim
