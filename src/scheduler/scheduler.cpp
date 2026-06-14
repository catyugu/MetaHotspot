#include "assembler/assembler.hpp"
#include "common/logger.hpp"
#include "nonlinear/nonlinear_solver.hpp"
#include "scheduler.hpp"
#include "time_scheme/time_scheme.hpp"

#include <Eigen/Core>
#include <algorithm>

namespace mhs::sim {

    void Scheduler::setModel(mhs::core::InternalModel* model)
    {
        model_ = model;
        if (model_) {
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

        const int N = static_cast<int>(model_->cells.cell_bcs.size());
        state_.T.assign(N, model_->initial_temperature);
        state_.residual.assign(N, 0.0);
        state_.current_time = 0.0;
        state_.time_step = 0;
        state_.dt = model_->transient_time_step;

        Assembler assembler(*model_);

        // 稳态求解分支
        if (model_->study_type == mhs::core::StudyType::Steady) {
            auto build_ls = [&]() -> LinearSystem {
                auto sops = assembler.assemble_static(state_);
                Eigen::Map<const Eigen::VectorXd> T_map(state_.T.data(), N);
                Eigen::VectorXd res = sops.f_static - sops.K * T_map;
                return {std::move(sops.K), std::move(sops.f_static)};
            };
            nonlinear_solve(build_ls, state_, *solver_);
            solution_ = state_.T;
            probe_recorder_.record(state_.current_time, state_.T);
            return;
        }

        // 瞬态求解分支
        time_scheme::TimeSchemeConfig cfg;
        cfg.kind = time_scheme::TimeSchemeKind::AdaptiveBdf;
        cfg.initial_dt = model_->transient_time_step / 10.0;
        cfg.max_dt = model_->transient_time_step * 10.0;
        cfg.output_dt = model_->transient_time_step;

        state_.history
            = mhs::core::TimeStepBuffer(static_cast<std::size_t>(N), std::max<std::size_t>(1, cfg.max_order + 1));

        auto scheme = time_scheme::create_scheme(cfg);
        scheme->initialize(state_.history, state_);
        probe_recorder_.record(state_.current_time, state_.T);

        const double duration = model_->transient_duration;

        while (state_.current_time < duration) {
            auto step = scheme->select_step(state_.history, state_.current_time, duration);
            state_.dt = step.dt;

            auto build_ls = [&]() -> LinearSystem {
                return scheme->build_system(assembler.assemble_static(state_), assembler.assemble_mass(state_),
                    state_.history, step.order, state_.dt);
            };

            auto result = nonlinear_solve(build_ls, state_, *solver_);
            if (!result.converged) {
                MHS_LOG_WARN("Non-Linear iteration did not converge at time step {}", state_.time_step);
            }

            // 核心变动：评估步长及截断误差的业务完全交由 time_scheme 处理
            auto step_result = scheme->evaluate_step(state_.history, state_.T, state_.dt);

            if (step_result.accepted) {
                state_.current_time += state_.dt;
                state_.time_step++;
                state_.history.push(state_.T, state_.current_time);

                MHS_LOG_INFO("Time: {} solved (dt={})", state_.current_time, state_.dt);

                if (scheme->is_output_boundary(state_.current_time)) {
                    probe_recorder_.record(state_.current_time, state_.T);
                }
            }
            else {
                // 回退状态，时间步未被接受（当前自适应实现中默认不拒绝）
                state_.T = state_.history.latest();
            }
        }

        solution_ = state_.T;
    }

} // namespace mhs::sim