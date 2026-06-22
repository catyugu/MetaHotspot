#include "assembler/assembler.hpp"
#include "common/logger.hpp"
#include "nonlinear/nonlinear_solver.hpp"
#include "preprocessor/fluid_preprocessor.hpp"
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
            MHS_FATAL("Scheduler: model or solver not set");
        }

        const int N = static_cast<int>(model_->cells.cell_bcs.size());
        state_.T.assign(N, model_->initial_temperature);
        state_.current_time = 0.0;
        state_.time_step = 0;
        state_.dt = model_->transient_time_step;

        Assembler assembler(*model_);
        mhs::sim::solveFluidFlow(*model_);
        // 稳态求解分支
        if (model_->study_type == mhs::core::StudyType::Steady) {
            // 流体压力求解(若模型含流体): pressure → flow_axes 在 T 求解前计算一次

            LinearSystemProvider build_ls = [&](const mhs::core::GlobalState& s) -> LinearSystem {
                auto ops = assembler.assemble(s);
                return {std::move(ops.K), std::move(ops.f)};
            };
            nonlinear_solve(build_ls, state_, *solver_);
            solution_ = state_.T;
            probe_recorder_.record(state_.current_time, state_.T);
            return;
        }

        // 瞬态求解分支
        time_scheme::TimeSchemeConfig cfg;
        cfg.max_order = 1;
        cfg.kind = time_scheme::TimeSchemeKind::AdaptiveBdf;
        cfg.initial_dt = model_->transient_time_step / 10.0;
        cfg.max_dt = model_->transient_time_step * 10.0;
        cfg.output_dt = model_->transient_time_step;

        state_.accepted
            = mhs::core::SolutionHistory(static_cast<std::size_t>(N), std::max<std::size_t>(1, cfg.max_order + 1));

        auto scheme = time_scheme::create_scheme(cfg);
        scheme->initialize(state_.accepted, state_);
        probe_recorder_.record(state_.current_time, state_.T);

        const double duration = model_->transient_duration;

        while (state_.current_time < duration) {
            auto step = scheme->select_step(state_.accepted, state_.current_time, duration);
            state_.dt = step.dt;

            LinearSystemProvider build_ls = [&](const mhs::core::GlobalState& s) -> LinearSystem {
                return scheme->build_system(assembler.assemble(s), s.accepted, step.order, s.dt);
            };

            auto result = nonlinear_solve(build_ls, state_, *solver_);
            if (!result.converged) {
                MHS_LOG_WARN("Non-Linear iteration did not converge at time step {}", state_.time_step);
            }

            // 核心变动：评估步长及截断误差的业务完全交由 time_scheme 处理
            auto step_result = scheme->evaluate_step(state_.accepted, state_.T, state_.dt);

            if (step_result.accepted) {
                state_.current_time += state_.dt;
                state_.time_step++;
                state_.accepted.accept(state_.T, state_.current_time);

                MHS_LOG_DEBUG("Time: {} solved (dt={})", state_.current_time, state_.dt);

                if (scheme->is_output_boundary(state_.current_time)) {
                    probe_recorder_.record(state_.current_time, state_.T);
                }
            }
            else {
                // 回退状态，时间步未被接受（当前自适应实现中默认不拒绝）
                state_.T = state_.accepted.current();
            }
        }

        solution_ = state_.T;
    }

} // namespace mhs::sim