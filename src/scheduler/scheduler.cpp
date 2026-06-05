#include "common/logger.hpp"
#include "nonlinear/nonlinear_solver.hpp"
#include "scheduler.hpp"

namespace mhs::sim {

    void Scheduler::setSolver(std::unique_ptr<LinearSolver> solver) { solver_ = std::move(solver); }

    void Scheduler::run()
    {
        if (!model_ || !solver_) {
            MHS_LOG_ERROR("Scheduler: model or solver not set");
            return;
        }

        // Initialize state
        int N = model_->cells.cell_count;
        state_.cell_count = N;
        state_.T.resize(N, model_->initial_temperature);
        state_.T_prev.resize(N, model_->initial_temperature);
        state_.residual.resize(N, 0.0);
        state_.current_time = 0.0;
        state_.time_step = 0;
        state_.dt = model_->transient_time_step;

        if (model_->study_type == mhs::core::StudyType::Steady) {
            nonlinear_solve(*model_, state_, *solver_);
            solution_ = state_.T;
            if (callback_) {
                callback_(state_.current_time, state_.time_step, state_.T);
            }
        }
        else {
            const double duration = model_->transient_duration;
            const double dt = model_->transient_time_step;
            state_.dt = dt;
            if (callback_) {
                callback_(state_.current_time, state_.time_step, state_.T);
            }

            while (state_.current_time < duration) {
                state_.T_prev = state_.T;
                auto result = nonlinear_solve(*model_, state_, *solver_);
                if (!result.converged) {
                    MHS_LOG_WARN("Non-Linear iteration did not converge at time step {}", state_.time_step);
                }

                state_.current_time += dt;
                state_.time_step++;
                MHS_LOG_INFO("Time: {} solved", state_.current_time);
                if (callback_) {
                    callback_(state_.current_time, state_.time_step, state_.T);
                }
            }

            solution_ = state_.T;
        }
    }

    const std::vector<double>& Scheduler::solution() const { return solution_; }

} // namespace mhs::sim