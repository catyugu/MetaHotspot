#include "scheduler.hpp"
#include "logger/logger.hpp"
#include "nonlinear/nonlinear_solver.hpp"

namespace mhs {

    void Scheduler::setSolver(std::unique_ptr<Solver> solver)
    {
        solver_ = std::move(solver);
    }

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
        state_.dt = config_.time_step;

        if (model_->study_type == StudyType::Steady) {
            nonlinear::solve(*model_, state_, *solver_,
                config_.underrelaxation, config_.max_newton_iterations, config_.newton_tolerance);
            solution_ = state_.T;
        }
        else {
            double duration = model_->transient_duration > 0.0 ? model_->transient_duration : config_.transient_duration;
            double dt = model_->transient_time_step > 0.0 ? model_->transient_time_step : config_.time_step;
            state_.dt = dt;

            while (state_.current_time < duration) {
                state_.T_prev = state_.T;
                auto result = nonlinear::solve(*model_, state_, *solver_,
                    config_.underrelaxation, config_.max_newton_iterations, config_.newton_tolerance);
                if (!result.converged) {
                    MHS_LOG_WARN("Newton iteration did not converge at time step {}", state_.time_step);
                }

                state_.current_time += dt;
                state_.time_step++;
            }

            solution_ = state_.T;
        }
    }

    const std::vector<double>& Scheduler::solution() const
    {
        return solution_;
    }

} // namespace mhs