#include "scheduler.hpp"
#include "assembler/assembler.hpp"
#include "logger/logger.hpp"
#include <cmath>
#include <algorithm>

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

        current_time_ = 0.0;
        current_step_ = 0;

        if (model_->study_type == StudyType::Steady) {
            // Steady-state: single nonlinear solve at t=0
            solve_nonlinear_step();
            solution_ = state_.T;
        }
        else {
            // Transient: time-stepping loop
            double duration = model_->transient_duration;
            if (duration <= 0.0) {
                duration = config_.transient_duration;
            }
            double dt = model_->transient_time_step;
            if (dt <= 0.0) {
                dt = config_.time_step;
            }
            state_.dt = dt;

            while (current_time_ < duration) {
                state_.T_prev = state_.T;
                bool converged = solve_nonlinear_step();
                if (!converged) {
                    MHS_LOG_WARN("Newton iteration did not converge at time step {}", current_step_);
                }

                current_time_ += dt;
                state_.current_time = current_time_;
                current_step_++;
                state_.time_step = current_step_;

                // Store history in ring buffers
                state_.T_history.push_back(state_.T);
                state_.nl_history.push_back(state_.residual);
                state_.dt_history.push_back(dt);

                if ((int)state_.T_history.size() > config_.ring_buffer_capacity) {
                    state_.T_history.pop_front();
                    state_.nl_history.pop_front();
                    state_.dt_history.pop_front();
                }
            }

            solution_ = state_.T;
        }
    }

    bool Scheduler::solve_nonlinear_step()
    {
        assembler::Assembler assembler(*model_);

        double omega = config_.underrelaxation;
        if (omega <= 0.0) {
            omega = 1.0;
        }

        for (int iter = 0; iter < config_.max_newton_iterations; iter++) {
            // Assemble linear system
            auto linear_system = assembler.assemble(state_);

            // Solve A * dT = b (where b = -residual)
            // For Newton: we solve A * dT = -residual
            // But since our assembler already computes b = Q*vol + BC contributions,
            // and A includes diffusion + BC diagonal terms,
            // the equation is A*T = b, and we need T = A^{-1} * b
            // For nonlinear problems, we solve A(T)*dT = b - A(T)*T = -residual
            auto solve_result = solver_->solve(linear_system.A, linear_system.b);

            if (!solve_result.success) {
                MHS_LOG_WARN("Linear solver failed at Newton iteration {}", iter);
                // Try direct solve anyway - use what we got
            }

            // Compute update: dT = solution - T_current
            // But since we're solving A*T = b directly (not residual form),
            // the solution IS the new temperature field
            // Apply under-relaxation: T_new = T_old + omega * (T_direct - T_old)

            double max_update = 0.0;
            for (int i = 0; i < state_.cell_count; i++) {
                double update = solve_result.solution(i) - state_.T[i];
                max_update = std::max(max_update, std::abs(update));
                state_.T[i] += omega * update;
            }

            // Update residual
            for (int i = 0; i < state_.cell_count; i++) {
                state_.residual[i] = linear_system.residual(i);
            }

            // Check convergence
            double max_residual = 0.0;
            for (int i = 0; i < state_.cell_count; i++) {
                max_residual = std::max(max_residual, std::abs(state_.residual[i]));
            }

            MHS_LOG_INFO("Newton iteration {}: max_update={:.6e}, max_residual={:.6e}",
                iter, max_update, max_residual);

            if (max_update < config_.newton_tolerance && max_residual < config_.newton_tolerance) {
                state_.status = ConvergenceStatus::Converged;
                return true;
            }
        }

        state_.status = ConvergenceStatus::Diverged;
        return false;
    }

    const std::vector<double>& Scheduler::solution() const
    {
        return solution_;
    }

} // namespace mhs