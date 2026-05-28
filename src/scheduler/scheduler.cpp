#include "scheduler.hpp"
#include "assembler/assembler.hpp"
#include "solver/solver.hpp"
#include "logger/logger.hpp"

namespace mhs {

namespace {

double compute_residual_norm(const Eigen::VectorXd& r)
{
    return r.norm();
}

} // namespace

void Scheduler::setModel(std::unique_ptr<model::InternalModel> model)
{
    model_ = std::move(model);

    // Initialize state
    if (model_) {
        solution_.resize(model_->mesh.cell_count, model_->initial_temperature);
        state_.T = solution_;
        state_.T_prev = solution_;
        state_.cell_count = model_->mesh.cell_count;
        state_.current_time = 0.0;
        state_.status = ConvergenceStatus::Running;
    }
}

void Scheduler::setSolver(std::unique_ptr<Solver> solver)
{
    solver_ = std::move(solver);
}

void Scheduler::run()
{
    if (!model_ || !solver_) {
        MHS_LOG_ERROR("Model or solver not set");
    }

    Assembler assembler;

    if (model_->study_type == StudyType::Steady) {
        // Steady-state simulation
        int max_iterations = 50;
        double tolerance = 1e-6;

        for (int iter = 0; iter < max_iterations; ++iter) {
            auto result = assembler.assemble(*model_, solution_, 0.0);
            auto solve_result = solver_->solve(result.A, result.b);

            if (!solve_result.success) {
                MHS_LOG_ERROR("Solver failed at iteration {}", iter);
                state_.status = ConvergenceStatus::Diverged;
                return;
            }

            // Update solution with under-relaxation
            double omega = 0.8; // under-relaxation factor
            for (size_t i = 0; i < solution_.size(); ++i) {
                solution_[i] = (1.0 - omega) * solution_[i] + omega * solve_result.solution(i);
            }

            // Check convergence using residual norm
            double residual_norm = compute_residual_norm(solve_result.solution - Eigen::Map<Eigen::VectorXd>(solution_.data(), solution_.size()));

            if (residual_norm < tolerance) {
                state_.status = ConvergenceStatus::Converged;
                MHS_LOG_INFO("Converged at iteration {}", iter);
                break;
            }

            if (iter == max_iterations - 1) {
                state_.status = ConvergenceStatus::Diverged;
                MHS_LOG_WARN("Did not converge after {} iterations", max_iterations);
            }
        }
    } else {
        // Transient simulation
        double t = 0.0;
        double dt = model_->transient_time_step;
        int step = 0;

        while (t < model_->transient_duration) {
            solveNonlinear(t);

            state_.T_prev = state_.T;
            state_.current_time = t;
            state_.time_step = step;
            ++step;
            t += dt;
        }
    }
}

void Scheduler::stepTime(double dt)
{
    solveNonlinear(state_.current_time);
    state_.T_prev = state_.T;
    state_.current_time += dt;
    state_.time_step++;
}

void Scheduler::solveNonlinear(double t)
{
    Assembler assembler;

    auto result = assembler.assemble(*model_, solution_, t);
    auto solve_result = solver_->solve(result.A, result.b);

    if (solve_result.success) {
        for (size_t i = 0; i < solution_.size(); ++i) {
            solution_[i] = solve_result.solution(i);
        }
        state_.T = solution_;
    }
}

const std::vector<double>& Scheduler::solution() const
{
    return solution_;
}

} // namespace mhs