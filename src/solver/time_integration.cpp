#include "solver/time_integration.hpp"

#include "common/constants.hpp"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace mhs::sim::time_scheme {
    namespace {

        LinearSystem build_bdf1(const Operators& ops, const mhs::core::SolutionHistory& history, double dt)
        {
            const mhs::core::Index count = static_cast<mhs::core::Index>(ops.f.size());
            assert(count <= static_cast<mhs::core::Index>(std::numeric_limits<Eigen::Index>::max()));
            const auto eigen_count = static_cast<Eigen::Index>(count);
            Eigen::Map<const Eigen::VectorXd> previous(history.current().data(), eigen_count);

            Eigen::SparseMatrix<double> matrix = ops.K;
            matrix += (1.0 / dt) * ops.C;
            Eigen::VectorXd rhs = ops.f + (ops.C * previous) / dt;
            return {std::move(matrix), std::move(rhs)};
        }

        LinearSystem build_bdf2(const Operators& ops, const mhs::core::SolutionHistory& history, double dt)
        {
            const mhs::core::Index count = static_cast<mhs::core::Index>(ops.f.size());
            assert(count <= static_cast<mhs::core::Index>(std::numeric_limits<Eigen::Index>::max()));
            const auto eigen_count = static_cast<Eigen::Index>(count);
            const double ratio = dt / history.previous_dt();
            const double alpha0 = (1.0 + 2.0 * ratio) / (dt * (1.0 + ratio));
            const double alpha1 = -(1.0 + ratio) / dt;
            const double alpha2 = (ratio * ratio) / (dt * (1.0 + ratio));

            Eigen::Map<const Eigen::VectorXd> current(history.current().data(), eigen_count);
            Eigen::Map<const Eigen::VectorXd> previous(history.at(1).data(), eigen_count);

            Eigen::SparseMatrix<double> matrix = ops.K;
            matrix += alpha0 * ops.C;
            Eigen::VectorXd rhs = ops.f - ops.C * (alpha1 * current + alpha2 * previous);
            return {std::move(matrix), std::move(rhs)};
        }

        double grid_tolerance(double time) noexcept { return mhs::core::zero_guard * std::max(1.0, std::abs(time)); }

    } // namespace

    LinearSystem build_system(
        IntegratorKind kind, const Operators& ops, const mhs::core::SolutionHistory& history, double dt)
    {
        if (kind == IntegratorKind::Bdf2 && history.size() >= 2)
            return build_bdf2(ops, history, dt);
        return build_bdf1(ops, history, dt);
    }

    ErrorEstimate estimate_error(const mhs::core::SolutionHistory& accepted, std::span<const double> trial_state,
        double trial_dt, const ErrorControlConfig& config)
    {
        const mhs::core::Index count = static_cast<mhs::core::Index>(trial_state.size());
        if (count == 0)
            return {0.0, 1.0};
        assert(count <= static_cast<mhs::core::Index>(std::numeric_limits<Eigen::Index>::max()));
        const auto eigen_count = static_cast<Eigen::Index>(count);

        Eigen::Map<const Eigen::VectorXd> trial(trial_state.data(), eigen_count);
        Eigen::Map<const Eigen::VectorXd> current(accepted.current().data(), eigen_count);

        Eigen::VectorXd error_vector;
        if (accepted.size() >= 2) {
            Eigen::Map<const Eigen::VectorXd> previous(accepted.at(1).data(), eigen_count);
            const double previous_dt = accepted.previous_dt();
            const double ratio = previous_dt > mhs::core::zero_guard ? trial_dt / previous_dt : 1.0;
            error_vector = ((trial - current) - ratio * (current - previous)).cwiseAbs();
        }
        else {
            error_vector = (trial - current).cwiseAbs();
        }

        const double error = error_vector.cwiseQuotient(trial.cwiseAbs().cwiseMax(1.0)).maxCoeff();
        const double factor = config.safety * std::pow(config.abs_tol / std::max(error, mhs::core::zero_guard), 0.5);
        return {error / config.abs_tol, std::clamp(factor, 0.5, 2.0)};
    }

    StepController::StepController(
        StepStrategy strategy, double min_dt, double max_dt, double duration, double output_interval, double fixed_dt)
        : strategy_(strategy)
        , duration_(duration)
        , output_interval_(output_interval)
        , min_dt_(min_dt)
        , max_dt_(max_dt)
        , fixed_dt_(fixed_dt)
        , next_output_(output_interval)
    {
    }

    double StepController::prepare(double dt_suggested, double current_t)
    {
        const double remaining = duration_ - current_t;
        if (remaining <= 0.0)
            return 0.0;

        while (output_interval_ > 0.0 && next_output_ <= current_t + grid_tolerance(current_t))
            next_output_ += output_interval_;

        const double proposed = strategy_ == StepStrategy::Fixed ? fixed_dt_ : dt_suggested;
        const double bounded = std::clamp(proposed, min_dt_, max_dt_);

        double next_boundary = duration_;
        if (output_interval_ > 0.0 && next_output_ < duration_ - grid_tolerance(duration_))
            next_boundary = next_output_;

        // Output states must be states actually solved by the integrator. A
        // boundary may therefore shorten one step below min_dt; min_dt remains
        // the lower bound for unconstrained step-size adaptation.
        const double to_boundary = next_boundary - current_t;
        return std::min(bounded, std::min(remaining, to_boundary));
    }

    bool StepController::output_due(double current_t)
    {
        const bool at_final = current_t >= duration_ - grid_tolerance(duration_);
        const bool at_grid = output_interval_ > 0.0 && current_t >= next_output_ - grid_tolerance(next_output_);
        if (output_interval_ > 0.0 && !at_grid && !at_final)
            return false;

        if (at_grid)
            next_output_ += output_interval_;
        return true;
    }

} // namespace mhs::sim::time_scheme
