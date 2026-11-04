#include "solver/time_integration.hpp"

#include "model/constants.hpp"

#include "core/csr_ops.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>

namespace mhs::solver::time_scheme {
    namespace {

        LinearSystem build_bdf1(const Operators& ops, const mhs::solver::SolutionHistory& history, double dt)
        {
            auto matrix = core::add_scaled(ops.K, 1.0 / dt, ops.C);
            auto rhs = ops.f;
            core::spmv(ops.C.view(), history.current(), rhs, 1.0 / dt, 1.0);
            return {std::move(matrix), std::move(rhs)};
        }

        LinearSystem build_bdf2(const Operators& ops, const mhs::solver::SolutionHistory& history, double dt)
        {
            const double ratio = dt / history.previous_dt();
            const double alpha0 = (1.0 + 2.0 * ratio) / (dt * (1.0 + ratio));
            const double alpha1 = -(1.0 + ratio) / dt;
            const double alpha2 = (ratio * ratio) / (dt * (1.0 + ratio));
            auto matrix = core::add_scaled(ops.K, alpha0, ops.C);
            core::Vector<double> combination(ops.f.size());
            const auto current = history.current(), previous = history.at(1);
            for (std::size_t i = 0; i < combination.size(); ++i)
                combination[i] = alpha1 * current[i] + alpha2 * previous[i];
            auto rhs = ops.f;
            core::spmv(ops.C.view(), combination, rhs, -1.0, 1.0);
            return {std::move(matrix), std::move(rhs)};
        }

        double grid_tolerance(double time) noexcept { return mhs::model::zero_guard * std::max(1.0, std::abs(time)); }

    } // namespace

    LinearSystem build_system(
        IntegratorKind kind, const Operators& ops, const mhs::solver::SolutionHistory& history, double dt)
    {
        if (kind == IntegratorKind::Bdf2 && history.size() >= 2)
            return build_bdf2(ops, history, dt);
        return build_bdf1(ops, history, dt);
    }

    ErrorEstimate estimate_error(const mhs::solver::SolutionHistory& accepted, std::span<const double> trial_state,
        double trial_dt, const ErrorControlConfig& config)
    {
        const std::size_t count = static_cast<std::size_t>(trial_state.size());
        if (count == 0)
            return {0.0, 1.0};
        const auto current = accepted.current();
        double relative_error = 0.0;
        const auto previous = accepted.size() >= 2 ? accepted.at(1) : std::span<const double> {};
        const double previous_dt = accepted.size() >= 2 ? accepted.previous_dt() : 0.0;
        const double ratio = previous_dt > mhs::model::zero_guard ? trial_dt / previous_dt : 1.0;
        for (std::size_t i = 0; i < count; ++i) {
            double error = trial_state[i] - current[i];
            if (!previous.empty())
                error -= ratio * (current[i] - previous[i]);
            relative_error = std::max(
                relative_error, std::abs(error) / std::max(std::abs(trial_state[i]), mhs::model::zero_guard));
        }

        const double factor
            = config.safety * std::pow(config.rel_tol / std::max(relative_error, mhs::model::zero_guard), 0.5);
        return {relative_error / config.rel_tol, std::clamp(factor, 0.5, 2.0)};
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

} // namespace mhs::solver::time_scheme
