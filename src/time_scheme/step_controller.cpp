#include "time_scheme/step_controller.hpp"

#include <algorithm>
#include <cmath>

namespace mhs::sim::time_scheme {

    namespace {
        constexpr double EPS = 1e-12;

        /// Relative tolerance for grid-point matching.
        inline double gtol(double t) noexcept { return EPS * std::max(1.0, std::abs(t)); }
    }

    StepController::StepController(StepStrategy strategy, double min_dt, double max_dt, double fixed_dt)
        : strategy_(strategy), min_dt_(min_dt), max_dt_(max_dt), fixed_dt_(fixed_dt)
    {
    }

    void StepController::rebuild(double duration, double output_dt)
    {
        if (output_dt > 0.0 && duration > 0.0)
            grid_ = OutputTimeGrid {duration, output_dt};
        else
            grid_ = OutputTimeGrid {};

        next_idx_ = 0;
        last_flushed_t_ = 0.0;
        planted_ = false;
    }

    double StepController::prepare(double dt_suggested, double current_t, double duration)
    {
        const double remaining = duration - current_t;
        if (remaining <= 0.0)
            return 0.0;

        double dt = dt_suggested;

        // Snap the physics-suggested step to the next output grid point.
        // Returns true (and updates `dt`) when such a snap is needed.
        const auto snap_to_next = [&](bool use_planting) {
            if (next_idx_ >= grid_.size())
                return;
            const double t_next = grid_.times()[next_idx_];
            const double tol = gtol(t_next);
            if (t_next <= current_t + tol)
                return;
            if (current_t + dt < t_next - tol)
                return;
            dt = use_planting && !planted_ ? 0.5 * (t_next - current_t) : (t_next - current_t);
            planted_ = true;
        };

        switch (strategy_) {
        case StepStrategy::Free:
            break;
        case StepStrategy::Strict:
            snap_to_next(/*use_planting=*/false);
            break;
        case StepStrategy::Intermediate:
            snap_to_next(/*use_planting=*/true);
            break;
        case StepStrategy::Manual:
            dt = fixed_dt_;
            break;
        }

        dt = std::clamp(dt, min_dt_, max_dt_);
        return std::min(dt, remaining);
    }

    std::vector<double> StepController::flush_outputs(double current_t)
    {
        // No output grid → every unique step end is a flush point.
        if (grid_.empty()) {
            if (current_t > last_flushed_t_ + gtol(last_flushed_t_)) {
                last_flushed_t_ = current_t;
                return {current_t};
            }
            return {};
        }

        // Collect every unconsumed grid point t satisfying
        //   last_flushed_t_ < t ≤ current_t   (within tolerance).
        std::vector<double> crossed;
        while (next_idx_ < grid_.size()) {
            const double t_out = grid_.times()[next_idx_];

            // Skip grid points at or behind the last flush (avoids duplicates
            // including the t₀ = 0 point that was already recorded externally).
            if (t_out <= last_flushed_t_ + gtol(last_flushed_t_)) {
                ++next_idx_;
                continue;
            }

            if (t_out > current_t + gtol(current_t))
                break;

            crossed.push_back(t_out);
            ++next_idx_;
        }

        if (!crossed.empty()) {
            last_flushed_t_ = crossed.back();
            planted_ = false; // New output interval → need a fresh internal point.
        }

        return crossed;
    }

} // namespace mhs::sim::time_scheme
