#include "step_controller.hpp"

#include <algorithm>
#include <cmath>

namespace mhs::sim::time_scheme {

    double StepController::error_norm(const std::vector<double>& e)
    {
        double mx = 0.0;
        for (double v : e)
            mx = std::max(mx, std::abs(v));
        return mx;
    }

    StepController::StepResult StepController::decide(double current_dt, std::size_t order, double err) const
    {
        // tol = abs_tol + rel_tol * ||T||∞  — but for simplicity we use
        // abs_tol only (rel_tol is reserved for mixed scaling).
        double tol = cfg_.abs_tol;

        if (err <= tol) {
            // Accept: grow dt by (tol/err)^{1/(order+1)} × safety
            double fac = cfg_.safety * std::pow(tol / std::max(err, 1e-30), 1.0 / static_cast<double>(order + 1));
            // Soft-constraint: 0.5 ≤ fac ≤ 2.0
            fac = std::clamp(fac, 0.5, 2.0);
            double next_dt = current_dt * fac;
            next_dt = std::clamp(next_dt, cfg_.min_dt, cfg_.max_dt);
            return {true, next_dt};
        }
        else {
            // Reject: shrink dt by (tol/err)^{1/order} × safety
            double fac = cfg_.safety * std::pow(tol / err, 1.0 / static_cast<double>(order));
            fac = std::clamp(fac, 0.5, 2.0);
            double next_dt = current_dt * fac;
            next_dt = std::clamp(next_dt, cfg_.min_dt, cfg_.max_dt);
            return {false, next_dt};
        }
    }

} // namespace mhs::sim::time_scheme
