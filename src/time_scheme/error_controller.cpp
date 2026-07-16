#include "data/tolerance_config.hpp"
#include "data/types.hpp"
#include "time_scheme/error_controller.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace mhs::sim::time_scheme {

    ErrorEstimate estimate_error(const mhs::core::SolutionHistory& accepted, const std::vector<double>& trial_T,
        double trial_dt, const ErrorControlConfig& cfg)
    {
        const mhs::Index N = static_cast<mhs::Index>(trial_T.size());
        if (N == 0)
            return {0.0, 1.0};
        assert(N <= static_cast<mhs::Index>(std::numeric_limits<Eigen::Index>::max()));
        const auto eigen_N = static_cast<Eigen::Index>(N);

        Eigen::Map<const Eigen::VectorXd> T_curr(trial_T.data(), eigen_N);
        Eigen::Map<const Eigen::VectorXd> T_prev(accepted.current().data(), eigen_N);

        // Local truncation error estimate.
        Eigen::VectorXd err_vec;
        if (accepted.size() >= 2) {
            Eigen::Map<const Eigen::VectorXd> T_prev2(accepted.at(1).data(), N);
            const double dt_prev = accepted.previous_dt();
            const double ratio = (dt_prev > mhs::core::zero_guard) ? (trial_dt / dt_prev) : 1.0;
            err_vec = ((T_curr - T_prev) - ratio * (T_prev - T_prev2)).cwiseAbs();
        }
        else {
            err_vec = (T_curr - T_prev).cwiseAbs();
        }

        // Normalise by max(|T|, 1) and take infinity norm.
        const Eigen::VectorXd max_T = T_curr.cwiseAbs().cwiseMax(1.0);
        const double err = err_vec.cwiseQuotient(max_T).maxCoeff();

        // PI-like factor: safety * (tol / err)^(1/p).
        const double fac = cfg.safety * std::pow(cfg.abs_tol / std::max(err, mhs::core::zero_guard), 0.5);

        // Clamp to a reasonable range so a single bad step can't crash the integrator.
        const double suggested_factor = std::clamp(fac, 0.5, 2.0);

        return {err / cfg.abs_tol, suggested_factor};
    }

} // namespace mhs::sim::time_scheme
