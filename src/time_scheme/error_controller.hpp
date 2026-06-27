#pragma once

#include "data/solution_history.hpp"
#include <vector>

namespace mhs::sim::time_scheme {

    /// Configuration for the adaptive error controller (PI-like).
    struct ErrorControlConfig {
        double abs_tol = 1e-4; ///< Absolute tolerance for LTE.
        double safety = 0.9; ///< Safety factor (0 < safety < 1 → conservative steps).
    };

    /// Result of a single error-estimation call.
    struct ErrorEstimate {
        double error_ratio = 0.0; ///< ‖LTE‖ / abs_tol.  ≤ 1 → step accepted.
        double suggested_factor = 1.0; ///< Multiplier for the next step (PI controller).
    };

    /// Compute the local truncation error and a PI-controller step-size suggestion.
    ///
    /// \param accepted  History of previously accepted steps (must have size ≥ 1).
    /// \param trial_T   Temperature vector computed by taking the just-finished step.
    /// \param trial_dt  The dt that was used for this trial step.
    /// \param cfg       Tolerance and clamping parameters.
    ///
    /// The caller is responsible for clamping the suggested dt:
    ///   next_dt = clamp(trial_dt * est.suggested_factor, cfg.min_dt, cfg.max_dt)
    /// and for applying the "recovery boost" when trial_dt was constrained:
    ///   if trial_dt < previous_suggestion && est.suggested_factor ≥ 1.0
    ///       next_dt = max(next_dt, previous_suggestion)
    ErrorEstimate estimate_error(const mhs::core::SolutionHistory& accepted, const std::vector<double>& trial_T,
        double trial_dt, const ErrorControlConfig& cfg);

} // namespace mhs::sim::time_scheme
