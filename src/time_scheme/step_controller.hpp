#pragma once

#include "time_scheme.hpp"

namespace mhs::sim::time_scheme {

    /// Step controller: decides accept/reject and computes the next dt
    /// based on the local truncation error estimate.  Follows the HNW
    /// (Hairer-Norsett-Wanner) strategy for step-size control.
    class StepController {
    public:
        explicit StepController(TimeSchemeConfig cfg) : cfg_(std::move(cfg)) { }

        /// Compute the infinity-norm of the error estimate.
        static double error_norm(const std::vector<double>& e);

        /// Decide whether to accept the step and compute the next dt.
        /// Returns {accepted, next_dt}.
        struct StepResult { bool accepted; double next_dt; };
        StepResult decide(double current_dt, std::size_t order, double error_norm) const;

    private:
        TimeSchemeConfig cfg_;
    };

} // namespace mhs::sim::time_scheme
