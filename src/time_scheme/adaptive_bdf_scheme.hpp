#pragma once

#include "time_scheme.hpp"

namespace mhs::sim::time_scheme {

    /// Adaptive-step, variable-order BDF (orders 1..max_order).
    /// The embedded error estimate is `T^{(k)} - T^{(k-1)}` (difference of
    /// the k-order and (k-1)-order predictions).
    class AdaptiveBdfScheme : public TimeScheme {
    public:
        explicit AdaptiveBdfScheme(TimeSchemeConfig cfg) : cfg_(std::move(cfg)) { }

        StepDecision select_step(const mhs::core::TimeStepBuffer& history, double current_t) const override;
        LinearSystem build_system(const StaticOpsResult& sops, const MassOpsResult& mops,
            const mhs::core::TimeStepBuffer& history, std::size_t order, double dt) const override;
        AcceptDecision accept_or_reject(const mhs::core::TimeStepBuffer& history_before,
            const std::vector<double>& T_candidate, const std::vector<double>& error_estimate) const override;

        const TimeSchemeConfig& config() const override { return cfg_; }

    private:
        TimeSchemeConfig cfg_;
    };

} // namespace mhs::sim::time_scheme
