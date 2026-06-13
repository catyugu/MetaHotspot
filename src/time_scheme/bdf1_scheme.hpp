#pragma once

#include "time_scheme.hpp"

namespace mhs::sim::time_scheme {

    /// Backward Euler (1st order) with fixed step.
    class Bdf1Scheme : public TimeScheme {
    public:
        explicit Bdf1Scheme(TimeSchemeConfig cfg) : cfg_(std::move(cfg)) { }

        StepDecision select_step(
            const mhs::core::TimeStepBuffer& history, double current_t, double duration) const override;
        LinearSystem build_system(const StaticOpsResult& sops, const MassOpsResult& mops,
            const mhs::core::TimeStepBuffer& history, std::size_t order, double dt) const override;
        AcceptDecision accept_or_reject(const mhs::core::TimeStepBuffer& history_before,
            const std::vector<double>& T_candidate, const std::vector<double>& error_estimate) const override;

        const TimeSchemeConfig& config() const override { return cfg_; }

    private:
        TimeSchemeConfig cfg_;
    };

} // namespace mhs::sim::time_scheme
