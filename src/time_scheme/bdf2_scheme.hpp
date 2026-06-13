#pragma once

#include "time_scheme.hpp"

namespace mhs::sim::time_scheme {

    /// Backward Differentiation Formula, order 2, fixed step.
    /// At startup (history.size() < 2), it falls back to BDF1.
    class Bdf2Scheme : public TimeScheme {
    public:
        explicit Bdf2Scheme(TimeSchemeConfig cfg) : cfg_(std::move(cfg)) { }

        void initialize(mhs::core::TimeStepBuffer& history, mhs::core::GlobalState& state) const override;
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
