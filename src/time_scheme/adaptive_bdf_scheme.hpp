#pragma once

#include "time_scheme.hpp"

namespace mhs::sim::time_scheme {

    /// Adaptive-step BDF1 scheme.
    ///
    /// Selects step size adaptively based on the step-change error estimate
    /// passed via accept_or_reject().  Also handles output-time alignment
    /// (clamping dt so that step boundaries land exactly on output_dt
    /// intervals) — the caller applies select_step()'s dt directly.
    ///
    /// Internal mutable state (next_dt, output_step) is updated inside
    /// select_step / accept_or_reject / is_output_boundary; the scheme
    /// interface remains nominally const so the factory can return a
    /// unique_ptr<TimeScheme>.
    class AdaptiveBdfScheme : public TimeScheme {
    public:
        explicit AdaptiveBdfScheme(TimeSchemeConfig cfg) : cfg_(std::move(cfg)) { }

        void initialize(mhs::core::TimeStepBuffer& history, mhs::core::GlobalState& state) const override;

        StepDecision select_step(
            const mhs::core::TimeStepBuffer& history, double current_t, double duration) const override;
        LinearSystem build_system(const StaticOpsResult& sops, const MassOpsResult& mops,
            const mhs::core::TimeStepBuffer& history, std::size_t order, double dt) const override;
        AcceptDecision accept_or_reject(const mhs::core::TimeStepBuffer& history_before,
            const std::vector<double>& T_candidate, const std::vector<double>& error_estimate) const override;

        bool is_output_boundary(double t) const override;

        const TimeSchemeConfig& config() const override { return cfg_; }

    private:
        TimeSchemeConfig cfg_;

        // --- mutable tracking state (updated by select_step / accept_or_reject) ---
        mutable double last_dt_ = 0.0; // dt that was just used
        mutable double next_dt_ = 0.0; // dt recommended for the next step
        mutable int output_step_ = 0; // next output frame index
    };

} // namespace mhs::sim::time_scheme
