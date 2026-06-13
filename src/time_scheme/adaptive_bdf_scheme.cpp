#include "adaptive_bdf_scheme.hpp"
#include "detail/build_ops.hpp"
#include "step_controller.hpp"

#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>

namespace mhs::sim::time_scheme {

    void AdaptiveBdfScheme::initialize(mhs::core::TimeStepBuffer& history, mhs::core::GlobalState& state) const
    {
        history.reset(state.T);
        next_dt_ = cfg_.initial_dt;
        last_dt_ = cfg_.initial_dt;
        output_step_ = 0;
    }

    StepDecision AdaptiveBdfScheme::select_step(
        const mhs::core::TimeStepBuffer& history, double current_t, double duration) const
    {
        // Use the dt recommended by the previous accept_or_reject call.
        double dt = (next_dt_ > 0.0) ? next_dt_ : cfg_.initial_dt;
        dt = std::clamp(dt, cfg_.min_dt, cfg_.max_dt);

        // Output-time alignment: clamp dt so the next step boundary lands
        // exactly on t = (output_step_ + 1) * output_dt.
        if (cfg_.output_dt > 0.0) {
            double t_next_out = static_cast<double>(output_step_ + 1) * cfg_.output_dt;
            if (t_next_out > current_t && t_next_out < current_t + dt) {
                dt = t_next_out - current_t;
            }
        }

        // Duration clamp.
        double remaining = duration - current_t;
        if (dt > remaining)
            dt = remaining;

        last_dt_ = dt;
        std::size_t order = (history.size() > cfg_.max_order) ? cfg_.max_order : 1;
        return {dt, order};
    }

    LinearSystem AdaptiveBdfScheme::build_system(const StaticOpsResult& sops, const MassOpsResult& mops,
        const mhs::core::TimeStepBuffer& history, std::size_t /*order*/, double dt) const
    {
        return detail::build_bdf1_ls(sops, mops, history, dt);
    }

    AcceptDecision AdaptiveBdfScheme::accept_or_reject(const mhs::core::TimeStepBuffer& /*history_before*/,
        const std::vector<double>& /*T_candidate*/, const std::vector<double>& error_estimate) const
    {
        StepController ctrl(cfg_);
        double err = StepController::error_norm(error_estimate);
        auto r = ctrl.decide(last_dt_, /*order=*/1, err);
        // Always accept (no step rejection loop in the scheduler yet);
        // the controller's next_dt is stored for the next select_step call.
        next_dt_ = r.next_dt;
        return AcceptDecision::Accept;
    }

    bool AdaptiveBdfScheme::is_output_boundary(double t) const
    {
        if (cfg_.output_dt <= 0.0)
            return true; // no explicit output grid → record every step
        double t_next_out = static_cast<double>(output_step_ + 1) * cfg_.output_dt;
        if (std::abs(t - t_next_out) <= 1e-9 * std::max(1.0, t)) {
            ++output_step_;
            return true;
        }
        return false;
    }

} // namespace mhs::sim::time_scheme
