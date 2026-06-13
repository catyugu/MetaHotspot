#include "adaptive_bdf_scheme.hpp"
#include "detail/build_ops.hpp"
#include "step_controller.hpp"

#include <Eigen/Sparse>
#include <algorithm>

namespace mhs::sim::time_scheme {

    StepDecision AdaptiveBdfScheme::select_step(const mhs::core::TimeStepBuffer& history, double /*current_t*/) const
    {
        std::size_t order = (history.size() >= 2) ? cfg_.max_order : 1;
        double dt = std::clamp(cfg_.initial_dt, cfg_.min_dt, cfg_.max_dt);
        return {dt, order};
    }

    LinearSystem AdaptiveBdfScheme::build_system(const StaticOpsResult& sops, const MassOpsResult& mops,
        const mhs::core::TimeStepBuffer& history, std::size_t order, double dt) const
    {
        if (order == 1 || history.size() < 2) {
            return detail::build_bdf1_ls(sops, mops, history, dt);
        }
        return detail::build_bdf2_ls(sops, mops, history, dt);
    }

    AcceptDecision AdaptiveBdfScheme::accept_or_reject(const mhs::core::TimeStepBuffer& /*history_before*/,
        const std::vector<double>& /*T_candidate*/, const std::vector<double>& error_estimate) const
    {
        StepController ctrl(cfg_);
        double e = StepController::error_norm(error_estimate);
        auto r = ctrl.decide(/*current_dt=*/1.0, /*order=*/1, e);
        return r.accepted ? AcceptDecision::Accept : AcceptDecision::Reject;
    }

} // namespace mhs::sim::time_scheme
