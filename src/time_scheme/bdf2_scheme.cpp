#include "bdf2_scheme.hpp"
#include "detail/build_ops.hpp"

#include <Eigen/Sparse>

namespace mhs::sim::time_scheme {

    StepDecision Bdf2Scheme::select_step(
        const mhs::core::TimeStepBuffer& history, double /*current_t*/, double /*duration*/) const
    {
        std::size_t order = (history.size() >= 3) ? 2 : 1;
        return {cfg_.initial_dt, order};
    }

    LinearSystem Bdf2Scheme::build_system(const StaticOpsResult& sops, const MassOpsResult& mops,
        const mhs::core::TimeStepBuffer& history, std::size_t order, double dt) const
    {
        if (order == 1 || history.size() <= order) {
            return detail::build_bdf1_ls(sops, mops, history, dt);
        }
        return detail::build_bdf2_ls(sops, mops, history, dt);
    }

    AcceptDecision Bdf2Scheme::accept_or_reject(const mhs::core::TimeStepBuffer& /*history_before*/,
        const std::vector<double>& /*T_candidate*/, const std::vector<double>& /*error_estimate*/) const
    {
        return AcceptDecision::Accept;
    }

} // namespace mhs::sim::time_scheme
