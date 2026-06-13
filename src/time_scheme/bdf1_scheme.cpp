#include "bdf1_scheme.hpp"
#include "detail/build_ops.hpp"

#include <Eigen/Sparse>

namespace mhs::sim::time_scheme {

    StepDecision Bdf1Scheme::select_step(
        const mhs::core::TimeStepBuffer& /*history*/, double /*current_t*/, double /*duration*/) const
    {
        return {cfg_.initial_dt, 1};
    }

    LinearSystem Bdf1Scheme::build_system(const StaticOpsResult& sops, const MassOpsResult& mops,
        const mhs::core::TimeStepBuffer& history, std::size_t /*order*/, double dt) const
    {
        return detail::build_bdf1_ls(sops, mops, history, dt);
    }

    AcceptDecision Bdf1Scheme::accept_or_reject(const mhs::core::TimeStepBuffer& /*history_before*/,
        const std::vector<double>& /*T_candidate*/, const std::vector<double>& /*error_estimate*/) const
    {
        return AcceptDecision::Accept;
    }

} // namespace mhs::sim::time_scheme
