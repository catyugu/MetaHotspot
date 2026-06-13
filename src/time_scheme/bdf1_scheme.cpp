#include "bdf1_scheme.hpp"

#include <Eigen/Sparse>

namespace mhs::sim::time_scheme {

    void Bdf1Scheme::initialize(mhs::core::TimeStepBuffer& history, mhs::core::GlobalState& state) const
    {
        // BDF1 needs only the most recent slot.
        history.reset(state.T);
    }

    StepDecision Bdf1Scheme::select_step(const mhs::core::TimeStepBuffer& /*history*/, double /*current_t*/) const
    {
        return {cfg_.initial_dt, 1};
    }

    LinearSystem Bdf1Scheme::build_system(const StaticOpsResult& sops, const MassOpsResult& mops,
        const mhs::core::TimeStepBuffer& history, std::size_t /*order*/, double dt) const
    {
        // BDF1 (Backward Euler):
        //   A = K + M_diag/dt
        //   b = f_static + M_diag * T_prev / dt
        const int N = static_cast<int>(sops.f_static.size());
        const auto& T_prev = history.latest();

        Eigen::SparseMatrix<double> A = sops.K;
        Eigen::VectorXd b = sops.f_static;

        for (int i = 0; i < N; ++i) {
            A.coeffRef(i, i) += mops.M_diag(i) / dt;
            b(i) += mops.M_diag(i) * T_prev[i] / dt;
        }

        return {A, b, b - A * Eigen::Map<const Eigen::VectorXd>(T_prev.data(), N)};
    }

    AcceptDecision Bdf1Scheme::accept_or_reject(const mhs::core::TimeStepBuffer& /*history_before*/,
        const std::vector<double>& /*T_candidate*/, const std::vector<double>& /*error_estimate*/) const
    {
        return AcceptDecision::Accept;
    }

} // namespace mhs::sim::time_scheme
