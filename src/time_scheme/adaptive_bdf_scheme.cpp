#include "adaptive_bdf_scheme.hpp"
#include "step_controller.hpp"

#include <Eigen/Sparse>

namespace mhs::sim::time_scheme {

    namespace {
        LinearSystem build_bdf1_ls(const StaticOpsResult& sops, const MassOpsResult& mops,
            const mhs::core::TimeStepBuffer& history, double dt)
        {
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

        LinearSystem build_bdf2_ls(const StaticOpsResult& sops, const MassOpsResult& mops,
            const mhs::core::TimeStepBuffer& history, double dt)
        {
            const int N = static_cast<int>(sops.f_static.size());
            const double h_n   = dt;
            const double h_np1 = history.dt_to(1);
            double delta = (h_np1 > 0.0) ? h_n / h_np1 : 1.0;
            const double alpha0 = (1.0 + 2.0 * delta) / (h_n * (1.0 + delta));
            const double alpha1 = -(1.0 + delta) / (h_n * delta);
            const double alpha2 = delta / (h_n * (1.0 + delta));
            const auto& T_n   = history.latest();
            const auto& T_nm1 = history.at(1);
            const auto& T_nm2 = history.at(2);
            Eigen::SparseMatrix<double> A = sops.K;
            Eigen::VectorXd b = sops.f_static;
            for (int i = 0; i < N; ++i) {
                A.coeffRef(i, i) += alpha0 * mops.M_diag(i);
                b(i) += alpha0 * mops.M_diag(i) * T_n[i]
                      + alpha1 * mops.M_diag(i) * T_nm1[i]
                      + alpha2 * mops.M_diag(i) * T_nm2[i];
            }
            return {A, b, b - A * Eigen::Map<const Eigen::VectorXd>(T_n.data(), N)};
        }
    } // namespace

    void AdaptiveBdfScheme::initialize(
        mhs::core::TimeStepBuffer& history, mhs::core::GlobalState& state) const
    {
        history.reset(state.T);
    }

    StepDecision AdaptiveBdfScheme::select_step(
        const mhs::core::TimeStepBuffer& history, double /*current_t*/) const
    {
        // Start at order 1, then promote to max_order as history grows.
        std::size_t order = (history.size() >= 2) ? cfg_.max_order : 1;
        // Clamp initial dt to [min_dt, max_dt]
        double dt = cfg_.initial_dt;
        if (dt < cfg_.min_dt) dt = cfg_.min_dt;
        if (dt > cfg_.max_dt) dt = cfg_.max_dt;
        return {dt, order};
    }

    LinearSystem AdaptiveBdfScheme::build_system(const StaticOpsResult& sops, const MassOpsResult& mops,
        const mhs::core::TimeStepBuffer& history, std::size_t order, double dt) const
    {
        if (order == 1 || history.size() < 2) {
            return build_bdf1_ls(sops, mops, history, dt);
        }
        return build_bdf2_ls(sops, mops, history, dt);
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
