#include "bdf2_scheme.hpp"

#include <Eigen/Sparse>

namespace mhs::sim::time_scheme {

    void Bdf2Scheme::initialize(mhs::core::TimeStepBuffer& history, mhs::core::GlobalState& state) const
    {
        history.reset(state.T);
    }

    StepDecision Bdf2Scheme::select_step(const mhs::core::TimeStepBuffer& history, double /*current_t*/) const
    {
        // For BDF2 we need 2 prior snapshots.  If we only have 1, return
        // order=1 so build_system falls back to BDF1 for the first step.
        std::size_t order = (history.size() >= 2) ? 2 : 1;
        return {cfg_.initial_dt, order};
    }

    namespace {
        // BDF1 system builder reused by the startup path
        LinearSystem build_bdf1_ls(
            const StaticOpsResult& sops, const MassOpsResult& mops, const mhs::core::TimeStepBuffer& history, double dt)
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
    } // namespace

    LinearSystem Bdf2Scheme::build_system(const StaticOpsResult& sops, const MassOpsResult& mops,
        const mhs::core::TimeStepBuffer& history, std::size_t order, double dt) const
    {
        if (order == 1 || history.size() < 2) {
            return build_bdf1_ls(sops, mops, history, dt);
        }

        // BDF2 coefficients (variable-step form):
        //   h_n   = t_n - t_{n-1}    (current step)
        //   h_np1 = t_{n-1} - t_{n-2} (previous step)
        //   δ     = h_n / h_np1
        //   α0    = (1 + 2δ) / (h_n · (1+δ))
        //   α1    = -(1+δ) / (h_n · δ)
        //   α2    = δ / (h_n · (1+δ))
        //   A     = α0·M + K
        //   b     = α0·M·T_n + α1·M·T_{n-1} + α2·M·T_{n-2} + f_static
        //
        // For fixed step (δ=1) this reduces to:
        //   α0=3/(2h), α1=-2/h, α2=1/(2h)

        const int N = static_cast<int>(sops.f_static.size());
        const double h_n = dt; // current step length
        const double h_np1 = history.dt_to(1); // previous step length

        // Guard against degenerate step (e.g. first call where history timestamps
        // haven't been set distinctly).  Falls back to fixed-step BDF2.
        double delta;
        if (h_np1 <= 0.0)
            delta = 1.0;
        else
            delta = h_n / h_np1;

        const double alpha0 = (1.0 + 2.0 * delta) / (h_n * (1.0 + delta));
        const double alpha1 = -(1.0 + delta) / (h_n * delta);
        const double alpha2 = delta / (h_n * (1.0 + delta));

        const auto& T_n = history.latest();
        const auto& T_nm1 = history.at(1);
        const auto& T_nm2 = history.at(2);

        Eigen::SparseMatrix<double> A = sops.K;
        Eigen::VectorXd b = sops.f_static;
        for (int i = 0; i < N; ++i) {
            A.coeffRef(i, i) += alpha0 * mops.M_diag(i);
            b(i) += alpha0 * mops.M_diag(i) * T_n[i] + alpha1 * mops.M_diag(i) * T_nm1[i]
                + alpha2 * mops.M_diag(i) * T_nm2[i];
        }

        return {A, b, b - A * Eigen::Map<const Eigen::VectorXd>(T_n.data(), N)};
    }

    AcceptDecision Bdf2Scheme::accept_or_reject(const mhs::core::TimeStepBuffer& /*history_before*/,
        const std::vector<double>& /*T_candidate*/, const std::vector<double>& /*error_estimate*/) const
    {
        return AcceptDecision::Accept;
    }

} // namespace mhs::sim::time_scheme
