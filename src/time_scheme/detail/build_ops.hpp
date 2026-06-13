#pragma once

#include "assembler/assembler.hpp"
#include "data/time_step_buffer.hpp"

#include <Eigen/Sparse>

namespace mhs::sim::time_scheme::detail {

    /// Build the BDF1 (Backward Euler) linear system:
    ///   A = K + M_diag / dt
    ///   b = f_static + M_diag * T_prev / dt
    inline LinearSystem build_bdf1_ls(
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

    /// Build the BDF2 (variable-step) linear system:
    ///   h_n   = dt (current step)
    ///   h_np1 = dt_to(1) (previous step)
    ///   δ     = h_n / h_np1
    ///   α0    = (1 + 2δ) / (h_n · (1+δ))
    ///   α1    = -(1+δ) / (h_n · δ)
    ///   α2    = δ / (h_n · (1+δ))
    ///   A     = α0·M_diag + K
    ///   b     = α0·M_diag·T_n + α1·M_diag·T_{n-1} + α2·M_diag·T_{n-2} + f_static
    ///
    /// Falls back to fixed-step coefficients (δ=1) when the previous step is degenerate.
    inline LinearSystem build_bdf2_ls(
        const StaticOpsResult& sops, const MassOpsResult& mops, const mhs::core::TimeStepBuffer& history, double dt)
    {
        const int N = static_cast<int>(sops.f_static.size());
        const double h_n = dt;
        const double h_np1 = history.dt_to(1);

        // Guard against degenerate step; falls back to fixed-step (δ=1).
        const double delta = (h_np1 > 0.0) ? h_n / h_np1 : 1.0;

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

} // namespace mhs::sim::time_scheme::detail
