#pragma once
#include "data/linear_system.hpp"
#include "data/time_step_buffer.hpp"
#include <Eigen/Sparse>

namespace mhs::sim::time_scheme::detail {

    /// Build the BDF1 (Backward Euler) linear system:
    /// A = K + M_diag / dt
    /// b = f_static + M_diag * T_prev / dt
    inline LinearSystem build_bdf1_ls(
        const StaticOpsResult& sops, const MassOpsResult& mops, const mhs::core::TimeStepBuffer& history, double dt)
    {
        const int N = static_cast<int>(sops.f_static.size());
        const auto& T_prev = history.latest();
        Eigen::Map<const Eigen::VectorXd> T_n(T_prev.data(), N);

        Eigen::SparseMatrix<double> A = sops.K;
        // 对角线更新：O(N) 无需破坏稀疏结构
        A.diagonal() += mops.M_diag / dt;

        Eigen::VectorXd b = sops.f_static + mops.M_diag.cwiseProduct(T_n) / dt;

        return {std::move(A), std::move(b)};
    }

    /// Build the BDF2 (variable-step) linear system:
    /// (\alpha_0 M + K) T_{n+1} = f - \alpha_1 M T_n - \alpha_2 M T_{n-1}
    inline LinearSystem build_bdf2_ls(
        const StaticOpsResult& sops, const MassOpsResult& mops, const mhs::core::TimeStepBuffer& history, double dt)
    {
        const int N = static_cast<int>(sops.f_static.size());
        const double h_n = dt;
        const double h_np1 = history.dt_to(1);
        // 防止退化，退化时跌落回定步长处理
        const double delta = (h_np1 > 0.0) ? h_n / h_np1 : 1.0;

        const double alpha0 = (1.0 + 2.0 * delta) / (h_n * (1.0 + delta));
        const double alpha1 = -(1.0 + delta) / (h_n * delta);
        const double alpha2 = delta / (h_n * (1.0 + delta));

        Eigen::Map<const Eigen::VectorXd> T_n(history.latest().data(), N);
        Eigen::Map<const Eigen::VectorXd> T_nm1(history.at(1).data(), N);

        Eigen::SparseMatrix<double> A = sops.K;
        A.diagonal() += alpha0 * mops.M_diag;

        // 修正后的标准 BDF2 右端项
        Eigen::VectorXd b = sops.f_static - mops.M_diag.cwiseProduct(alpha1 * T_n + alpha2 * T_nm1);

        return {std::move(A), std::move(b)};
    }

} // namespace mhs::sim::time_scheme::detail