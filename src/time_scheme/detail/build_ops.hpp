#pragma once
#include "assembler/assembler.hpp"
#include "data/linear_system.hpp"
#include "data/solution_history.hpp"
#include <Eigen/Sparse>

namespace mhs::sim::time_scheme::detail {

    /// Build the BDF1 (Backward Euler) linear system:
    /// A = K + M_diag / dt
    /// b = f + M_diag * T_prev / dt
    inline LinearSystem build_bdf1_ls(const AssemblyResult& ops, const mhs::core::SolutionHistory& history, double dt)
    {
        const int N = static_cast<int>(ops.f.size());
        const auto& T_prev = history.current();
        Eigen::Map<const Eigen::VectorXd> T_n(T_prev.data(), N);

        const Eigen::VectorXd m_over_dt = ops.M_diag / dt;

        Eigen::SparseMatrix<double> A = ops.K;
        // 对角线更新：O(N) 无需破坏稀疏结构
        A.diagonal() += m_over_dt;

        Eigen::VectorXd b = ops.f + m_over_dt.cwiseProduct(T_n);

        return {std::move(A), std::move(b)};
    }

    /// Build the BDF2 (variable-step) linear system:
    /// (\alpha_0 M + K) T_{n+1} = f - \alpha_1 M T_n - \alpha_2 M T_{n-1}
    inline LinearSystem build_bdf2_ls(const AssemblyResult& ops, const mhs::core::SolutionHistory& history, double dt)
    {
        const int N = static_cast<int>(ops.f.size());
        const double h_n = dt;
        const double h_nm1 = history.previous_dt();
        const double delta = h_n / h_nm1;

        const double alpha0 = (1.0 + 2.0 * delta) / (h_n * (1.0 + delta));

        const double alpha1 = -(1.0 + delta) / h_n;
        const double alpha2 = (delta * delta) / (h_n * (1.0 + delta));

        Eigen::Map<const Eigen::VectorXd> T_n(history.current().data(), N);
        Eigen::Map<const Eigen::VectorXd> T_nm1(history.at(1).data(), N);

        Eigen::SparseMatrix<double> A = ops.K;
        A.diagonal() += alpha0 * ops.M_diag;

        Eigen::VectorXd b = ops.f - ops.M_diag.cwiseProduct(alpha1 * T_n + alpha2 * T_nm1);

        return {std::move(A), std::move(b)};
    }

} // namespace mhs::sim::time_scheme::detail
