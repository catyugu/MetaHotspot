#include "linear_solver/eigen_sparse_lu_solver.hpp"
#include "linear_solver/solver_utils.hpp"

namespace mhs::sim {

    void EigenSparseLUSolver::compute(const Eigen::SparseMatrix<double>& A)
    {
        const bool reuse = analyzed_ && same_pattern(A, A_);
        A_ = A;
        if (reuse) {
            solver_.factorize(A_);
        }
        else {
            solver_.analyzePattern(A_);
            solver_.factorize(A_);
            analyzed_ = true;
        }
    }

    Eigen::VectorXd EigenSparseLUSolver::solve(const Eigen::VectorXd& b)
    {
        success_ = (solver_.info() == Eigen::Success);
        if (!success_) {
            iterations_ = 0;
            residual_ = 0.0;
            return Eigen::VectorXd();
        }

        Eigen::VectorXd x = solver_.solve(b);
        success_ = (solver_.info() == Eigen::Success);
        iterations_ = 1; // direct solver
        if (success_) {
            residual_ = (A_ * x - b).norm();
        }
        else {
            residual_ = 0.0;
        }
        return x;
    }

} // namespace mhs::sim
