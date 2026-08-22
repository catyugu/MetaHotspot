#include "numerics/linear/pardiso_lu_solver.hpp"
#include "numerics/linear/solver_utils.hpp"

#include <Eigen/PardisoSupport>

namespace mhs::sim {

    void PardisoLUSolver::compute(const Eigen::SparseMatrix<double>& A)
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

    Eigen::VectorXd PardisoLUSolver::solve(const Eigen::VectorXd& b, Eigen::Ref<const Eigen::VectorXd> /*x0*/)
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