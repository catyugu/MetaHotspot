#include "linear_solver/eigen_bicgstab_solver.hpp"
#include <Eigen/IterativeLinearSolvers>

namespace mhs::sim {

    void EigenBiCGSTABSolver::compute(const Eigen::SparseMatrix<double>& A)
    {
        A_ = A;
        solver_.compute(A_);
        solver_.setMaxIterations(config_.max_iterations);
        solver_.setTolerance(config_.tolerance);
    }

    Eigen::VectorXd EigenBiCGSTABSolver::solve(const Eigen::VectorXd& b)
    {
        Eigen::VectorXd x = solver_.solve(b);
        success_ = (solver_.info() == Eigen::Success);
        iterations_ = static_cast<int>(solver_.iterations());
        residual_ = solver_.error();
        return x;
    }

} // namespace mhs::sim
