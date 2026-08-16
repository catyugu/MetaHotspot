#include "numerics/linear/eigen_bicgstab_solver.hpp"
#include <Eigen/IterativeLinearSolvers>

#include <stdexcept>

namespace mhs::sim {

    void EigenBiCGSTABSolver::compute(const Eigen::SparseMatrix<double>& A)
    {
        A_ = A;
        solver_.compute(A_);
        solver_.setMaxIterations(config_.max_iterations);
        solver_.setTolerance(config_.tolerance);
    }

    Eigen::VectorXd EigenBiCGSTABSolver::solve(
        const Eigen::VectorXd& b, Eigen::Ref<const Eigen::VectorXd> x0)
    {
        if (x0.size() != b.size()) {
            throw std::invalid_argument(
                "EigenBiCGSTABSolver::solve: initial guess size must match b (" + std::to_string(x0.size())
                + " != " + std::to_string(b.size()) + ")");
        }
        Eigen::VectorXd x = solver_.solveWithGuess(b, x0);
        success_ = (solver_.info() == Eigen::Success);
        iterations_ = static_cast<int>(solver_.iterations());
        residual_ = solver_.error();
        return x;
    }

} // namespace mhs::sim
