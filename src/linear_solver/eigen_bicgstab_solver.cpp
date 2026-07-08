#include "linear_solver/eigen_bicgstab_solver.hpp"
#include <Eigen/Sparse>

namespace mhs::sim {

    SolveResult EigenBiCGSTABSolver::solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b)
    {
        Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);
        solver.setMaxIterations(config_.max_iterations);
        solver.setTolerance(config_.tolerance);

        Eigen::VectorXd x = solver.solve(b);

        return {x, solver.info() == Eigen::Success, solver.error(), static_cast<int>(solver.iterations())};
    }

    void EigenBiCGSTABSolver::set_config(const SolverConfig& cfg) { config_ = cfg; }

} // namespace mhs::sim
