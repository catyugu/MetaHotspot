#include "solver/pardiso_solver.hpp"

#include <Eigen/PardisoSupport>

namespace mhs::sim {

    SolveResult PardisoSolver::solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b)
    {
        Eigen::PardisoLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);

        if (solver.info() != Eigen::Success) {
            return {Eigen::VectorXd(), false, 0.0, 0};
        }

        Eigen::VectorXd x = solver.solve(b);

        return {
            x, solver.info() == Eigen::Success, (A * x - b).norm(),
            1 // direct solver, 1 iteration
        };
    }

    void PardisoSolver::set_config(const SolverConfig& cfg) { config_ = cfg; }

} // namespace mhs::sim
