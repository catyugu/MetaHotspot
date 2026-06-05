#include "solver/sparse_lu_solver.hpp"
#include <Eigen/Sparse>

namespace mhs::sim {

    SolveResult SparseLUSolver::solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b)
    {
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
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

    void SparseLUSolver::set_config(const SolverConfig& cfg) { config_ = cfg; }

} // namespace mhs::sim
