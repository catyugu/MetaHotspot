#include "linear_solver/eigen_sparse_lu_solver.hpp"
#include <Eigen/Sparse>

namespace mhs::sim {

    SolveResult EigenSparseLUSolver::solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b)
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

} // namespace mhs::sim
