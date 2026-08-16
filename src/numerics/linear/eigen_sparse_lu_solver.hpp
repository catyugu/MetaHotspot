#pragma once

#include "numerics/linear/linear_solver.hpp"

#include <Eigen/Sparse>

namespace mhs::sim {

    // Direct sparse LU solver (Eigen::SparseLU wrapper).
    // Used for general asymmetric systems; single-threaded.
    //
    // compute(A) detects whether the sparsity pattern changed and implicitly
    // reuses symbolic analysis when possible.
    class EigenSparseLUSolver : public DirectSolver {
    public:
        void compute(const Eigen::SparseMatrix<double>& A) override;
        Eigen::VectorXd solve(const Eigen::VectorXd& b) override;

    private:
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_;
        Eigen::SparseMatrix<double> A_;
        bool analyzed_ = false;
    };

} // namespace mhs::sim
