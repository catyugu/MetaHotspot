#pragma once

#ifdef MHS_ENABLE_PARDISO
#include "numerics/linear/linear_solver.hpp"

#include <Eigen/PardisoSupport>

namespace mhs::sim {

    // Direct sparse LU solver backed by Intel MKL Pardiso. Drop-in replacement
    // for EigenSparseLUSolver (general unsymmetric systems). Compiled in only when
    // MHS_ENABLE_PARDISO is defined; without it the factory falls back to
    // EigenSparseLUSolver.
    //
    // compute(A) detects whether the sparsity pattern changed and implicitly
    // reuses symbolic analysis when possible.
    class PardisoLUSolver : public LinearSolver {
    public:
        void compute(const Eigen::SparseMatrix<double>& A) override;
        Eigen::VectorXd solve(const Eigen::VectorXd& b) override;

    private:
        Eigen::PardisoLU<Eigen::SparseMatrix<double>> solver_;
        Eigen::SparseMatrix<double> A_;
        bool analyzed_ = false;
    };

} // namespace mhs::sim

#endif