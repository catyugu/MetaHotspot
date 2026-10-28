#pragma once

#ifdef MHS_ENABLE_PARDISO
#include "numerics/linear/linear_solver.hpp"

#include <Eigen/PardisoSupport>

namespace mhs::sim {

    // Direct sparse LU solver backed by Intel MKL Pardiso (general unsymmetric
    // systems). Compiled only when MHS_ENABLE_PARDISO is defined; it is the
    // sole direct backend, so without MKL a direct-solve request throws.
    //
    // compute(A) detects whether the sparsity pattern changed and implicitly
    // reuses symbolic analysis when possible.
    class PardisoLUSolver : public LinearSolver {
    public:
        void compute(const Eigen::SparseMatrix<double>& A) override;
        Eigen::VectorXd solve(const Eigen::VectorXd& b, Eigen::Ref<const Eigen::VectorXd> x0) override;

    private:
        Eigen::PardisoLU<Eigen::SparseMatrix<double>> solver_;
        Eigen::SparseMatrix<double> A_;
        bool analyzed_ = false;
    };

} // namespace mhs::sim

#endif