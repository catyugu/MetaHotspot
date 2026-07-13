#pragma once

#include "linear_solver/linear_solver.hpp"

#include <Eigen/PardisoSupport>

namespace mhs::sim {

    // Direct sparse LU solver backed by Intel MKL Pardiso. Drop-in replacement
    // for EigenSparseLUSolver (general unsymmetric systems). Compiled in only when
    // MHS_ENABLE_PARDISO is defined; without it the factory falls back to
    // EigenSparseLUSolver. Config is unused, but accepted via the base so solvers
    // share a uniform configuration surface.
    class PardisoLUSolver : public LinearSolver {
    public:
        SolveResult solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b) override;
    };

} // namespace mhs::sim
