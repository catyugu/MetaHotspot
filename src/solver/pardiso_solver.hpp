#pragma once

#include "solver/solver.hpp"

#include <Eigen/PardisoSupport>

namespace mhs {

    // Direct sparse LU solver backed by Intel MKL Pardiso. Drop-in replacement
    // for SparseLUSolver (general unsymmetric systems). Compiled in only when
    // MHS_ENABLE_PARDISO is defined; without it the factory falls back to
    // SparseLUSolver.
    class PardisoSolver : public Solver {
    public:
        SolveResult solve(const Eigen::SparseMatrix<double>& A,
                          const Eigen::VectorXd& b) override;
        void set_config(const SolverConfig& cfg) override;

    private:
        SolverConfig config_;
    };

} // namespace mhs
