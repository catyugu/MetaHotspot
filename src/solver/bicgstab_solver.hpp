#pragma once

#include "solver/solver.hpp"

namespace mhs {

    // Iterative BiCGSTAB solver (Eigen::BiCGSTAB wrapper).
    // Tolerance and max_iterations come from SolverConfig.
    class BiCGSTABSolver : public Solver {
    public:
        SolveResult solve(const Eigen::SparseMatrix<double>& A,
                          const Eigen::VectorXd& b) override;
        void set_config(const SolverConfig& cfg) override;

    private:
        SolverConfig config_;
    };

} // namespace mhs
