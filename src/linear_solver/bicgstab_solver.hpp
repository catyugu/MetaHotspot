#pragma once

#include "linear_solver/linear_solver.hpp"

namespace mhs::sim {

    // Iterative BiCGSTAB solver (Eigen::BiCGSTAB wrapper).
    // Tolerance and max_iterations come from SolverConfig.
    class BiCGSTABSolver : public LinearSolver {
    public:
        SolveResult solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b) override;
        void set_config(const SolverConfig& cfg) override;

    private:
        SolverConfig config_;
    };

} // namespace mhs::sim
