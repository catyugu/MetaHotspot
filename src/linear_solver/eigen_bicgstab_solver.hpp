#pragma once

#include "linear_solver/linear_solver.hpp"

namespace mhs::sim {

    // Iterative EigenBiCGSTAB solver (Eigen::EigenBiCGSTAB wrapper).
    // Reads tolerance and max_iterations from the base class's SolverConfig,
    // seeded by the factory from SolverSpec::config.
    class EigenBiCGSTABSolver : public LinearSolver {
    public:
        SolveResult solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b) override;
    };

} // namespace mhs::sim
