#pragma once

#include "linear_solver/linear_solver.hpp"

#include <Eigen/IterativeLinearSolvers>

namespace mhs::sim {

    // Iterative BiCGSTAB solver (Eigen::BiCGSTAB wrapper).
    // Reads tolerance and max_iterations from the base class's SolverConfig,
    // seeded by the factory from SolverSpec::config.
    class EigenBiCGSTABSolver : public LinearSolver {
    public:
        void compute(const Eigen::SparseMatrix<double>& A) override;
        Eigen::VectorXd solve(const Eigen::VectorXd& b) override;

    private:
        Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver_;
        Eigen::SparseMatrix<double> A_; // cached for residual computation
    };

} // namespace mhs::sim
