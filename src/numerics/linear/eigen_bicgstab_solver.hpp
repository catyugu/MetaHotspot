#pragma once

#include "numerics/linear/linear_solver.hpp"

#include <Eigen/IterativeLinearSolvers>

namespace mhs::sim {

    // Iterative BiCGSTAB solver (Eigen::BiCGSTAB wrapper).
    // Reads tolerance and max_iterations from the base class's SolverConfig,
    // seeded by the factory from SolverSpec::config.
    //
    // solve(b, x0) warm-starts the Krylov iteration from the given initial
    // guess via Eigen's solveWithGuess — callers should seed x0 with the
    // previous solution whenever the systems change slowly.
    class EigenBiCGSTABSolver : public IterativeSolver {
    public:
        void compute(const Eigen::SparseMatrix<double>& A) override;
        Eigen::VectorXd solve(const Eigen::VectorXd& b, Eigen::Ref<const Eigen::VectorXd> x0) override;

    private:
        Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver_;
        Eigen::SparseMatrix<double> A_; // cached for residual computation
    };

} // namespace mhs::sim
