#pragma once

#include "linear_solver/linear_solver.hpp"

namespace mhs::sim {

    // Direct sparse LU solver (Eigen::SparseLU wrapper).
    // Used for general asymmetric systems; single-threaded.
    class SparseLUSolver : public LinearSolver {
    public:
        SolveResult solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b) override;
        void set_config(const SolverConfig& cfg) override;

    private:
        SolverConfig config_;
    };

} // namespace mhs::sim
