#pragma once

#include "linear_solver/linear_solver.hpp"

namespace mhs::sim {

    // Direct sparse LU solver (Eigen::EigenSparseLU wrapper).
    // Used for general asymmetric systems; single-threaded. Config is unused,
    // but accepted via the base so solvers share a uniform configuration surface.
    class EigenSparseLUSolver : public LinearSolver {
    public:
        SolveResult solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b) override;
    };

} // namespace mhs::sim
