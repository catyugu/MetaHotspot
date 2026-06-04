#pragma once

#include "solver/solver.hpp"

namespace mhs {

    // Direct sparse LU solver (Eigen::SparseLU wrapper).
    // Used for general asymmetric systems; single-threaded.
    class SparseLUSolver : public Solver {
    public:
        SolveResult solve(const Eigen::SparseMatrix<double>& A,
                          const Eigen::VectorXd& b) override;
        void set_config(const SolverConfig& cfg) override;

    private:
        SolverConfig config_;
    };

} // namespace mhs
