#pragma once

#include "numerics/linear/linear_solver.hpp"

#include <memory>

namespace mhs::sim {

    /// AMG-preconditioned Krylov iterative solver built on AMGCL (smoothed-
    /// aggregation AMG). The Krylov method is auto-selected on the assembled
    /// operator:
    ///
    ///   * symmetric positive definite (pure conduction/convection) -> CG,
    ///   * non-symmetric (e.g. fluid-coupled cases with upwind advection
    ///     introduced by circulating coolant / mass-flow boundaries) -> GMRES.
    ///
    /// So it is a robust default for both purely thermal and fluid-coupled
    /// solves, while keeping the scalable AMG preconditioner in both branches.
    ///
    /// The header exposes only a pimpl so that no AMGCL header leaks into
    /// translation units compiled with the project's strict warning flags.
    /// Uses the same SolverConfig knobs (tolerance, max_iterations) as the
    /// other backends and warm-starts from the supplied initial guess.
    class AmgCgSolver : public IterativeSolver {
    public:
        AmgCgSolver();
        ~AmgCgSolver() override;

        void compute(const Eigen::SparseMatrix<double>& A) override;
        Eigen::VectorXd solve(const Eigen::VectorXd& b, Eigen::Ref<const Eigen::VectorXd> x0) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace mhs::sim
