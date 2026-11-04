#pragma once

#include "core/csr_matrix.hpp"
#include <memory>
#include <span>

namespace mhs::numerics {

    enum class SolverType { Pardiso, AmgCg };

    // LinearSolver configuration (shared by direct and iterative backends).
    struct SolverConfig {
        // Iterative knobs (AmgCg).
        double tolerance = 1e-8;
        int max_iterations = 1000;
    };

    struct SolverSpec {
        SolverType type = SolverType::AmgCg;
        SolverConfig config {};
    };

    struct LinearSolveResult {
        bool success = false;
        int iterations = 0;
        double residual = 0.0;
    };

    class LinearSolver {
    public:
        virtual ~LinearSolver() = default;
        // Build factorization/preconditioner. The backend owns what it retains;
        // the input matrix may be destroyed after compute returns.
        virtual void compute(const core::CSRMatrix<double>& A) = 0;
        // All spans have A.rows() elements. x may alias x0, but must
        // not overlap b. Direct backends ignore x0.
        virtual LinearSolveResult solve(std::span<const double> b, std::span<const double> x0, std::span<double> x) = 0;
        void set_config(SolverConfig cfg) { config_ = cfg; }
        const SolverConfig& config() const { return config_; }

    protected:
        SolverConfig config_;
    };

    using SolverPtr = std::unique_ptr<LinearSolver>;

    /// Build a solver from a spec. The default is the self-tuning AMGCL solver
    /// (CG on symmetric, GMRES on non-symmetric operators) so that no direct
    /// MKL/Pardiso dependency is required; Pardiso remains available as an
    /// optional direct backend when MKL is enabled.
    SolverPtr create_solver(const SolverSpec& spec = {});

} // namespace mhs::numerics
