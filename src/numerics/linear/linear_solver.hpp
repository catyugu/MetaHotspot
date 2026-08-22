#pragma once

#include <Eigen/Sparse>
#include <memory>

namespace mhs::sim {

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

    /// Base for all solver backends. Every solver accepts an initial guess x0:
    ///   - Iterative backends (AmgCg) warm-start from it.
    ///   - Direct backends (Pardiso) silently ignore it (caller may pass
    ///     VectorXd::Zero(n) for a cold start).
    class LinearSolver {
    public:
        virtual ~LinearSolver() = default;

        /// Factorize (direct) or build preconditioner (iterative).
        /// Must be called before solve(...).
        virtual void compute(const Eigen::SparseMatrix<double>& A) = 0;

        /// Solve A * x = b. The initial guess x0 is accepted by all backends;
        /// direct backends ignore it, iterative backends use it as warm-start.
        virtual Eigen::VectorXd solve(const Eigen::VectorXd& b, Eigen::Ref<const Eigen::VectorXd> x0) = 0;

        // Configuration
        void set_config(SolverConfig cfg) { config_ = cfg; }
        const SolverConfig& config() const { return config_; }

        // Diagnostics from the last solve()
        bool success() const { return success_; }
        int iterations() const { return iterations_; }
        double residual() const { return residual_; }

    protected:
        SolverConfig config_;
        bool success_ = false;
        int iterations_ = 0;
        double residual_ = 0.0;
    };

    using SolverPtr = std::unique_ptr<LinearSolver>;
    /// Retained alias for backward compatibility during the refactoring.
    using SolverHandle = SolverPtr;

    // ── Dispatch helpers ──────────────────────────────────────────────────

    inline void solver_compute(SolverHandle& handle, const Eigen::SparseMatrix<double>& A) { handle->compute(A); }

    /// Solve A * x = b with an initial guess. Pass VectorXd::Zero(n) for a
    /// cold start — this is the only solve path, shared by all backends.
    inline Eigen::VectorXd solver_solve(
        SolverHandle& handle, const Eigen::VectorXd& b, Eigen::Ref<const Eigen::VectorXd> x0)
    { return handle->solve(b, x0); }

    inline bool solver_success(const SolverHandle& handle) { return handle->success(); }

    inline int solver_iterations(const SolverHandle& handle) { return handle->iterations(); }

    /// Build a solver from a spec. The default is the self-tuning AMGCL solver
    /// (CG on symmetric, GMRES on non-symmetric operators) so that no direct
    /// MKL/Pardiso dependency is required; Pardiso remains available as an
    /// optional direct backend when MKL is enabled.
    SolverHandle create_solver(const SolverSpec& spec = {});

} // namespace mhs::sim
