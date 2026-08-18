#pragma once

#include <Eigen/Sparse>
#include <memory>
#include <type_traits>
#include <variant>

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

    /// Common base for all backends: factorization/preconditioner setup,
    /// configuration and last-solve diagnostics. The solve interface is NOT
    /// declared here — direct and iterative solvers expose genuinely different
    /// signatures (iterative requires an initial guess for warm start).
    class LinearSolver {
    public:
        virtual ~LinearSolver() = default;

        /// Build factorization (direct) or preconditioner (iterative).
        /// Must be called before solve(...).
        virtual void compute(const Eigen::SparseMatrix<double>& A) = 0;

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

    /// Direct solver: A * x = b with no initial-guess concept.
    class DirectSolver : public LinearSolver {
    public:
        virtual Eigen::VectorXd solve(const Eigen::VectorXd& b) = 0;
    };

    /// Iterative solver: A * x = b given an initial guess x0 (warm start).
    /// The initial guess is REQUIRED — pass an explicit zero vector to cold start.
    class IterativeSolver : public LinearSolver {
    public:
        virtual Eigen::VectorXd solve(const Eigen::VectorXd& b, Eigen::Ref<const Eigen::VectorXd> x0) = 0;
    };

    using DirectSolverPtr = std::unique_ptr<DirectSolver>;
    using IterativeSolverPtr = std::unique_ptr<IterativeSolver>;
    /// Type-erased holder for either backend; driven through the dispatch helpers below.
    using SolverHandle = std::variant<DirectSolverPtr, IterativeSolverPtr>;

    // ── Dispatch helpers: uniform drive over SolverHandle ─────────────────

    inline void solver_compute(SolverHandle& handle, const Eigen::SparseMatrix<double>& A)
    {
        std::visit([&](auto& ptr) { ptr->compute(A); }, handle);
    }

    /// Solve A * x = b without an initial guess. Direct backends only — an
    /// iterative backend has no cold-start interface and throws.
    inline Eigen::VectorXd solver_solve(SolverHandle& handle, const Eigen::VectorXd& b)
    {
        return std::visit(
            [&](auto& ptr) -> Eigen::VectorXd {
                using SolverT = std::remove_reference_t<decltype(*ptr)>;
                if constexpr (std::is_base_of_v<IterativeSolver, SolverT>)
                    throw std::logic_error("iterative solve requires an initial guess (use the x0 overload)");
                else
                    return ptr->solve(b);
            },
            handle);
    }

    /// Solve A * x = b. The initial guess x0 is forwarded to iterative backends
    /// (warm start) and ignored by direct backends.
    inline Eigen::VectorXd solver_solve(
        SolverHandle& handle, const Eigen::VectorXd& b, Eigen::Ref<const Eigen::VectorXd> x0)
    {
        return std::visit(
            [&](auto& ptr) -> Eigen::VectorXd {
                using SolverT = std::remove_reference_t<decltype(*ptr)>;
                if constexpr (std::is_base_of_v<IterativeSolver, SolverT>)
                    return ptr->solve(b, x0);
                else
                    return ptr->solve(b);
            },
            handle);
    }

    inline bool solver_success(const SolverHandle& handle)
    {
        return std::visit([](const auto& ptr) { return ptr->success(); }, handle);
    }

    inline int solver_iterations(const SolverHandle& handle)
    {
        return std::visit([](const auto& ptr) { return ptr->iterations(); }, handle);
    }

    /// Build a solver from a spec. The default is the self-tuning AMGCL solver
    /// (CG on symmetric, GMRES on non-symmetric operators) so that no direct
    /// MKL/Pardiso dependency is required; Pardiso remains available as an
    /// optional direct backend when MKL is enabled.
    SolverHandle create_solver(const SolverSpec& spec = {});

} // namespace mhs::sim
