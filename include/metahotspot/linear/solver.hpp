#pragma once

#include "core/csr_matrix.hpp"
#include <memory>
#include <span>

namespace mhs::linear {

    enum class SolverType { Pardiso, Amg };

    // Solver configuration (shared by direct and iterative backends).
    struct SolverConfig {
        // Iterative knobs (Amg).
        double tolerance = 1e-8;
        int max_iterations = 1000;
    };

    struct SolverSpec {
        SolverType type = SolverType::Amg;
        SolverConfig config {};
    };

    struct SolveResult {
        bool success = false;
        int iterations = 0;
        // Backend estimate of ||A*x-b||_2 / ||b||_2; absolute residual
        // for a numerically zero right-hand side.
        double relative_residual = 0.0;
    };

    class Solver {
    public:
        virtual ~Solver() = default;
        // Build factorization/preconditioner. The backend owns what it retains;
        // the input matrix may be destroyed after compute returns.
        virtual void compute(core::CSRMatrixView<double> A) = 0;
        // Both spans have A.rows() elements and must not overlap. x is the
        // initial guess on entry and the solution on return. Direct backends
        // ignore its initial contents.
        virtual SolveResult solve(std::span<const double> b, std::span<double> x) = 0;
        const SolverConfig& config() const { return config_; }

    protected:
        explicit Solver(SolverConfig config) : config_(config) { }
        const SolverConfig config_;
    };

    using SolverPtr = std::unique_ptr<Solver>;

    /// Build a solver from a spec. The default is the self-tuning AMGCL solver
    /// (CG on symmetric, GMRES on non-symmetric operators) so that no direct
    /// MKL/Pardiso dependency is required; Pardiso remains available as an
    /// optional direct backend when MKL is enabled.
    SolverPtr create_solver(const SolverSpec& spec = {});

} // namespace mhs::linear
