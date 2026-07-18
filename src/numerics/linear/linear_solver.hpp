#pragma once

#include <Eigen/Sparse>
#include <memory>

namespace mhs::sim {

    enum class SolverType { Pardiso, EigenSparseLU, EigenBiCGSTAB };

    // LinearSolver configuration
    struct SolverConfig {
        // EigenBiCGSTAB-only knobs.
        double tolerance = 1e-8;
        int max_iterations = 1000;
    };

    struct SolverSpec {
        SolverType type = SolverType::Pardiso;
        SolverConfig config {};
    };

    class LinearSolver {
    public:
        virtual ~LinearSolver() = default;

        /// Build factorization (direct) or preconditioner (iterative).
        /// Must be called before solve(b).
        virtual void compute(const Eigen::SparseMatrix<double>& A) = 0;

        /// Solve A * x = b. Requires compute(A) to have been called first.
        virtual Eigen::VectorXd solve(const Eigen::VectorXd& b) = 0;

        // Configuration
        void set_config(SolverConfig cfg) { config_ = cfg; }
        const SolverConfig& config() const { return config_; }

        // Diagnostics from the last solve()
        bool success() const { return success_; }
        int iterations() const { return iterations_; }
        double residual() const { return residual_; }

        // Factory: build a solver from a spec.
        static std::unique_ptr<LinearSolver> create(const SolverSpec& spec = {});

    protected:
        SolverConfig config_;
        bool success_ = false;
        int iterations_ = 0;
        double residual_ = 0.0;
    };

} // namespace mhs::sim
