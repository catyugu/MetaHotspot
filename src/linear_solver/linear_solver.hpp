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

    // Solve result (no state on solver)
    struct SolveResult {
        Eigen::VectorXd solution;
        bool success;
        double residual_norm;
        int iterations;
    };

    // Base linear solver class (virtual interface).
    // Renamed from `Solver` to disambiguate from the nonlinear iteration
    // pathway (`mhs::sim::nonlinear_solve`).
    class LinearSolver {
    public:
        virtual ~LinearSolver() = default;

        // Solve A * x = b
        virtual SolveResult solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b) = 0;

        // Inject configuration before solve()
        virtual void set_config(const SolverConfig& cfg) = 0;

        // Factory method
        static std::unique_ptr<LinearSolver> create(SolverType type);
    };

} // namespace mhs::sim
