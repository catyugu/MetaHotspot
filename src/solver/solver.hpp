#pragma once

#include <Eigen/Sparse>
#include <memory>

namespace mhs {

    enum class SolverType { Pardiso, SparseLU, BiCGSTAB };

    // Solver configuration
    struct SolverConfig {
        // Active solver. Defaults to the best available backend at build time.
        SolverType type =
#ifdef MHS_ENABLE_PARDISO
            SolverType::Pardiso;
#else
            SolverType::SparseLU;
#endif

        // BiCGSTAB-only knobs.
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

    // Base solver class (virtual interface)
    class Solver {
    public:
        virtual ~Solver() = default;

        // Solve A * x = b
        virtual SolveResult solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b) = 0;

        // Inject configuration before solve()
        virtual void set_config(const SolverConfig& cfg) = 0;

        // Factory method
        static std::unique_ptr<Solver> create(SolverType type);
    };

} // namespace mhs