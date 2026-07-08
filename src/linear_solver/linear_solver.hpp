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
        SolverType type =
#ifdef MHS_ENABLE_PARDISO
            SolverType::Pardiso;
#else
            SolverType::EigenSparseLU;
#endif
        SolverConfig config {};
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
    //
    // Every concrete solver needs a SolverConfig before solve(), so the config
    // lives on the base. Subclasses read `config_` directly.
    class LinearSolver {
    public:
        virtual ~LinearSolver() = default;

        // Solve A * x = b
        virtual SolveResult solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b) = 0;

        // Replace the configuration used by solve(). Called by the factory with
        // the spec-supplied config; subclasses read the base's `config_`.
        void set_config(SolverConfig cfg) { config_ = cfg; }
        const SolverConfig& config() const { return config_; }

        // Factory: build a solver from a spec. Type + config both default
        // (EigenSparseLU + SolverConfig defaults), so `LinearSolver::create({})`
        // returns the default solver with default config.
        static std::unique_ptr<LinearSolver> create(const SolverSpec& spec = {});

    protected:
        SolverConfig config_;
    };

} // namespace mhs::sim
