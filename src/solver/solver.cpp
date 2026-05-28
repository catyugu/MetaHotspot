#include "solver.hpp"
#include <Eigen/Sparse>

namespace mhs {

    // SparseLU solver implementation
    class SparseLUSolver : public Solver {
    public:
        SolveResult solve(const Eigen::SparseMatrix<double>& A,
                          const Eigen::VectorXd& b) override
        {
            Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
            solver.compute(A);

            if (solver.info() != Eigen::Success) {
                return {Eigen::VectorXd(), false, 0.0, 0};
            }

            Eigen::VectorXd x = solver.solve(b);

            return {
                x,
                solver.info() == Eigen::Success,
                (A * x - b).norm(),
                1  // direct solver, 1 iteration
            };
        }
    };

    // BiCGSTAB solver implementation
    class BiCGSTABSolver : public Solver {
    public:
        SolveResult solve(const Eigen::SparseMatrix<double>& A,
                          const Eigen::VectorXd& b) override
        {
            (void)A;  // A is used by the solver internally
            Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
            solver.setMaxIterations(config_.max_iterations);
            solver.setTolerance(config_.tolerance);

            Eigen::VectorXd x = solver.solve(b);

            return {
                x,
                solver.info() == Eigen::Success,
                solver.error(),
                static_cast<int>(solver.iterations())
            };
        }

        void set_config(const SolverConfig& config) { config_ = config; }

    private:
        SolverConfig config_;
    };

    std::unique_ptr<Solver> Solver::create(SolverType type)
    {
        switch (type) {
        case SolverType::SparseLU:
            return std::make_unique<SparseLUSolver>();
        case SolverType::BiCGSTAB:
            return std::make_unique<BiCGSTABSolver>();
        default:
            return std::make_unique<SparseLUSolver>();
        }
    }

} // namespace mhs