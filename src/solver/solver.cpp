#include "solver.hpp"
#include <Eigen/Sparse>

namespace mhs {

    class SparseLUSolver : public Solver {
    public:
        void analyzePattern(const Eigen::SparseMatrix<double>& A) override;
        void factorize(const Eigen::SparseMatrix<double>& A) override;
        Eigen::VectorXd solve(const Eigen::VectorXd& b) override;

    private:
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_;
    };

    class BiCGSTABSolver : public Solver {
    public:
        void analyzePattern(const Eigen::SparseMatrix<double>& A) override;
        void factorize(const Eigen::SparseMatrix<double>& A) override;
        Eigen::VectorXd solve(const Eigen::VectorXd& b) override;

    private:
        Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver_;
    };

    void SparseLUSolver::analyzePattern(const Eigen::SparseMatrix<double>& A)
    {
        solver_.analyzePattern(A);
    }

    void SparseLUSolver::factorize(const Eigen::SparseMatrix<double>& A)
    {
        solver_.factorize(A);
    }

    Eigen::VectorXd SparseLUSolver::solve(const Eigen::VectorXd& b)
    {
        return solver_.solve(b);
    }

    void BiCGSTABSolver::analyzePattern(const Eigen::SparseMatrix<double>& A)
    {
        (void)A;
    }

    void BiCGSTABSolver::factorize(const Eigen::SparseMatrix<double>& A)
    {
        (void)A;
    }

    Eigen::VectorXd BiCGSTABSolver::solve(const Eigen::VectorXd& b)
    {
        return Eigen::VectorXd::Zero(b.size());
    }

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