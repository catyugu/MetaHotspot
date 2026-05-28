#pragma once

#include <Eigen/Sparse>
#include <string>

namespace mhs {

    enum class SolverType {
        SparseLU,
        BiCGSTAB
    };

    class Solver {
    public:
        virtual ~Solver() = default;

        virtual void analyzePattern(const Eigen::SparseMatrix<double>& A) = 0;
        virtual void factorize(const Eigen::SparseMatrix<double>& A) = 0;
        virtual Eigen::VectorXd solve(const Eigen::VectorXd& b) = 0;

        static std::unique_ptr<Solver> create(SolverType type);
    };

} // namespace mhs