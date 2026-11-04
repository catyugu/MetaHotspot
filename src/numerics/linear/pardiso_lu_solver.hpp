#pragma once

#include "numerics/linear/linear_solver.hpp"
#include <Eigen/PardisoSupport>

namespace mhs::numerics {
    class PardisoLUSolver final : public LinearSolver {
    public:
        void compute(const core::CSRMatrix<double>& A) override;
        LinearSolveResult solve(std::span<const double> b, std::span<const double> x0, std::span<double> x) override;

    private:
        using Sparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;
        Eigen::PardisoLU<Sparse> solver_;
        Sparse A_;
        bool analyzed_ = false;
    };
} // namespace mhs::numerics
