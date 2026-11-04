#include "pardiso_lu_solver.hpp"
#include <algorithm>
#include <stdexcept>

namespace mhs::numerics {
    void PardisoLUSolver::compute(const core::CSRMatrix<double>& A)
    {
        if (A.rows() != A.cols() || A.rows() == 0)
            throw std::invalid_argument("PARDISO requires a nonempty square matrix");
        const bool reuse = analyzed_ && A_.rows() == static_cast<Eigen::Index>(A.rows())
            && A_.cols() == static_cast<Eigen::Index>(A.cols()) && A_.nonZeros() == static_cast<Eigen::Index>(A.nnz())
            && std::equal(A.row_ptr().begin(), A.row_ptr().end(), A_.outerIndexPtr())
            && std::equal(A.col_idx().begin(), A.col_idx().end(), A_.innerIndexPtr());
        Eigen::Map<const Sparse> mapped(static_cast<Eigen::Index>(A.rows()), static_cast<Eigen::Index>(A.cols()),
            static_cast<Eigen::Index>(A.nnz()), A.row_ptr().data(), A.col_idx().data(), A.values().data());
        A_ = mapped;
        if (!reuse)
            solver_.analyzePattern(A_);
        solver_.factorize(A_);
        analyzed_ = solver_.info() == Eigen::Success;
    }

    LinearSolveResult PardisoLUSolver::solve(std::span<const double> b, std::span<const double> x0, std::span<double> x)
    {
        if (b.size() != static_cast<std::size_t>(A_.rows()) || x0.size() != b.size() || x.size() != b.size())
            throw std::invalid_argument("PARDISO solve dimension mismatch");
        if (!analyzed_)
            return {};
        const auto n = static_cast<Eigen::Index>(b.size());
        Eigen::Map<const Eigen::VectorXd> rhs(b.data(), n);
        Eigen::Map<Eigen::VectorXd> solution(x.data(), n);
        solution = solver_.solve(rhs);
        const bool success = solver_.info() == Eigen::Success;
        return {success, 1, success ? (A_ * solution - rhs).norm() : 0.0};
    }
} // namespace mhs::numerics
