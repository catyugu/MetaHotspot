#include "linear/solver.hpp"
#include <Eigen/PardisoSupport>
#include <algorithm>
#include <stdexcept>

namespace mhs::linear {
    namespace {
        class PardisoLUSolver final : public Solver {
        public:
            explicit PardisoLUSolver(SolverConfig config) : Solver(config) { }
            void compute(core::CSRMatrixView<double> A) override;
            SolveResult solve(std::span<const double> b, std::span<double> x) override;

        private:
            using Sparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;
            Eigen::PardisoLU<Sparse> solver_;
            Sparse A_;
            bool analyzed_ = false;
        };

        void PardisoLUSolver::compute(core::CSRMatrixView<double> A)
        {
            if (A.rows != A.cols || A.rows == 0)
                throw std::invalid_argument("PARDISO requires a nonempty square matrix");
            const bool reuse = analyzed_ && A_.rows() == static_cast<Eigen::Index>(A.rows)
                && A_.cols() == static_cast<Eigen::Index>(A.cols)
                && A_.nonZeros() == static_cast<Eigen::Index>(A.values.size())
                && std::equal(A.row_ptr.begin(), A.row_ptr.end(), A_.outerIndexPtr())
                && std::equal(A.col_idx.begin(), A.col_idx.end(), A_.innerIndexPtr());
            Eigen::Map<const Sparse> mapped(static_cast<Eigen::Index>(A.rows), static_cast<Eigen::Index>(A.cols),
                static_cast<Eigen::Index>(A.values.size()), A.row_ptr.data(), A.col_idx.data(), A.values.data());
            A_ = mapped;
            if (!reuse)
                solver_.analyzePattern(A_);
            solver_.factorize(A_);
            analyzed_ = solver_.info() == Eigen::Success;
        }

        SolveResult PardisoLUSolver::solve(std::span<const double> b, std::span<double> x)
        {
            if (b.size() != static_cast<std::size_t>(A_.rows()) || x.size() != b.size())
                throw std::invalid_argument("PARDISO solve dimension mismatch");
            if (!analyzed_)
                return {};
            const auto n = static_cast<Eigen::Index>(b.size());
            Eigen::Map<const Eigen::VectorXd> rhs(b.data(), n);
            Eigen::Map<Eigen::VectorXd> solution(x.data(), n);
            solution = solver_.solve(rhs);
            const bool success = solver_.info() == Eigen::Success;
            const double scale = rhs.norm();
            return {success, 1, success ? (A_ * solution - rhs).norm() / (scale > 0.0 ? scale : 1.0) : 0.0};
        }
    } // namespace

    SolverPtr make_pardiso_solver(SolverConfig config) { return std::make_unique<PardisoLUSolver>(config); }
} // namespace mhs::linear
