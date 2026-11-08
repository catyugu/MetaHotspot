#include "linear/solver.hpp"
#include <Eigen/SparseLU>
#include <cmath>
#include <stdexcept>

namespace mhs::linear {
    namespace {
        // Fallback direct solver for mixed Stokes/DAE operators, which are
        // indefinite and cannot be passed to the SPD AMG/CG path.
        class SparseLUSolver final : public Solver {
        public:
            explicit SparseLUSolver(SolverConfig config) : Solver(config) { }

            void compute(core::CSRMatrixView<double> matrix) override
            {
                if (!matrix.rows || matrix.rows != matrix.cols)
                    throw std::invalid_argument("SparseLU requires a nonempty square matrix");
                using RowMajor = Eigen::SparseMatrix<double, Eigen::RowMajor, core::SparseIndex>;
                using Index = Eigen::Index;
                Eigen::Map<const RowMajor> mapped(static_cast<Index>(matrix.rows), static_cast<Index>(matrix.cols),
                    static_cast<Index>(matrix.values.size()), matrix.row_ptr.data(), matrix.col_idx.data(),
                    matrix.values.data());
                A_ = mapped;
                A_.makeCompressed();
                lu_.analyzePattern(A_);
                if (lu_.info() != Eigen::Success)
                    throw std::runtime_error("SparseLU symbolic analysis failed");
                lu_.factorize(A_);
                ready_ = lu_.info() == Eigen::Success;
                if (!ready_)
                    throw std::runtime_error("SparseLU factorization failed (singular system or missing gauge)");
            }

            SolveResult solve(std::span<const double> b, std::span<double> x) override
            {
                if (!ready_ || b.size() != x.size() || b.size() != static_cast<std::size_t>(A_.rows()))
                    throw std::invalid_argument("SparseLU solve called with invalid state or dimensions");
                const auto n = static_cast<Eigen::Index>(b.size());
                Eigen::Map<const Eigen::VectorXd> rhs(b.data(), n);
                Eigen::Map<Eigen::VectorXd> result(x.data(), n);
                result = lu_.solve(rhs);
                const bool success = lu_.info() == Eigen::Success && result.allFinite();
                const double normalizer = std::max(1.0, rhs.norm());
                const double residual = success ? (A_ * result - rhs).norm() / normalizer : INFINITY;
                return {success && std::isfinite(residual), 1, residual};
            }

        private:
            using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, core::SparseIndex>;
            Sparse A_;
            Eigen::SparseLU<Sparse> lu_;
            bool ready_ = false;
        };
    }

    SolverPtr make_sparse_lu_solver(SolverConfig config) { return std::make_unique<SparseLUSolver>(config); }
} // namespace mhs::linear
