#include "linear/solver.hpp"
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/gmres.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <omp.h>
#include <stdexcept>
#include <tuple>

// Enable AMGCL's existing indexed builtin kernels for borrowed contiguous
// buffers. No allocation or copy is required at the solve boundary.
namespace amgcl::backend {
    template <class T, std::size_t Extent> struct is_builtin_vector<std::span<T, Extent>> : std::is_arithmetic<T> { };
}

namespace mhs::linear {

    namespace {

        using Backend = amgcl::backend::builtin<double>;
        using Precond = amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>;

        // Symmetric -> CG (fastest for SPD); non-symmetric -> restarted GMRES.
        using CgSolver = amgcl::make_solver<Precond, amgcl::solver::cg<Backend>>;
        using GmresSolver = amgcl::make_solver<Precond, amgcl::solver::gmres<Backend>>;

        // GMRES restart / Krylov-subspace budget.
        constexpr unsigned kGmresRestart = 64;

        // The performance sweet point
        constexpr std::ptrdiff_t kRowsPerThread = 8192;
        constexpr int kMaxThreads = 6;

        /// amgcl reads the team size from OpenMP itself, so the only handle on it
        /// is the calling thread's setting. The previous value is restored on
        /// scope exit, leaving MKL's own threading untouched.
        class ScopedTeam {
        public:
            explicit ScopedTeam(std::ptrdiff_t rows) : previous_(omp_get_max_threads())
            {
                const std::ptrdiff_t wanted = (rows + kRowsPerThread - 1) / kRowsPerThread;
                const std::ptrdiff_t limit = std::min<std::ptrdiff_t>(kMaxThreads, previous_);
                omp_set_num_threads(static_cast<int>(std::clamp(wanted, std::ptrdiff_t {1}, limit)));
            }
            ScopedTeam(const ScopedTeam&) = delete;
            ScopedTeam& operator=(const ScopedTeam&) = delete;
            ~ScopedTeam() { omp_set_num_threads(previous_); }

        private:
            int previous_;
        };

        bool numerically_symmetric(core::CSRMatrixView<double> a, double tolerance)
        {
            if (a.rows != a.cols)
                return false;
            double asym2 = 0, scale2 = 0;
            for (std::size_t i = 0; i < a.rows; ++i)
                for (auto k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
                    const auto j = static_cast<std::size_t>(a.col_idx[k]);
                    const double v = a.values[k];
                    const auto columns = a.col_idx.subspan(static_cast<std::size_t>(a.row_ptr[j]),
                        static_cast<std::size_t>(a.row_ptr[j + 1] - a.row_ptr[j]));
                    const auto pos = std::lower_bound(columns.begin(), columns.end(), i);
                    const bool present = pos != columns.end() && static_cast<std::size_t>(*pos) == i;
                    const double transpose = present ? a.values[a.row_ptr[j] + (pos - columns.begin())] : 0.0;
                    scale2 += v * v;
                    const double d = v - transpose;
                    asym2 += d * d;
                    // The missing transpose is a second entry of A - A^T.
                    if (!present)
                        asym2 += v * v;
                }
            return std::sqrt(asym2) <= tolerance * std::sqrt(scale2) + 1e-300;
        }

        class AmgSolver final : public Solver {
        public:
            explicit AmgSolver(SolverConfig config) : Solver(config) { }
            void compute(core::CSRMatrixView<double> A) override;
            SolveResult solve(std::span<const double> b, std::span<double> x) override;

        private:
            bool symmetric_ = true;
            std::ptrdiff_t rows_ = 0;
            std::unique_ptr<CgSolver> cg_;
            std::unique_ptr<GmresSolver> gmres_;
        };

        void AmgSolver::compute(core::CSRMatrixView<double> A)
        {
            // Detect symmetry once: pure conduction/convection stays symmetric,
            // fluid-coupled cases with upwind advection introduce asymmetric
            // off-diagonal terms (thermal assembly). CG needs SPD, so route any
            // non-symmetric operator to GMRES.
            symmetric_ = numerically_symmetric(A, 1e-6);
            rows_ = static_cast<std::ptrdiff_t>(A.rows);
            const ScopedTeam team(rows_);

            if (A.rows != A.cols || A.rows == 0)
                throw std::invalid_argument("AMG requires a nonempty square matrix");
            const auto matrix = std::make_tuple(A.rows, A.row_ptr, A.col_idx, A.values);

            try {
                if (symmetric_) {
                    CgSolver::params prm;
                    prm.solver.tol = config_.tolerance;
                    prm.solver.maxiter = static_cast<std::size_t>(config_.max_iterations);
                    cg_ = std::make_unique<CgSolver>(matrix, prm);
                    gmres_.reset();
                }
                else {
                    GmresSolver::params prm;
                    prm.solver.tol = config_.tolerance;
                    prm.solver.maxiter = static_cast<unsigned>(config_.max_iterations);
                    prm.solver.M = kGmresRestart;
                    gmres_ = std::make_unique<GmresSolver>(matrix, prm);
                    cg_.reset();
                }
            }
            catch (const std::exception& e) {
                cg_.reset();
                gmres_.reset();
                throw std::runtime_error(std::string("AMG preconditioner setup failed: ") + e.what());
            }
        }

        SolveResult AmgSolver::solve(std::span<const double> b, std::span<double> x)
        {
            const bool have_solver = symmetric_ ? static_cast<bool>(cg_) : static_cast<bool>(gmres_);
            if (!have_solver) {
                throw std::logic_error("AmgSolver::solve called before compute()");
            }
            if (b.size() != x.size() || b.size() != static_cast<std::size_t>(rows_))
                throw std::invalid_argument("AmgSolver::solve dimension mismatch");

            std::size_t iters = 0;
            double residual = 0.0;
            const ScopedTeam team(rows_);
            try {
                if (symmetric_) {
                    std::tie(iters, residual) = (*cg_)(b, x);
                }
                else {
                    std::tie(iters, residual) = (*gmres_)(b, x);
                }
            }
            catch (const std::exception& e) {
                throw std::runtime_error(std::string("AMG solve failed: ") + e.what());
            }

            // AMGCL reports the relative residual, including a zero-RHS fallback.
            return {std::isfinite(residual)
                    && (iters < static_cast<std::size_t>(config_.max_iterations) || residual <= config_.tolerance),
                static_cast<int>(iters), residual};
        }

    } // namespace

    SolverPtr make_amg_solver(SolverConfig config) { return std::make_unique<AmgSolver>(config); }
} // namespace mhs::linear
