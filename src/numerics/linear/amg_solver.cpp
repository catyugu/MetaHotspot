#include "amg_solver.hpp"
#include "numerics/sparse_ops.hpp"
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/gmres.hpp>

#include <algorithm>
#include <exception>
#include <omp.h>
#include <stdexcept>
#include <tuple>

// Enable AMGCL's existing indexed builtin kernels for borrowed contiguous
// buffers. No allocation or copy is required at the solve boundary.
namespace amgcl::backend {
    template <class T, std::size_t Extent> struct is_builtin_vector<std::span<T, Extent>> : std::is_arithmetic<T> { };
}

namespace mhs::numerics {

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

    } // namespace

    struct AmgCgSolver::Impl {
        bool symmetric = true;
        std::ptrdiff_t rows = 0;
        std::shared_ptr<CgSolver> cg;
        std::shared_ptr<GmresSolver> gmres;
    };

    AmgCgSolver::AmgCgSolver() : impl_(std::make_unique<Impl>()) { }

    AmgCgSolver::~AmgCgSolver() = default;

    void AmgCgSolver::compute(const core::CSRMatrix<double>& A)
    {
        // Detect symmetry once: pure conduction/convection stays symmetric,
        // fluid-coupled cases with upwind advection introduce asymmetric
        // off-diagonal terms (thermal assembly). CG needs SPD, so route any
        // non-symmetric operator to GMRES.
        impl_->symmetric = numerically_symmetric(A, 1e-6);
        impl_->rows = A.rows();
        const ScopedTeam team(impl_->rows);

        if (A.rows() != A.cols() || A.rows() == 0)
            throw std::invalid_argument("AMG requires a nonempty square matrix");
        const auto matrix = std::make_tuple(A.rows(), A.row_ptr(), A.col_idx(), A.values());

        try {
            if (impl_->symmetric) {
                CgSolver::params prm;
                prm.solver.tol = config_.tolerance;
                prm.solver.maxiter = static_cast<std::size_t>(config_.max_iterations);
                impl_->cg = std::make_shared<CgSolver>(matrix, prm);
                impl_->gmres.reset();
            }
            else {
                GmresSolver::params prm;
                prm.solver.tol = config_.tolerance;
                prm.solver.maxiter = static_cast<unsigned>(config_.max_iterations);
                prm.solver.M = kGmresRestart;
                impl_->gmres = std::make_shared<GmresSolver>(matrix, prm);
                impl_->cg.reset();
            }
        }
        catch (const std::exception& e) {
            impl_->cg.reset();
            impl_->gmres.reset();
            throw std::runtime_error(std::string("AMG preconditioner setup failed: ") + e.what());
        }
    }

    LinearSolveResult AmgCgSolver::solve(std::span<const double> b, std::span<const double> x0, std::span<double> x)
    {
        const bool have_solver = impl_->symmetric ? static_cast<bool>(impl_->cg) : static_cast<bool>(impl_->gmres);
        if (!have_solver) {
            throw std::logic_error("AmgCgSolver::solve called before compute()");
        }
        if (b.size() != x0.size() || b.size() != x.size() || b.size() != static_cast<std::size_t>(impl_->rows)) {
            throw std::invalid_argument("AmgCgSolver::solve: initial guess size must match b ("
                + std::to_string(x0.size()) + " != " + std::to_string(b.size()) + ")");
        }

        if (x.data() != x0.data())
            std::copy(x0.begin(), x0.end(), x.begin());
        std::size_t iters = 0;
        double residual = 0.0;
        const ScopedTeam team(impl_->rows);
        try {
            if (impl_->symmetric) {
                std::tie(iters, residual) = (*impl_->cg)(b, x);
            }
            else {
                std::tie(iters, residual) = (*impl_->gmres)(b, x);
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

} // namespace mhs::numerics
