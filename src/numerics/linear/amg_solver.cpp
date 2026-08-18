#include "numerics/linear/amg_solver.hpp"

// AMGCL is a large third-party header template library. Its builtin backend
// normally compiles warning-clean, but it is instantiated inside this TU, so
// suppress every warning class before pulling the headers in — this keeps the
// project's strict /WX / -Werror flags from tripping on AMGCL templates (the
// whole subsystem is compiled separately below and never gets mhs_options).
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244 4267 4100 4127 4389 4456 4457 4458 4459 4701 4702 4703 4996 4242 6287)
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wextra"
#pragma clang diagnostic ignored "-Wpedantic"
#pragma clang diagnostic ignored "-Wunused-parameter"
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <amgcl/adapter/eigen.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/gmres.hpp>

// Let the builtin backend operate on Eigen::VectorXd directly (zero copy).
AMGCL_USE_EIGEN_VECTORS_WITH_BUILTIN_BACKEND()

#include <Eigen/Sparse>
#include <exception>
#include <stdexcept>
#include <tuple>

namespace mhs::sim {

    namespace {

        using Backend = amgcl::backend::builtin<double>;
        using Precond = amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>;

        // Symmetric -> CG (fastest for SPD); non-symmetric -> restarted GMRES.
        using CgSolver = amgcl::make_solver<Precond, amgcl::solver::cg<Backend>>;
        using GmresSolver = amgcl::make_solver<Precond, amgcl::solver::gmres<Backend>>;

        // GMRES restart / Krylov-subspace budget.
        constexpr unsigned kGmresRestart = 64;

    } // namespace

    struct AmgCgSolver::Impl {
        bool symmetric = true;
        int max_iterations = 1000;
        std::shared_ptr<CgSolver> cg;
        std::shared_ptr<GmresSolver> gmres;
    };

    AmgCgSolver::AmgCgSolver() : impl_(std::make_unique<Impl>()) { }

    AmgCgSolver::~AmgCgSolver() = default;

    void AmgCgSolver::compute(const Eigen::SparseMatrix<double>& A)
    {
        // Detect symmetry once: pure conduction/convection stays symmetric,
        // fluid-coupled cases with upwind advection introduce asymmetric
        // off-diagonal terms (fluid_assembler). CG needs SPD, so route any
        // non-symmetric operator to GMRES.
        const double asym = (A - Eigen::SparseMatrix<double>(A.transpose())).norm();
        const double scale = A.norm();
        impl_->symmetric = (asym <= 1e-6 * scale + 1e-300);
        impl_->max_iterations = config_.max_iterations;

        // AMGCL reads the operator by rows, so hand it a row-major copy. The
        // AMG hierarchy is built eagerly here and the copy can go out of scope.
        Eigen::SparseMatrix<double, Eigen::RowMajor> row_major(A);
        row_major.makeCompressed();

        try {
            if (impl_->symmetric) {
                CgSolver::params prm;
                prm.solver.tol = config_.tolerance;
                prm.solver.maxiter = static_cast<std::size_t>(config_.max_iterations);
                impl_->cg = std::make_shared<CgSolver>(row_major, prm);
                impl_->gmres.reset();
            }
            else {
                GmresSolver::params prm;
                prm.solver.tol = config_.tolerance;
                prm.solver.maxiter = static_cast<unsigned>(config_.max_iterations);
                prm.solver.M = kGmresRestart;
                impl_->gmres = std::make_shared<GmresSolver>(row_major, prm);
                impl_->cg.reset();
            }
        }
        catch (const std::exception& e) {
            impl_->cg.reset();
            impl_->gmres.reset();
            success_ = false;
            iterations_ = 0;
            residual_ = 0.0;
            throw std::runtime_error(std::string("AMG preconditioner setup failed: ") + e.what());
        }
    }

    Eigen::VectorXd AmgCgSolver::solve(const Eigen::VectorXd& b, Eigen::Ref<const Eigen::VectorXd> x0)
    {
        const bool have_solver = impl_->symmetric ? static_cast<bool>(impl_->cg) : static_cast<bool>(impl_->gmres);
        if (!have_solver) {
            throw std::logic_error("AmgCgSolver::solve called before compute()");
        }
        if (b.size() != x0.size()) {
            throw std::invalid_argument("AmgCgSolver::solve: initial guess size must match b ("
                + std::to_string(x0.size()) + " != " + std::to_string(b.size()) + ")");
        }

        Eigen::VectorXd x = x0; // warm start
        std::size_t iters = 0;
        double residual = 0.0;
        try {
            if (impl_->symmetric) {
                std::tie(iters, residual) = (*impl_->cg)(b, x);
            }
            else {
                std::tie(iters, residual) = (*impl_->gmres)(b, x);
            }
        }
        catch (const std::exception& e) {
            success_ = false;
            iterations_ = 0;
            residual_ = 0.0;
            throw std::runtime_error(std::string("AMG solve failed: ") + e.what());
        }

        iterations_ = static_cast<int>(iters);
        residual_ = residual;
        // Converged unless the iteration budget was exhausted (fall back to a
        // residual check so that convergence on the very last step still counts).
        const double b_norm = b.norm();
        success_ = (static_cast<int>(iters) < impl_->max_iterations) || residual <= config_.tolerance * b_norm + 1e-300;
        return x;
    }

} // namespace mhs::sim
