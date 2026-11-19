#include "amg_solver.hpp"

#include <algorithm>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <cmath>
#include <omp.h>
#include <span>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace amgcl::backend {
    template <class T, std::size_t Extent> struct is_builtin_vector<std::span<T, Extent>> : std::is_arithmetic<T> { };
}

namespace mhs::app::erom {
    namespace {
        auto csr(const Sparse& matrix)
        {
            return std::make_tuple(static_cast<std::size_t>(matrix.rows()),
                std::span<const int>(matrix.outerIndexPtr(), static_cast<std::size_t>(matrix.rows()) + 1),
                std::span<const int>(matrix.innerIndexPtr(), static_cast<std::size_t>(matrix.nonZeros())),
                std::span<const double>(matrix.valuePtr(), static_cast<std::size_t>(matrix.nonZeros())));
        }

        class ScopedTeam {
        public:
            explicit ScopedTeam(Eigen::Index rows) : previous_(omp_get_max_threads())
            {
                omp_set_num_threads(
                    static_cast<int>(std::clamp<Eigen::Index>((rows + 8191) / 8192, 1, std::min(6, previous_))));
            }
            ~ScopedTeam() { omp_set_num_threads(previous_); }

        private:
            int previous_;
        };
    } // namespace

    struct ReusableAmg::Impl {
        using Backend = amgcl::backend::builtin<double>;
        using Hierarchy = amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>;
        using Cg = amgcl::solver::cg<Backend>;
        Hierarchy hierarchy;
        Cg cg;
        Eigen::Index rows;
        double tolerance;
        mutable std::size_t iterations = 0;

        static Cg::params parameters(double tolerance)
        {
            Cg::params result;
            result.tol = tolerance * 0.1;
            result.maxiter = 2000;
            return result;
        }
        Impl(const Sparse& matrix, double tol)
            : hierarchy(csr(matrix))
            , cg(static_cast<std::size_t>(matrix.rows()), parameters(tol))
            , rows(matrix.rows())
            , tolerance(tol)
        {
        }
    };

    ReusableAmg::ReusableAmg(const Sparse& reference, double tolerance)
    {
        if (!reference.isCompressed() || reference.rows() == 0 || reference.rows() != reference.cols()
            || !std::isfinite(tolerance) || tolerance <= 0 || tolerance >= 1)
            throw std::invalid_argument("invalid extraction AMG matrix or tolerance");
        const ScopedTeam team(reference.rows());
        impl_ = std::make_unique<Impl>(reference, tolerance);
    }
    ReusableAmg::~ReusableAmg() = default;

    Vector ReusableAmg::apply(const Vector& rhs) const
    {
        if (rhs.size() != impl_->rows)
            throw std::invalid_argument("extraction AMG dimension mismatch");
        const ScopedTeam team(impl_->rows);
        Vector result(rhs.size());
        impl_->hierarchy.apply(std::span<const double>(rhs.data(), static_cast<std::size_t>(rhs.size())),
            std::span<double>(result.data(), static_cast<std::size_t>(result.size())));
        return result;
    }

    Vector ReusableAmg::solve(const Sparse& matrix, const Vector& rhs, Vector initial) const
    {
        if (rhs.size() != impl_->rows || initial.size() != rhs.size() || matrix.rows() != rhs.size()
            || matrix.cols() != rhs.size() || !matrix.isCompressed() || !rhs.allFinite() || !initial.allFinite())
            throw std::invalid_argument("extraction AMG-CG dimension mismatch or nonfinite input");
        const double norm = rhs.norm();
        if (norm == 0)
            return Vector::Zero(rhs.size());
        // Normalize even tiny physical loads so AMGCL's absolute zero-RHS cutoff
        // cannot silently discard a valid temperature response.
        const Vector scaled_rhs = rhs / norm;
        initial /= norm;
        const ScopedTeam team(impl_->rows);
        auto iterate = [&](const Vector& b, Vector& x) {
            const auto [iterations, reported] = impl_->cg(csr(matrix), impl_->hierarchy,
                std::span<const double>(b.data(), static_cast<std::size_t>(b.size())),
                std::span<double>(x.data(), static_cast<std::size_t>(x.size())));
            impl_->iterations += iterations;
            if (!std::isfinite(reported))
                throw std::runtime_error("nonfinite extraction AMG-CG residual");
        };
        iterate(scaled_rhs, initial);
        double residual = 0;
        for (int refinement = 0; refinement < 4; ++refinement) {
            const Vector defect = scaled_rhs - matrix * initial;
            residual = defect.norm();
            if (std::isfinite(residual) && residual <= impl_->tolerance)
                return initial * norm;
            if (!std::isfinite(residual) || refinement == 3)
                break;
            // Recursive CG residuals can drift on weakly anchored thermal
            // operators. Correct the explicitly recomputed defect, reusing P.
            const Vector b = defect / residual;
            Vector correction = Vector::Zero(rhs.size());
            iterate(b, correction);
            initial += residual * correction;
        }
        std::ostringstream message;
        message << "extraction AMG-CG did not converge: true relative residual " << residual << ", required "
                << impl_->tolerance;
        throw std::runtime_error(message.str());
    }

    std::size_t ReusableAmg::iterations() const { return impl_->iterations; }
} // namespace mhs::app::erom
