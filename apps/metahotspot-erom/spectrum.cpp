#include "amg_solver.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>

namespace mhs::app::erom {
    namespace {
        // Twice-reorthogonalized vectors keep the small Rayleigh-Ritz problems
        // ordinary symmetric eigenproblems, including rank-deficient restarts.
        bool append(Matrix& basis, Vector vector)
        {
            const double norm = vector.norm();
            for (int pass = 0; pass < 2; ++pass)
                vector -= basis * (basis.transpose() * vector);
            const double remaining = vector.norm();
            if (remaining <= std::numeric_limits<double>::epsilon() * vector.size() * norm)
                return false;
            const auto col = basis.cols();
            basis.conservativeResize(Eigen::NoChange, col + 1);
            basis.col(col) = vector / remaining;
            return true;
        }

        double smallest(const Sparse& matrix)
        {
            if (matrix.rows() <= 3) {
                Eigen::SelfAdjointEigenSolver<Matrix> eig {Matrix(matrix)};
                if (eig.info() != Eigen::Success)
                    throw std::runtime_error("global spectral eigensolve failed");
                return eig.eigenvalues()[0];
            }
            ReusableAmg preconditioner(matrix);
            Vector x = Vector::Constant(matrix.rows(), 1 / std::sqrt(static_cast<double>(matrix.rows())));
            Vector direction = Vector::Zero(x.size());
            for (int iteration = 0; iteration < 256; ++iteration) {
                const Vector image = matrix * x;
                const double lambda = x.dot(image);
                const Vector residual = image - lambda * x;
                if (residual.norm() <= 1e-8 * image.norm())
                    return lambda;
                Matrix trial(x.size(), 0);
                append(trial, x);
                append(trial, preconditioner.apply(residual));
                append(trial, direction);
                const Matrix projected = trial.transpose() * (matrix * trial);
                Eigen::SelfAdjointEigenSolver<Matrix> eig(projected);
                if (eig.info() != Eigen::Success)
                    break;
                Vector next = trial * eig.eigenvectors().col(0);
                direction = next - x.dot(next) * x;
                x = std::move(next);
            }
            throw std::runtime_error("global low-end preconditioned eigensolve did not converge");
        }

        double largest(const Sparse& matrix)
        {
            const Eigen::Index cap = std::min<Eigen::Index>(64, matrix.rows());
            std::mt19937_64 random(0);
            std::normal_distribution<double> normal;
            Vector seed = Vector::NullaryExpr(matrix.rows(), [&] { return normal(random); });
            seed.normalize();
            for (int restart = 0; restart < 32; ++restart) {
                Matrix basis(matrix.rows(), 0);
                append(basis, seed);
                for (Eigen::Index i = 0; i + 1 < cap; ++i)
                    if (!append(basis, matrix * basis.col(i)))
                        break;
                const Matrix projected = basis.transpose() * (matrix * basis);
                Eigen::SelfAdjointEigenSolver<Matrix> eig(projected);
                if (eig.info() != Eigen::Success)
                    break;
                const double lambda = eig.eigenvalues().tail(1)[0];
                seed = basis * eig.eigenvectors().rightCols(1);
                const Vector image = matrix * seed;
                if ((image - lambda * seed).norm() <= 1e-8 * image.norm())
                    return lambda;
            }
            throw std::runtime_error("global high-end Lanczos eigensolve did not converge");
        }

        // Descending arithmetic-geometric means evaluate Jacobi dn using the
        // complementary modulus directly. This avoids rounding k to 1 when the
        // thermal spectrum spans many decades.
        double dn(double fraction, double complement)
        {
            double a = 1, b = complement;
            core::Vector<double> ratios;
            while (std::abs(a - b) > 4 * std::numeric_limits<double>::epsilon() * a) {
                const double next = (a + b) * 0.5;
                ratios.push_back((a - b) / (2 * next));
                b = std::sqrt(a * b);
                a = next;
            }
            if (ratios.empty())
                return 1;
            double phi = std::ldexp(fraction * std::numbers::pi / 2, static_cast<int>(ratios.size()));
            double previous = phi;
            for (auto it = ratios.rbegin(); it != ratios.rend(); ++it) {
                previous = phi;
                phi = (phi + std::asin(std::clamp(*it * std::sin(phi), -1.0, 1.0))) * 0.5;
            }
            return std::cos(phi) / std::cos(previous - phi);
        }
    } // namespace

    core::Vector<double> matching_shifts(
        const Sparse& low, const Sparse& high, const Vector& capacity, double tolerance, Statistics& statistics)
    {
        const Vector scale = capacity.array().sqrt().inverse();
        auto normalized = [&](const Sparse& matrix) {
            Sparse result = matrix;
            for (int row = 0; row < result.outerSize(); ++row)
                for (Sparse::InnerIterator entry(result, row); entry; ++entry)
                    entry.valueRef() *= scale[entry.row()] * scale[entry.col()];
            return result;
        };
        const double lo = smallest(normalized(low)), hi = largest(normalized(high));
        if (!std::isfinite(lo) || !std::isfinite(hi) || lo <= 0 || hi < lo * (1 - 1e-8))
            throw std::runtime_error("HTC training operators must have a finite positive global spectrum");
        statistics.lambda_min = lo;
        statistics.lambda_max = std::max(lo, hi);
        const double count = std::ceil((std::log(4.0) - std::log(tolerance))
            * (std::log(4.0) + std::log(statistics.lambda_max) - std::log(lo)) / (std::numbers::pi * std::numbers::pi));
        if (!std::isfinite(count) || count > 100000)
            throw std::runtime_error("invalid global matching-point count");
        statistics.shifts = std::max(1, static_cast<int>(count));
        core::Vector<double> shifts;
        const double complement = lo / statistics.lambda_max;
        for (int i = 0; i < statistics.shifts; ++i) {
            const double fraction = (2.0 * i + 1) / (2.0 * statistics.shifts);
            const double shift = statistics.lambda_max * dn(fraction, complement);
            if (!std::isfinite(shift) || shift <= 0)
                throw std::runtime_error("invalid real matching point");
            shifts.push_back(shift);
        }
        std::ranges::sort(shifts, std::greater<>());
        return shifts;
    }
} // namespace mhs::app::erom
