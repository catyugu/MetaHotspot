#include "amg_solver.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace mhs::app::erom {
    core::Vector<double> matching_shifts(
        const Sparse& low, const Sparse& high, const Vector& capacity, double tolerance, Statistics& statistics);
    Matrix compress_snapshots(const core::Vector<Vector>& snapshots, double tolerance);

    Sparse add_diagonal(const Sparse& matrix, const Vector& diagonal)
    {
        Sparse result = matrix;
        for (Eigen::Index row = 0; row < result.rows(); ++row)
            result.coeffRef(row, row) += diagonal[row];
        result.makeCompressed();
        return result;
    }

    Matrix solve_positive(const Matrix& matrix, const Matrix& rhs)
    {
        Eigen::LLT<Matrix> solver(matrix);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("reduced operator is not positive definite");
        Matrix result = solver.solve(rhs);
        if (!result.allFinite())
            throw std::runtime_error("nonfinite reduced response");
        return result;
    }
    Vector solve_positive(const Matrix& matrix, const Vector& rhs) { return solve_positive(matrix, Matrix(rhs)); }

    namespace {
        struct ResponseSpace {
            Matrix basis;
            core::Vector<Matrix> operators;
            Vector source;

            ResponseSpace(Eigen::Index n, Eigen::Index parameters) : basis(n, 0), operators(parameters + 2) { }

            Vector response(const Vector& h, double shift) const
            {
                Matrix reduced = operators[0] + shift * operators[1];
                for (Eigen::Index i = 0; i < h.size(); ++i)
                    reduced += h[i] * operators[static_cast<std::size_t>(i) + 2];
                return basis * solve_positive(reduced, source);
            }

            void append(Vector vector, const Sparse& K, const Vector& capacity, const Matrix& slopes, const Vector& g)
            {
                const double original = vector.norm();
                for (int pass = 0; pass < 2; ++pass)
                    vector -= basis * (basis.transpose() * vector);
                const double norm = vector.norm();
                if (norm <= std::numeric_limits<double>::epsilon() * vector.size() * original)
                    throw std::runtime_error("response enrichment stagnated before the residual tolerance was met");
                vector /= norm;
                const auto r = basis.cols();
                for (std::size_t i = 0; i < operators.size(); ++i) {
                    Vector image;
                    if (i == 0)
                        image = K * vector;
                    else if (i == 1)
                        image = capacity.cwiseProduct(vector);
                    else
                        image = slopes.col(static_cast<Eigen::Index>(i) - 2).cwiseProduct(vector);
                    const Vector edge = basis.transpose() * image;
                    auto& projected = operators[i];
                    projected.conservativeResize(r + 1, r + 1);
                    projected.topRightCorner(r, 1) = edge;
                    projected.bottomLeftCorner(1, r) = edge.transpose();
                    projected(r, r) = vector.dot(image);
                }
                source.conservativeResize(r + 1);
                source[r] = vector.dot(g);
                basis.conservativeResize(Eigen::NoChange, r + 1);
                basis.col(r) = vector;
            }
        };
    } // namespace

    Matrix build_basis(const Sparse& K, const Vector& capacity, const Matrix& slopes,
        const core::Vector<std::pair<double, double>>& ranges, const Matrix& inputs, const Options& options,
        Statistics& statistics)
    {
        const auto n = K.rows();
        Vector lower(slopes.cols()), upper(slopes.cols());
        for (Eigen::Index i = 0; i < slopes.cols(); ++i) {
            lower[i] = ranges[static_cast<std::size_t>(i)].first;
            upper[i] = ranges[static_cast<std::size_t>(i)].second;
        }
        const Sparse minimum = add_diagonal(K, slopes * lower), maximum = add_diagonal(K, slopes * upper);
        const auto shifts = matching_shifts(minimum, maximum, capacity, options.tolerance, statistics);
        core::Vector<ResponseSpace> spaces;
        for (Eigen::Index port = 0; port < inputs.cols(); ++port)
            spaces.emplace_back(n, slopes.cols());
        core::Vector<Vector> snapshots;
        std::mt19937_64 random(20260805);
        std::uniform_real_distribution<double> uniform;
        for (const double shift : shifts) {
            std::unique_ptr<ReusableAmg> solver;
            for (Eigen::Index port = 0; port < inputs.cols(); ++port) {
                const Vector g = inputs.col(port);
                const double norm = g.norm();
                if (norm == 0)
                    continue;
                auto& space = spaces[static_cast<std::size_t>(port)];
                int accepted = 0;
                while (accepted < 10) {
                    Vector h(lower.size());
                    for (Eigen::Index i = 0; i < h.size(); ++i)
                        h[i] = std::exp(
                            std::log(lower[i]) + uniform(random) * (std::log(upper[i]) - std::log(lower[i])));
                    const Vector diagonal = shift * capacity + slopes * h;
                    Vector initial = Vector::Zero(n);
                    if (space.basis.cols()) {
                        initial = space.response(h, shift);
                        const double residual = (K * initial + diagonal.cwiseProduct(initial) - g).norm() / norm;
                        if (std::isfinite(residual) && residual <= options.tolerance) {
                            ++accepted;
                            continue;
                        }
                    }
                    if (snapshots.size() >= static_cast<std::size_t>(options.max_order))
                        throw std::runtime_error("FANTASTIC extraction reached max_order");
                    if (!solver) {
                        solver = std::make_unique<ReusableAmg>(
                            add_diagonal(minimum, shift * capacity), options.tolerance / 10);
                        ++statistics.response_amg_setups;
                    }
                    Vector response = solver->solve(add_diagonal(K, diagonal), g, std::move(initial));
                    ++statistics.response_solves;
                    snapshots.push_back(response);
                    space.append(std::move(response), K, capacity, slopes, g);
                    accepted = 0;
                }
            }
            if (solver)
                statistics.cg_iterations += solver->iterations();
        }
        if (snapshots.empty())
            throw std::runtime_error("EROM extraction has no nonzero training input");
        return compress_snapshots(snapshots, options.tolerance);
    }
} // namespace mhs::app::erom
