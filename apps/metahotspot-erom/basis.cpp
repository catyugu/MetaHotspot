#include "amg_solver.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <set>
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

    Vector solve_positive(const Matrix& matrix, const Vector& rhs)
    {
        Eigen::LLT<Matrix> solver(matrix);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("reduced operator is not positive definite");
        Vector result = solver.solve(rhs);
        if (!result.allFinite())
            throw std::runtime_error("nonfinite reduced response");
        return result;
    }

    namespace {
        struct ResponseSpace {
            Matrix basis, boundary, stiffness, mass;
            Vector source;

            ResponseSpace(Eigen::Index n, Eigen::Index nb) : basis(n, 0), boundary(nb, 0) { }

            Vector response(const Vector& closure, double shift, const Vector& ambient) const
            {
                const Matrix reduced
                    = stiffness + shift * mass + boundary.transpose() * (closure.asDiagonal() * boundary);
                const Vector rhs = ambient.size() ? Vector(boundary.transpose() * ambient) : source;
                return basis * solve_positive(reduced, rhs);
            }

            void append(Vector vector, const Prepared& project, const Vector* g)
            {
                const double original = vector.norm();
                for (int pass = 0; pass < 2; ++pass)
                    vector -= basis * (basis.transpose() * vector);
                const double norm = vector.norm();
                if (norm <= std::numeric_limits<double>::epsilon() * vector.size() * original)
                    throw std::runtime_error("response enrichment stagnated before the residual tolerance was met");
                vector /= norm;
                const auto r = basis.cols();
                auto append_projection = [&](Matrix& projected, const Vector& image) {
                    const Vector edge = basis.transpose() * image;
                    projected.conservativeResize(r + 1, r + 1);
                    projected.topRightCorner(r, 1) = edge;
                    projected.bottomLeftCorner(1, r) = edge.transpose();
                    projected(r, r) = vector.dot(image);
                };
                append_projection(stiffness, project.K * vector);
                append_projection(mass, project.capacity.cwiseProduct(vector));
                if (g) {
                    source.conservativeResize(r + 1);
                    source[r] = vector.dot(*g);
                }
                basis.conservativeResize(Eigen::NoChange, r + 1);
                basis.col(r) = vector;
                boundary.conservativeResize(Eigen::NoChange, r + 1);
                boundary.col(r) = vector(project.boundary_cells);
            }
        };

        // Regions without power inputs train against physical ambient inputs, one per HTC group and region.
        core::Vector<std::pair<Eigen::Index, int>> passive_inputs(
            const Prepared& project, const core::Vector<int>& regions)
        {
            core::Vector<bool> driven(static_cast<std::size_t>(*std::ranges::max_element(regions) + 1), false);
            for (Eigen::Index cell = 0; cell < project.B.rows(); ++cell)
                if (project.B.row(cell).squaredNorm() > 0)
                    driven[regions[cell]] = true;
            std::set<std::pair<Eigen::Index, int>> inputs;
            for (std::size_t id = 0; id < project.boundary_cells.size(); ++id)
                if (const auto region = regions[project.boundary_cells[id]]; !driven[region])
                    inputs.emplace(project.interfaces[id].boundary_group, region);
            return {inputs.begin(), inputs.end()};
        }
    } // namespace

    Matrix build_basis(const Prepared& project, const Vector& lower, const Vector& upper,
        const core::Vector<int>& regions, double tolerance, Statistics& statistics)
    {
        constexpr std::size_t max_responses = 512;
        const auto& K = project.K;
        const auto& capacity = project.capacity;
        const auto n = K.rows();
        const auto nb = static_cast<Eigen::Index>(project.boundary_cells.size());
        const Vector log_lower = lower.array().log(), log_upper = upper.array().log();
        Vector diagonal = Vector::Zero(n);
        Vector closure(nb);
        project.set_conductances(closure, lower);
        project.add_boundary(diagonal, closure);
        const Sparse minimum = add_diagonal(K, diagonal);
        diagonal.setZero();
        project.set_conductances(closure, upper);
        project.add_boundary(diagonal, closure);
        const Sparse maximum = add_diagonal(K, diagonal);
        const auto shifts = matching_shifts(minimum, maximum, capacity, tolerance, statistics);
        const auto passive = passive_inputs(project, regions);
        const auto powered = static_cast<std::size_t>(project.B.cols());
        core::Vector<ResponseSpace> spaces;
        for (std::size_t port = 0; port < powered + passive.size(); ++port)
            spaces.emplace_back(n, nb);
        core::Vector<Vector> snapshots;
        std::mt19937_64 random(20260805);
        std::uniform_real_distribution<double> uniform;
        for (const double shift : shifts) {
            std::unique_ptr<ReusableAmg> solver;
            const Vector shifted_capacity = shift * capacity;
            for (std::size_t port = 0; port < spaces.size(); ++port) {
                const bool is_passive = port >= powered;
                Vector g = is_passive ? Vector::Zero(n).eval() : Vector(project.B.col(static_cast<Eigen::Index>(port)));
                Vector ambient(is_passive ? nb : 0);
                double norm = is_passive ? 0 : g.norm();
                auto& space = spaces[port];
                int accepted = 0;
                while (accepted < 10) {
                    Vector h(lower.size());
                    for (Eigen::Index i = 0; i < h.size(); ++i)
                        h[i] = std::clamp(
                            std::exp(std::lerp(log_lower[i], log_upper[i], uniform(random))), lower[i], upper[i]);
                    project.set_conductances(closure, h);
                    diagonal = shifted_capacity;
                    project.add_boundary(diagonal, closure);
                    if (is_passive) {
                        const auto [group, region] = passive[port - powered];
                        ambient.setZero();
                        for (auto id = project.training[group].begin; id < project.training[group].end; ++id)
                            if (regions[project.boundary_cells[id]] == region)
                                ambient[id] = closure[id];
                        g.setZero();
                        project.add_boundary(g, ambient);
                        norm = g.norm();
                    }
                    Vector initial = Vector::Zero(n);
                    if (space.basis.cols()) {
                        initial = space.response(closure, shift, ambient);
                        const double residual = (K * initial + diagonal.cwiseProduct(initial) - g).norm() / norm;
                        if (std::isfinite(residual) && residual <= tolerance) {
                            ++accepted;
                            continue;
                        }
                    }
                    if (snapshots.size() >= max_responses)
                        throw std::runtime_error("FANTASTIC extraction reached the response solve limit");
                    if (!solver) {
                        solver = std::make_unique<ReusableAmg>(add_diagonal(minimum, shifted_capacity), tolerance / 10);
                        ++statistics.response_amg_setups;
                    }
                    Vector response = solver->solve(add_diagonal(K, diagonal), g, std::move(initial));
                    ++statistics.response_solves;
                    snapshots.push_back(response);
                    space.append(std::move(response), project, is_passive ? nullptr : &g);
                    accepted = 0;
                }
            }
            if (solver)
                statistics.cg_iterations += solver->iterations();
        }
        if (snapshots.empty())
            throw std::runtime_error("EROM extraction has no nonzero training input");
        return compress_snapshots(snapshots, tolerance);
    }
} // namespace mhs::app::erom
