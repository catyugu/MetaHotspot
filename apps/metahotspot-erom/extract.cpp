#include "amg_solver.hpp"
#include "compiler/fom_compiler.hpp"
#include "solver/thermal_physics.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace mhs::app::erom {
    void Prepared::set_conductances(Eigen::Ref<Vector> values, const Vector& h) const
    {
        for (Eigen::Index id = 0; id < values.size(); ++id) {
            const auto& face = interfaces[id];
            values[id] = solver::convection_conductance(face.facet.area(), h[face.boundary_group], face.resistance);
        }
    }

    void Prepared::add_boundary(Eigen::Ref<Vector> cells, Eigen::Ref<const Vector> facets) const
    {
        for (std::size_t i = 0; i < boundary_cells.size(); ++i)
            cells[boundary_cells[i]] += facets[static_cast<Eigen::Index>(i)];
    }

    namespace {
        core::Vector<int> connected_regions(const Sparse& K)
        {
            core::Vector<int> region(static_cast<std::size_t>(K.rows()), -1);
            int next = 0;
            for (int cell = 0; cell < K.rows(); ++cell) {
                if (region[cell] >= 0)
                    continue;
                core::Vector<int> pending {cell};
                region[cell] = next;
                while (!pending.empty()) {
                    const int current = pending.back();
                    pending.pop_back();
                    for (Sparse::InnerIterator entry(K, current); entry; ++entry)
                        if (entry.value() != 0 && region[entry.col()] < 0) {
                            region[entry.col()] = next;
                            pending.push_back(static_cast<int>(entry.col()));
                        }
                }
                ++next;
            }
            return region;
        }

        bool covers_regions(const Prepared& project, const core::Vector<int>& regions, const Vector& h)
        {
            core::Vector<bool> anchored(static_cast<std::size_t>(*std::ranges::max_element(regions) + 1), false);
            for (Eigen::Index i = 0; i < h.size(); ++i)
                if (h[i] > 0)
                    anchored[regions[project.boundary_cells[i]]] = true;
            return std::ranges::find(anchored, false) == anchored.end();
        }

        double validate_outputs(const Prepared& project, const Matrix& boundary, const Matrix& stiffness,
            const Matrix& B, const Matrix& L, const Vector& lower, const Vector& upper, double tolerance,
            const core::Vector<int>& regions)
        {
            if (project.outputs.empty())
                return 0;
            const auto n = project.K.rows(), ng = lower.size();
            Vector closure(static_cast<Eigen::Index>(project.boundary_cells.size())), rhs(n);
            double worst = 0;
            auto check = [&](const Vector& h, bool defaults = false) {
                project.set_conductances(closure, h);
                if (defaults && !covers_regions(project, regions, closure))
                    return;
                Vector diagonal = Vector::Zero(n);
                project.add_boundary(diagonal, closure);
                const Sparse matrix = add_diagonal(project.K, diagonal);
                ReusableAmg solver(matrix, std::min(1e-8, tolerance * 1e-3));
                const Matrix reduced = stiffness + boundary.transpose() * (closure.asDiagonal() * boundary);
                Eigen::LLT<Matrix> factor(reduced);
                if (factor.info() != Eigen::Success)
                    throw std::runtime_error("reduced operator is not positive definite");

                auto compare = [&](const Vector& exact, const Vector& approximate) {
                    const Vector error = (exact - approximate).cwiseAbs().array() / exact.cwiseAbs().array().max(1.0);
                    if (!error.allFinite())
                        throw std::runtime_error("nonfinite validation temperature");
                    worst = std::max(worst, error.maxCoeff());
                };
                if (!defaults && L.rows() < B.cols() + ng) {
                    // The full and Galerkin operators are symmetric. One adjoint
                    // solve per output gives the same responses for every input.
                    for (Eigen::Index j = 0; j < L.rows(); ++j) {
                        rhs = Vector(project.L.row(j).transpose());
                        const Vector exact = solver.solve(matrix, rhs, Vector::Zero(n));
                        const Vector approximate = factor.solve(L.row(j).transpose());
                        if (B.cols())
                            compare(project.B.transpose() * exact, B.transpose() * approximate);
                        const Vector reduced_trace = boundary * approximate;
                        Vector ambient_exact(ng), ambient_approximate(ng);
                        for (Eigen::Index i = 0; i < ng; ++i) {
                            ambient_exact[i] = ambient_approximate[i] = 0;
                            for (auto id = project.training[i].begin; id < project.training[i].end; ++id) {
                                ambient_exact[i] += closure[id] * exact[project.boundary_cells[id]];
                                ambient_approximate[i] += closure[id] * reduced_trace[id];
                            }
                        }
                        compare(ambient_exact, ambient_approximate);
                    }
                    return;
                }
                auto compare_load = [&](const auto& load) {
                    compare(project.L * solver.solve(matrix, rhs, Vector::Zero(n)), L * factor.solve(load));
                };
                Vector load(B.rows());
                if (defaults) {
                    Vector powers(B.cols());
                    for (Eigen::Index i = 0; i < powers.size(); ++i)
                        powers[i] = project.sources[i].default_power;
                    rhs.noalias() = project.B * powers;
                    load.noalias() = B * powers;
                }
                else
                    for (Eigen::Index i = 0; i < B.cols(); ++i) {
                        rhs = project.B.col(i);
                        compare_load(B.col(i));
                    }
                for (Eigen::Index i = 0; i < ng; ++i) {
                    if (!defaults) {
                        rhs.setZero();
                        load.setZero();
                    }
                    const auto& group = project.training[i];
                    const double delta = defaults ? group.boundary_group.ambient - project.reference_temperature : 1;
                    for (auto id = group.begin; id < group.end; ++id) {
                        const double power = closure[id] * delta;
                        rhs[project.boundary_cells[id]] += power;
                        load.noalias() += power * boundary.row(id).transpose();
                    }
                    if (!defaults)
                        compare_load(load);
                }
                if (defaults)
                    compare_load(load);
            };
            check(lower);
            check(lower.cwiseSqrt().cwiseProduct(upper.cwiseSqrt()));
            check(upper);
            if (ng > 1)
                for (int alternate = 0; alternate < 2; ++alternate) {
                    Vector h(ng);
                    for (Eigen::Index i = 0; i < ng; ++i)
                        h[i] = (i + alternate) % 2 ? upper[i] : lower[i];
                    check(h);
                }
            Vector defaults(ng);
            for (Eigen::Index i = 0; i < ng; ++i)
                defaults[i] = project.training[i].boundary_group.coefficient;
            check(defaults, true);
            if (worst > tolerance) {
                std::ostringstream message;
                message << "output validation error " << worst << " exceeds tolerance " << tolerance;
                throw std::runtime_error(message.str());
            }
            return worst;
        }

        core::Matrix<double> native(const Matrix& matrix)
        {
            core::Matrix<double> result(
                static_cast<std::size_t>(matrix.rows()), static_cast<std::size_t>(matrix.cols()));
            using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            Eigen::Map<RowMajor>(result.data(), matrix.rows(), matrix.cols()) = matrix;
            return result;
        }
    } // namespace

    Extraction extract(
        const model::FomDefinition& definition, const model::MeshSpec& mesh, const HtcRanges& ranges, double tolerance)
    {
        if (!std::isfinite(tolerance) || tolerance <= 0 || tolerance >= 1)
            throw std::invalid_argument("EROM extraction tolerance must be in (0, 1)");
        if (ranges.empty())
            throw std::invalid_argument("EROM extraction requires HTC groups");
        for (const auto& [id, range] : ranges)
            if (!std::isfinite(range.minimum) || !std::isfinite(range.maximum) || range.minimum <= 0
                || range.maximum < range.minimum)
                throw std::invalid_argument("invalid HTC range for group " + std::to_string(id));
        auto project = prepare(compiler::compile_fom(definition, mesh), definition, ranges);
        if (!project.capacity.allFinite() || (project.capacity.array() <= 0).any())
            throw std::invalid_argument("EROM extraction requires finite positive cell capacities");
        const auto regions = connected_regions(project.K);
        if (!covers_regions(project, regions, Vector::Ones(static_cast<Eigen::Index>(project.boundary_cells.size()))))
            throw std::invalid_argument("each thermally disconnected FOM region needs an HTC training surface");

        Statistics statistics;
        const auto ng = static_cast<Eigen::Index>(project.training.size());
        Vector lower(ng), upper(ng);
        for (Eigen::Index i = 0; i < ng; ++i) {
            lower[i] = project.training[i].boundary_group.minimum;
            upper[i] = project.training[i].boundary_group.maximum;
        }
        const Matrix basis = build_basis(project, lower, upper, regions, tolerance, statistics);
        Matrix C = basis.transpose() * (project.capacity.asDiagonal() * basis);
        Matrix K = basis.transpose() * (project.K * basis);
        C = ((C + C.transpose()) * 0.5).eval();
        K = ((K + K.transpose()) * 0.5).eval();
        K.diagonal() = K.diagonal().cwiseMax(0.0);
        const Matrix B = basis.transpose() * project.B, L = project.L * basis;
        const Vector rhs = basis.transpose() * project.capacity;
        const Vector uniform = solve_positive(C, rhs);
        const Matrix boundary = basis(project.boundary_cells, Eigen::placeholders::all);
        statistics.output_error = validate_outputs(project, boundary, K, B, L, lower, upper, tolerance, regions);
        core::Vector<model::EromBoundaryGroup> surfaces;
        surfaces.reserve(project.training.size());
        for (auto& group : project.training)
            surfaces.push_back(std::move(group.boundary_group));
        model::Erom asset {native(K), native(C), native(B), native(L), native(boundary), std::move(project.interfaces),
            std::move(surfaces), std::move(project.outline), std::move(project.sources), std::move(project.outputs),
            project.reference_temperature, {uniform.begin(), uniform.end()}};
        model::validate_erom(asset);
        return {std::move(asset), statistics};
    }
} // namespace mhs::app::erom
