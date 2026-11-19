#include "amg_solver.hpp"
#include "compiler/fom_compiler.hpp"
#include "solver/thermal_physics.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace mhs::app::erom {
    void Prepared::set_conductances(Eigen::Ref<Vector> values, Eigen::Index group, double h) const
    {
        for (const auto id : asset.surfaces[group].facets)
            values[id] = solver::convection_conductance(asset.facets[id].area(), h, asset.R[id]);
    }

    Vector Prepared::conductances(const Vector& h) const
    {
        Vector result(static_cast<Eigen::Index>(boundary_cells.size()));
        for (Eigen::Index i = 0; i < h.size(); ++i)
            set_conductances(result, i, h[i]);
        return result;
    }

    Vector Prepared::parameters(double model::EromSurfaceTraining::* field) const
    {
        core::Vector<double> values;
        for (const auto& surface : asset.surfaces) {
            if (!surface.training)
                break; // prepare writes trained surfaces first
            values.push_back((*surface.training).*field);
        }
        return Eigen::Map<const Vector>(values.data(), static_cast<Eigen::Index>(values.size()));
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
            const Matrix& B, const Matrix& L, double tolerance, const core::Vector<int>& regions)
        {
            if (project.asset.outputs.empty())
                return 0;
            const Vector lower = project.parameters(&model::EromSurfaceTraining::minimum);
            const Vector upper = project.parameters(&model::EromSurfaceTraining::maximum);
            const auto n = project.K.rows(), ng = lower.size();
            double worst = 0;
            auto check = [&](const Vector& h, bool defaults = false) {
                const Vector closure = project.conductances(h);
                if (defaults && !covers_regions(project, regions, closure))
                    return;
                Vector diagonal = Vector::Zero(n);
                project.add_boundary(diagonal, closure);
                const Sparse matrix = add_diagonal(project.K, diagonal);
                ReusableAmg solver(matrix, std::min(1e-8, tolerance * 1e-3));

                Matrix ambient = Matrix::Zero(boundary.rows(), ng);
                for (Eigen::Index i = 0; i < ng; ++i)
                    for (const auto facet : project.asset.surfaces[i].facets)
                        ambient(facet, i) = closure[facet];
                const Matrix reduced_ambient = boundary.transpose() * ambient;
                Matrix loads(B.rows(), defaults ? 1 : B.cols() + ng);
                Vector rhs(n);
                if (defaults) {
                    Vector powers(B.cols());
                    for (Eigen::Index i = 0; i < powers.size(); ++i)
                        powers[i] = project.asset.sources[i].default_power;
                    const Vector delta = project.parameters(&model::EromSurfaceTraining::ambient).array()
                        - project.asset.reference_temperature;
                    rhs = project.B * powers;
                    project.add_boundary(rhs, ambient * delta);
                    loads.col(0) = B * powers + reduced_ambient * delta;
                }
                else {
                    loads.leftCols(B.cols()) = B;
                    loads.rightCols(ng) = reduced_ambient;
                }
                const Matrix reduced = stiffness + boundary.transpose() * (closure.asDiagonal() * boundary);
                const Matrix approximate = L * solve_positive(reduced, loads);
                for (Eigen::Index i = 0; i < loads.cols(); ++i) {
                    if (!defaults) {
                        if (i < B.cols())
                            rhs = project.B.col(i);
                        else {
                            rhs.setZero();
                            project.add_boundary(rhs, ambient.col(i - B.cols()));
                        }
                    }
                    const Vector exact = project.L * solver.solve(matrix, rhs, Vector::Zero(n));
                    const Vector error
                        = (exact - approximate.col(i)).cwiseAbs().array() / exact.cwiseAbs().array().max(1.0);
                    if (!error.allFinite())
                        throw std::runtime_error("nonfinite validation temperature");
                    worst = std::max(worst, error.maxCoeff());
                }
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
            check(project.parameters(&model::EromSurfaceTraining::coefficient), true);
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

        Extraction result;
        auto& e = project.asset;
        const Matrix basis = build_basis(project, regions, tolerance, result.statistics);
        Matrix C = basis.transpose() * (project.capacity.asDiagonal() * basis);
        Matrix K = basis.transpose() * (project.K * basis);
        C = ((C + C.transpose()) * 0.5).eval();
        K = ((K + K.transpose()) * 0.5).eval();
        K.diagonal() = K.diagonal().cwiseMax(0.0);
        const Matrix B = basis.transpose() * project.B, L = project.L * basis;
        const Vector rhs = basis.transpose() * project.capacity;
        const Vector uniform = solve_positive(C, rhs);
        const Matrix boundary = basis(project.boundary_cells, Eigen::placeholders::all);
        result.statistics.output_error = validate_outputs(project, boundary, K, B, L, tolerance, regions);
        e.K = native(K);
        e.C = native(C);
        e.B = native(B);
        e.L = native(L);
        e.uniform_temperature_mode.assign(uniform.begin(), uniform.end());
        e.Vb = native(boundary);
        model::validate_erom(e);
        result.asset = std::move(e);
        return result;
    }
} // namespace mhs::app::erom
