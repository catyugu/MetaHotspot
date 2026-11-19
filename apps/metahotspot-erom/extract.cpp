#include "amg_solver.hpp"
#include "compiler/fom_compiler.hpp"
#include "solver/thermal_physics.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace mhs::app::erom {
    HtcOperators::HtcOperators(const Prepared& project, std::size_t group_count)
        : asset(project.asset)
        , group_count(group_count)
        , cells(project.boundary_cells.begin(), project.boundary_cells.end())
    {
        std::ranges::sort(cells);
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
        facet_rows.reserve(project.boundary_cells.size());
        for (const auto cell : project.boundary_cells)
            facet_rows.push_back(std::ranges::lower_bound(cells, cell) - cells.begin());
    }

    void HtcOperators::add_conductance(Eigen::Ref<Vector> diagonal, std::size_t group, double h) const
    {
        if (h == 0)
            return;
        for (const auto id : asset.surfaces[group].facets)
            diagonal[facet_rows[id]] += solver::convection_conductance(asset.facets[id].area(), h, asset.R[id]);
    }

    Vector HtcOperators::diagonal(const Vector& h) const
    {
        Vector result = Vector::Zero(static_cast<Eigen::Index>(cells.size()));
        for (std::size_t i = 0; i < group_count; ++i)
            add_conductance(result, i, h[static_cast<Eigen::Index>(i)]);
        return result;
    }

    Vector HtcOperators::parameters(double model::EromSurfaceTraining::* field) const
    {
        Vector result(static_cast<Eigen::Index>(group_count));
        for (std::size_t i = 0; i < group_count; ++i)
            result[static_cast<Eigen::Index>(i)] = (*asset.surface_training[i]).*field;
        return result;
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

        double validate_outputs(const Prepared& project, const HtcOperators& htc, const Matrix& basis,
            const Matrix& stiffness, const Matrix& B, const Matrix& L, double tolerance,
            const core::Vector<int>& regions)
        {
            if (project.asset.outputs.empty())
                return 0;
            const Matrix boundary = basis(htc.cells, Eigen::placeholders::all);
            const auto ng = static_cast<Eigen::Index>(htc.group_count);
            const Vector lower = htc.parameters(&model::EromSurfaceTraining::minimum);
            const Vector upper = htc.parameters(&model::EromSurfaceTraining::maximum);
            core::Vector<Vector> cases {lower, lower.cwiseSqrt().cwiseProduct(upper.cwiseSqrt()), upper};
            if (ng > 1)
                for (int alternate = 0; alternate < 2; ++alternate) {
                    Vector h(ng);
                    for (Eigen::Index i = 0; i < ng; ++i)
                        h[i] = cases[static_cast<std::size_t>((i + alternate) % 2) * 2][i];
                    cases.push_back(std::move(h));
                }
            cases.push_back(htc.parameters(&model::EromSurfaceTraining::coefficient));
            double worst = 0;
            for (std::size_t index = 0; index < cases.size(); ++index) {
                const bool is_default = index + 1 == cases.size();
                const auto n = project.K.rows();
                Matrix ambient = Matrix::Zero(boundary.rows(), ng);
                for (Eigen::Index i = 0; i < ng; ++i)
                    htc.add_conductance(ambient.col(i), static_cast<std::size_t>(i), cases[index][i]);
                const Vector closure = ambient.rowwise().sum();
                if (is_default) {
                    core::Vector<bool> anchored(
                        static_cast<std::size_t>(*std::ranges::max_element(regions) + 1), false);
                    for (std::size_t i = 0; i < htc.cells.size(); ++i)
                        if (closure[static_cast<Eigen::Index>(i)] > 0)
                            anchored[regions[htc.cells[i]]] = true;
                    if (std::ranges::find(anchored, false) != anchored.end())
                        continue;
                }
                Vector diagonal = Vector::Zero(n);
                diagonal(htc.cells) = closure;
                const Sparse matrix = add_diagonal(project.K, diagonal);
                ReusableAmg solver(matrix, std::min(1e-8, tolerance * 1e-3));
                Matrix loads = Matrix::Zero(n, is_default ? 1 : project.B.cols() + ng);
                Matrix reduced_loads;
                if (is_default) {
                    Vector powers(static_cast<Eigen::Index>(project.asset.sources.size()));
                    for (std::size_t i = 0; i < project.asset.sources.size(); ++i)
                        powers[static_cast<Eigen::Index>(i)] = project.asset.sources[i].default_power;
                    loads.col(0) = project.B * powers;
                    const Vector delta = htc.parameters(&model::EromSurfaceTraining::ambient).array()
                        - project.asset.reference_temperature;
                    loads.col(0)(htc.cells) += ambient * delta;
                    reduced_loads = basis.transpose() * loads;
                }
                else {
                    loads.leftCols(project.B.cols()) = project.B;
                    loads.rightCols(ng)(htc.cells, Eigen::placeholders::all) = ambient;
                    reduced_loads.resize(basis.cols(), loads.cols());
                    reduced_loads.leftCols(B.cols()) = B;
                    reduced_loads.rightCols(ng) = boundary.transpose() * ambient;
                }
                const Matrix reduced_matrix = stiffness + boundary.transpose() * (closure.asDiagonal() * boundary);
                const Matrix approximate = L * solve_positive(reduced_matrix, reduced_loads);
                for (Eigen::Index i = 0; i < loads.cols(); ++i) {
                    const Vector full = solver.solve(matrix, Vector(loads.col(i)), Vector::Zero(n));
                    const Vector exact = project.L * full;
                    const Vector error
                        = (exact - approximate.col(i)).cwiseAbs().array() / exact.cwiseAbs().array().max(1.0);
                    if (!error.allFinite())
                        throw std::runtime_error("nonfinite validation temperature");
                    worst = std::max(worst, error.maxCoeff());
                }
            }
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
        const HtcOperators htc(project, ranges.size());
        const auto regions = connected_regions(project.K);
        core::Vector<bool> anchored(static_cast<std::size_t>(*std::ranges::max_element(regions) + 1), false);
        for (const auto cell : htc.cells)
            anchored[regions[cell]] = true;
        if (std::ranges::find(anchored, false) != anchored.end())
            throw std::invalid_argument("each thermally disconnected FOM region needs an HTC training surface");

        Extraction result;
        auto& e = project.asset;
        const Matrix basis = build_basis(project, htc, regions, tolerance, result.statistics);
        Matrix C = basis.transpose() * (project.capacity.asDiagonal() * basis);
        Matrix K = basis.transpose() * (project.K * basis);
        C = ((C + C.transpose()) * 0.5).eval();
        K = ((K + K.transpose()) * 0.5).eval();
        K.diagonal() = K.diagonal().cwiseMax(0.0);
        const Matrix B = basis.transpose() * project.B, L = project.L * basis;
        const Vector rhs = basis.transpose() * project.capacity;
        const Vector uniform = solve_positive(C, rhs);
        result.statistics.output_error = validate_outputs(project, htc, basis, K, B, L, tolerance, regions);
        e.K = native(K);
        e.C = native(C);
        e.B = native(B);
        e.L = native(L);
        e.uniform_temperature_mode.assign(uniform.begin(), uniform.end());
        e.Vb = core::Matrix<double>(project.boundary_cells.size(), static_cast<std::size_t>(basis.cols()));
        for (std::size_t i = 0; i < project.boundary_cells.size(); ++i)
            for (Eigen::Index j = 0; j < basis.cols(); ++j)
                e.Vb(i, static_cast<std::size_t>(j)) = basis(project.boundary_cells[i], j);
        model::validate_erom(e);
        result.asset = std::move(e);
        return result;
    }
} // namespace mhs::app::erom
