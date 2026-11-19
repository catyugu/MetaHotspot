#include "amg_solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace mhs::app::erom {
    namespace {
        Vector conductance(const SurfaceData& surface, double h, Eigen::Index n)
        {
            Vector diagonal = Vector::Zero(n);
            for (std::size_t i = 0; i < surface.cells.size(); ++i)
                diagonal[surface.cells[i]] += surface.facets[i].area() * h / (1 + surface.resistance[i] * h);
            return diagonal;
        }

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

        Matrix training_inputs(const Prepared& project, const core::Vector<int>& regions)
        {
            core::Vector<Vector> columns;
            core::Vector<bool> driven(static_cast<std::size_t>(*std::ranges::max_element(regions) + 1), false);
            for (Eigen::Index cell = 0; cell < project.B.rows(); ++cell)
                if (project.B.row(cell).squaredNorm() > 0)
                    driven[regions[static_cast<std::size_t>(cell)]] = true;
            for (const auto& surface : project.surfaces) {
                if (!surface.training)
                    continue;
                // Passive regions have no volumetric power port. Their physical
                // input is a uniform ambient-temperature change on an HTC surface.
                // Use its exact lower-HTC load, separately for each connected region.
                std::map<int, Vector> passive;
                const double h = surface.training->minimum;
                for (std::size_t i = 0; i < surface.cells.size(); ++i) {
                    const auto cell = surface.cells[i];
                    const int region = regions[cell];
                    if (driven[region])
                        continue;
                    auto [entry, inserted] = passive.try_emplace(region);
                    if (inserted)
                        entry->second = Vector::Zero(project.K.rows());
                    entry->second[cell] += surface.facets[i].area() * h / (1 + surface.resistance[i] * h);
                }
                for (auto& [region, g] : passive)
                    columns.push_back(g.normalized());
            }
            Matrix result(project.K.rows(), project.B.cols() + static_cast<Eigen::Index>(columns.size()));
            result.leftCols(project.B.cols()) = project.B;
            for (std::size_t i = 0; i < columns.size(); ++i)
                result.col(project.B.cols() + static_cast<Eigen::Index>(i)) = columns[i];
            return result;
        }

        double validate_outputs(const Prepared& project, const core::Vector<const SurfaceData*>& trained,
            const Matrix& basis, const Matrix& stiffness, const Matrix& B, const Matrix& L, const Options& options,
            const core::Vector<int>& regions)
        {
            if (project.outputs.empty())
                return 0;
            const auto ng = static_cast<Eigen::Index>(trained.size());
            core::Vector<Vector> cases(3, Vector(ng));
            for (Eigen::Index i = 0; i < ng; ++i) {
                const auto& h = *trained[static_cast<std::size_t>(i)]->training;
                cases[0][i] = h.minimum;
                cases[1][i] = std::sqrt(h.minimum) * std::sqrt(h.maximum);
                cases[2][i] = h.maximum;
            }
            if (ng > 1)
                for (int alternate = 0; alternate < 2; ++alternate) {
                    Vector h(ng);
                    for (Eigen::Index i = 0; i < ng; ++i)
                        h[i] = cases[static_cast<std::size_t>((i + alternate) % 2) * 2][i];
                    cases.push_back(std::move(h));
                }
            Vector defaults(ng);
            for (Eigen::Index i = 0; i < ng; ++i)
                defaults[i] = trained[static_cast<std::size_t>(i)]->training->coefficient;
            cases.push_back(std::move(defaults));
            double worst = 0;
            for (std::size_t index = 0; index < cases.size(); ++index) {
                const bool is_default = index + 1 == cases.size();
                const auto n = project.K.rows();
                Matrix ambient(n, ng);
                for (Eigen::Index i = 0; i < ng; ++i)
                    ambient.col(i) = conductance(*trained[static_cast<std::size_t>(i)], cases[index][i], n);
                const Vector diagonal = ambient.rowwise().sum();
                if (is_default) {
                    core::Vector<bool> anchored(
                        static_cast<std::size_t>(*std::ranges::max_element(regions) + 1), false);
                    for (Eigen::Index i = 0; i < n; ++i)
                        if (diagonal[i] > 0)
                            anchored[regions[static_cast<std::size_t>(i)]] = true;
                    if (std::ranges::find(anchored, false) != anchored.end())
                        continue;
                }
                const Sparse matrix = add_diagonal(project.K, diagonal);
                ReusableAmg solver(matrix, std::min(1e-8, options.output_error * 1e-3));
                Matrix loads(n, is_default ? 1 : project.B.cols() + ng);
                Matrix reduced_loads;
                if (is_default) {
                    Vector powers(static_cast<Eigen::Index>(project.sources.size()));
                    for (std::size_t i = 0; i < project.sources.size(); ++i)
                        powers[static_cast<Eigen::Index>(i)] = project.sources[i].default_power;
                    loads.col(0) = project.B * powers;
                    for (Eigen::Index i = 0; i < ng; ++i)
                        loads.col(0) += ambient.col(i)
                            * (trained[static_cast<std::size_t>(i)]->training->ambient - project.reference);
                    reduced_loads = basis.transpose() * loads;
                }
                else {
                    loads.leftCols(project.B.cols()) = project.B;
                    loads.rightCols(ng) = ambient;
                    reduced_loads.resize(basis.cols(), loads.cols());
                    reduced_loads.leftCols(B.cols()) = B;
                    reduced_loads.rightCols(ng) = basis.transpose() * ambient;
                }
                const Matrix reduced_matrix = stiffness + basis.transpose() * (diagonal.asDiagonal() * basis);
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
            if (worst > options.output_error)
                throw std::runtime_error("output validation error " + std::to_string(worst) + " exceeds limit "
                    + std::to_string(options.output_error));
            return worst;
        }

        core::Vector<model::Facet> outline(core::Vector<model::Facet> facets)
        {
            for (const int lo : {0, 2}) {
                const int other = 2 - lo;
                auto key = [&](const model::Facet& f) {
                    return std::tuple {
                        f.axis, f.normal, f.coordinate, f.bounds[other], f.bounds[other + 1], f.bounds[lo]};
                };
                std::ranges::stable_sort(facets, {}, key);
                core::Vector<model::Facet> merged;
                for (const auto& f : facets) {
                    if (!merged.empty()) {
                        auto& last = merged.back();
                        if (last.axis == f.axis && last.normal == f.normal && last.coordinate == f.coordinate
                            && last.bounds[other] == f.bounds[other] && last.bounds[other + 1] == f.bounds[other + 1]
                            && last.bounds[lo + 1] == f.bounds[lo]) {
                            last.bounds[lo + 1] = f.bounds[lo + 1];
                            continue;
                        }
                    }
                    merged.push_back(f);
                }
                facets = std::move(merged);
            }
            return facets;
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

    Extraction extract(const Prepared& project, const Options& options)
    {
        if (!std::isfinite(options.tolerance) || options.tolerance <= 0 || options.tolerance >= 1
            || options.max_order < 1 || !std::isfinite(options.output_error) || options.output_error <= 0
            || options.output_error >= 1)
            throw std::invalid_argument("invalid EROM extraction tolerance, order or output error");
        if (!project.capacity.allFinite() || (project.capacity.array() <= 0).any())
            throw std::invalid_argument("EROM extraction requires finite positive cell capacities");
        core::Vector<const SurfaceData*> trained;
        for (const auto& surface : project.surfaces)
            if (surface.training)
                trained.push_back(&surface);
        if (trained.empty())
            throw std::invalid_argument("EROM extraction requires HTC surfaces");
        const auto regions = connected_regions(project.K);
        core::Vector<bool> anchored(static_cast<std::size_t>(*std::ranges::max_element(regions) + 1), false);
        for (const auto* surface : trained)
            for (const auto cell : surface->cells)
                anchored[regions[cell]] = true;
        if (std::ranges::find(anchored, false) != anchored.end())
            throw std::invalid_argument("each thermally disconnected FOM region needs an HTC training surface");

        const auto n = project.K.rows();
        Matrix slopes(n, static_cast<Eigen::Index>(trained.size()));
        Vector offset = Vector::Zero(n);
        core::Vector<std::pair<double, double>> ranges;
        for (std::size_t i = 0; i < trained.size(); ++i) {
            const auto& surface = *trained[i];
            const auto& h = *surface.training;
            const Vector lower = conductance(surface, h.minimum, n);
            if (h.maximum == h.minimum)
                slopes.col(static_cast<Eigen::Index>(i)) = lower / h.minimum;
            else {
                slopes.col(static_cast<Eigen::Index>(i))
                    = (conductance(surface, h.maximum, n) - lower) / (h.maximum - h.minimum);
                offset += lower - h.minimum * slopes.col(static_cast<Eigen::Index>(i));
            }
            ranges.emplace_back(h.minimum, h.maximum);
        }
        Extraction result;
        auto& e = result.asset;
        const Matrix basis = build_basis(add_diagonal(project.K, offset), project.capacity, slopes, ranges,
            training_inputs(project, regions), options, result.statistics);
        Matrix C = basis.transpose() * (project.capacity.asDiagonal() * basis);
        Matrix K = basis.transpose() * (project.K * basis);
        C = ((C + C.transpose()) * 0.5).eval();
        K = ((K + K.transpose()) * 0.5).eval();
        K.diagonal() = K.diagonal().cwiseMax(0.0);
        const Matrix B = basis.transpose() * project.B, L = project.L * basis;
        const Vector rhs = basis.transpose() * project.capacity;
        const Vector uniform = solve_positive(C, rhs);
        result.statistics.output_error = validate_outputs(project, trained, basis, K, B, L, options, regions);
        e.K = native(K);
        e.C = native(C);
        e.B = native(B);
        e.L = native(L);
        e.sources = project.sources;
        e.outputs = project.outputs;
        e.reference_temperature = project.reference;
        e.initial_temperature = project.initial;
        e.training_tolerance = options.tolerance;
        for (Eigen::Index i = 0; i < uniform.size(); ++i) {
            e.uniform_temperature_mode.push_back(uniform[i]);
            e.initial_state.push_back(uniform[i] * (project.initial - project.reference));
        }
        core::Vector<const SurfaceData*> surfaces;
        for (const auto& surface : project.surfaces)
            surfaces.push_back(&surface);
        std::ranges::stable_sort(surfaces, {}, [](const auto* s) { return !s->training.has_value(); });
        std::size_t boundary_count = 0;
        for (const auto* surface : trained)
            boundary_count += surface->cells.size();
        e.Vb = core::Matrix<double>(boundary_count, static_cast<std::size_t>(basis.cols()));
        std::size_t row = 0;
        for (const auto* surface : surfaces) {
            model::Surface target {surface->id, surface->name, {}};
            e.surface_training.push_back(surface->training);
            const auto facets = surface->training ? surface->facets : outline(surface->facets);
            for (std::size_t i = 0; i < facets.size(); ++i) {
                target.facets.push_back(static_cast<model::FacetId>(e.facets.size()));
                e.facets.push_back(facets[i]);
                if (surface->training) {
                    for (Eigen::Index j = 0; j < basis.cols(); ++j)
                        e.Vb(row, static_cast<std::size_t>(j)) = basis(surface->cells[i], j);
                    e.R.push_back(surface->resistance[i]);
                    ++row;
                }
            }
            e.surfaces.push_back(std::move(target));
        }
        model::validate_erom(e);
        return result;
    }
} // namespace mhs::app::erom
