#include "thermal_reconstruction.hpp"
#include <limits>
#include <optional>
#include <span>

namespace mhs::solver {
    namespace {
        BoundaryEvaluation exterior_sample(const model::Fom& model, double temperature, double time, model::FacetId id)
        {
            const auto& facet = model.mesh.exterior_facets[id];
            return evaluate_boundary(model.boundary_groups[facet.boundary_group].condition, facet.geometry, temperature,
                time, boundary_resistance(model, id, temperature, time));
        }

        // Green-Gauss gradient on an orthogonal cell. Internal face temperatures
        // use the same two-point thermal-resistance law as the diffusion assembly.
        mhs::model::Vec3 reconstruct_cell_gradient(const mhs::model::Fom& model, std::span<const double> temperature,
            double time, mhs::model::CellId cell, const BoundarySampler& sample_boundary,
            std::optional<double> conductivity_temperature)
        {
            using namespace mhs::core;
            using namespace mhs::model;
            const Vec3 k {cell_conductivity(model, cell, 0, conductivity_temperature.value_or(temperature[cell]), time),
                cell_conductivity(model, cell, 1, conductivity_temperature.value_or(temperature[cell]), time),
                cell_conductivity(model, cell, 2, conductivity_temperature.value_or(temperature[cell]), time)};
            const auto geometry = model.mesh.cell(cell);
            Vec3 gradient {};
            for (std::size_t axis = 0; axis < 3; ++axis) {
                const auto area = geometry.face_area(axis);
                for (int sign : {-1, 1}) {
                    const auto neighbor = model.mesh.neighbor(cell, axis, sign);
                    if (neighbor == invalid_id)
                        continue;
                    const double k_neighbor = cell_conductivity(model, neighbor, axis,
                        conductivity_temperature.value_or(temperature[neighbor]), time);
                    const auto [r_cell, r_neighbor]
                        = interior_resistances(model, cell, neighbor, axis, k[axis], k_neighbor);
                    const double trace
                        = (r_neighbor * temperature[cell] + r_cell * temperature[neighbor]) / (r_cell + r_neighbor);
                    gradient[axis] += sign * area * trace;
                }
            }
            for (auto id : model.mesh.boundary_ids(cell)) {
                const auto& facet = model.mesh.exterior_facets[id].geometry;
                const auto axis = static_cast<std::size_t>(facet.axis);
                const auto trace = sample_boundary ? sample_boundary(id, nullptr)
                                                   : exterior_sample(model, temperature[cell], time, id);
                gradient[axis] += facet.normal * facet.area() * trace.temperature;
            }
            const auto volume = geometry.volume();
            return {gradient[0] / volume, gradient[1] / volume, gradient[2] / volume};
        }
    } // namespace

    static std::optional<double> boundary_value(const mhs::model::Fom& model, std::span<const double> temperature,
        double time, mhs::model::CellId cell, const mhs::model::Vec3& p, const BoundarySampler& sample_boundary)
    {
        if (cell == mhs::core::invalid_id)
            return std::nullopt;
        double sum = 0, dirichlet_sum = 0;
        size_t count = 0, dirichlet_count = 0;
        for (auto id : model.mesh.boundary_ids(cell)) {
            const auto& facet = model.mesh.exterior_facets[id].geometry;
            if (!facet.contains(p))
                continue;
            const auto trace
                = sample_boundary ? sample_boundary(id, &p) : exterior_sample(model, temperature[cell], time, id);
            sum += trace.temperature;
            ++count;
            if (trace.dirichlet) {
                dirichlet_sum += trace.temperature;
                ++dirichlet_count;
            }
        }
        if (dirichlet_count)
            return dirichlet_sum / static_cast<double>(dirichlet_count);
        if (count)
            return sum / static_cast<double>(count);
        return std::nullopt;
    }

    std::optional<double> boundary_value_at(const model::Fom& model, std::span<const double> temperature, double time,
        model::CellId cell, const model::Vec3& point)
    {
        return boundary_value(model, temperature, time, cell, point, {});
    }

    double sample_point(const mhs::model::Fom& model, std::span<const double> temperature, double time,
        mhs::model::CellId cell, const mhs::model::Vec3& p, const BoundarySampler& sample_boundary,
        std::optional<double> conductivity_temperature)
    {
        if (cell == mhs::core::invalid_id)
            return std::numeric_limits<double>::quiet_NaN();
        if (const auto trace = boundary_value(model, temperature, time, cell, p, sample_boundary))
            return *trace;
        const auto gradient = reconstruct_cell_gradient(model, temperature, time, cell, sample_boundary, conductivity_temperature);
        const auto center = model.mesh.cell(cell).center();
        return temperature[cell] + gradient[0] * (p[0] - center[0]) + gradient[1] * (p[1] - center[1])
            + gradient[2] * (p[2] - center[2]);
    }

} // namespace mhs::solver
