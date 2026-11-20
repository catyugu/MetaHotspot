#include "thermal_reconstruction.hpp"
#include <limits>
#include <optional>
#include <span>

namespace mhs::solver {
    namespace {
        BoundaryEvaluation exterior_sample(const model::Fom& model, double temperature, double time, model::FacetId id)
        {
            return evaluate_boundary(model.boundary_groups[model.facet_boundary_group[id]].condition,
                model.mesh.facets[id], temperature, time, boundary_resistance(model, id, temperature, time));
        }

        // Green-Gauss gradient on an orthogonal cell. Internal face temperatures
        // use the same two-point thermal-resistance law as the diffusion assembly.
        mhs::model::Vec3 reconstruct_cell_gradient(const mhs::model::Fom& model, std::span<const double> temperature,
            double time, mhs::model::CellId cell, const BoundarySampler& sample_boundary)
        {
            using namespace mhs::core;
            using namespace mhs::model;
            const Vec3 k {cell_conductivity(model, cell, 0, temperature[cell], time),
                cell_conductivity(model, cell, 1, temperature[cell], time),
                cell_conductivity(model, cell, 2, temperature[cell], time)};
            Vec3 gradient {};
            for (auto id : model.mesh.incident_facets(cell)) {
                const auto& f = model.mesh.facets[id];
                const auto& cells = model.mesh.cell_facets[id];
                if (cells.exterior())
                    continue;
                const auto axis = static_cast<std::size_t>(f.axis);
                const auto neighbor = cells.minus == cell ? cells.plus : cells.minus;
                const double k_neighbor = cell_conductivity(model, neighbor, axis, temperature[neighbor], time);
                const auto [r_minus, r_plus] = cells.minus == cell
                    ? interior_facet_resistances(model, id, k[axis], k_neighbor)
                    : interior_facet_resistances(model, id, k_neighbor, k[axis]);
                const double r_cell = cells.minus == cell ? r_minus : r_plus;
                const double r_neighbor = cells.minus == cell ? r_plus : r_minus;
                const double trace
                    = (r_neighbor * temperature[cell] + r_cell * temperature[neighbor]) / (r_minus + r_plus);
                gradient[axis] += (cells.minus == cell ? 1 : -1) * f.area() * trace;
            }
            for (auto id : model.mesh.incident_facets(cell)) {
                const auto& facet = model.mesh.facets[id];
                if (!model.mesh.cell_facets[id].exterior())
                    continue;
                const auto axis = static_cast<std::size_t>(facet.axis);
                const auto trace = sample_boundary ? sample_boundary(id, nullptr)
                                                   : exterior_sample(model, temperature[cell], time, id);
                gradient[axis] += facet.normal * facet.area() * trace.temperature;
            }
            const auto volume = model.mesh.cell(cell).volume();
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
        for (auto id : model.mesh.incident_facets(cell)) {
            const auto& facet = model.mesh.facets[id];
            if (!model.mesh.cell_facets[id].exterior() || !facet.contains(p))
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
        mhs::model::CellId cell, const mhs::model::Vec3& p, const BoundarySampler& sample_boundary)
    {
        if (cell == mhs::core::invalid_id)
            return std::numeric_limits<double>::quiet_NaN();
        if (const auto trace = boundary_value(model, temperature, time, cell, p, sample_boundary))
            return *trace;
        const auto gradient = reconstruct_cell_gradient(model, temperature, time, cell, sample_boundary);
        const auto center = model.mesh.cell(cell).center();
        return temperature[cell] + gradient[0] * (p[0] - center[0]) + gradient[1] * (p[1] - center[1])
            + gradient[2] * (p[2] - center[2]);
    }

} // namespace mhs::solver
