#include "thermal_reconstruction.hpp"
#include "thermal_physics.hpp"
#include <limits>
#include <optional>
#include <span>

namespace mhs::solver {
    namespace {
        double conductivity(const mhs::model::Model& model, std::span<const double> temperature,
            mhs::model::CellId cell, double time, int axis)
        {
            const auto p = model.discrete.cells[cell].center();
            const mhs::expression::FieldContext ctx {p[0], p[1], p[2], temperature[cell], time};
            const auto& m = model.material_table[model.cell_material[cell]];
            return (axis == 0 ? m.kx : axis == 1 ? m.ky : m.kz).eval(ctx);
        }

        BoundarySample exterior_sample(const model::Model& model, std::span<const double> temperature, double time,
            model::CellId cell, model::FacetId id)
        {
            const auto& facet = model.discrete.facets[id];
            const auto axis = static_cast<std::size_t>(facet.axis);
            const auto& condition = model.boundary_conditions[model.facet_thermal_condition[id]];
            const auto center = facet.center();
            const expression::FieldContext context {center[0], center[1], center[2], temperature[cell], time};
            const double d = model.discrete.cells[cell].size()[axis] / 2;
            const double k = conductivity(model, temperature, cell, time, static_cast<int>(facet.axis));
            const auto trace
                = evaluate_boundary(condition, context, facet.area(), facet.area() / side_thermal_resistance(d, k));
            return {trace.temperature, std::holds_alternative<model::Dirichlet>(condition)};
        }

        // Green-Gauss gradient on an orthogonal cell. Internal face temperatures
        // use the same two-point thermal-resistance law as the diffusion assembly.
        mhs::model::Vec3 reconstruct_cell_gradient(const mhs::model::Model& model, std::span<const double> temperature,
            double time, mhs::model::CellId cell, const BoundarySampler& sample_boundary)
        {
            using namespace mhs::core;
            using namespace mhs::model;
            const Vec3 k {conductivity(model, temperature, cell, time, 0),
                conductivity(model, temperature, cell, time, 1), conductivity(model, temperature, cell, time, 2)};
            Vec3 gradient {};
            for (auto id : model.discrete.incident_facets(cell)) {
                const auto& f = model.discrete.facets[id];
                const auto& cells = model.discrete.cell_facets[id];
                if (cells.exterior())
                    continue;
                const auto axis = static_cast<std::size_t>(f.axis);
                const auto neighbor = cells.minus == cell ? cells.plus : cells.minus;
                const double k_neighbor = conductivity(model, temperature, neighbor, time, static_cast<int>(f.axis));
                const auto [r_minus, r_plus] = cells.minus == cell
                    ? interior_facet_resistances(model, id, k[axis], k_neighbor)
                    : interior_facet_resistances(model, id, k_neighbor, k[axis]);
                const double r_cell = cells.minus == cell ? r_minus : r_plus;
                const double r_neighbor = cells.minus == cell ? r_plus : r_minus;
                const double trace
                    = (r_neighbor * temperature[cell] + r_cell * temperature[neighbor]) / (r_minus + r_plus);
                gradient[axis] += (cells.minus == cell ? 1 : -1) * f.area() * trace;
            }
            for (auto id : model.discrete.incident_facets(cell)) {
                const auto& facet = model.discrete.facets[id];
                if (!model.discrete.cell_facets[id].exterior())
                    continue;
                const auto axis = static_cast<std::size_t>(facet.axis);
                const auto trace = sample_boundary ? sample_boundary(id, nullptr)
                                                   : exterior_sample(model, temperature, time, cell, id);
                gradient[axis] += facet.normal * facet.area() * trace.temperature;
            }
            const auto volume = model.discrete.cells[cell].volume();
            return {gradient[0] / volume, gradient[1] / volume, gradient[2] / volume};
        }
    } // namespace

    static std::optional<double> boundary_value(const mhs::model::Model& model, std::span<const double> temperature,
        double time, mhs::model::CellId cell, const mhs::model::Vec3& p, const BoundarySampler& sample_boundary)
    {
        if (cell == mhs::core::invalid_id)
            return std::nullopt;
        double sum = 0, dirichlet_sum = 0;
        size_t count = 0, dirichlet_count = 0;
        for (auto id : model.discrete.incident_facets(cell)) {
            const auto& facet = model.discrete.facets[id];
            if (!model.discrete.cell_facets[id].exterior() || !facet.contains(p))
                continue;
            const auto trace
                = sample_boundary ? sample_boundary(id, &p) : exterior_sample(model, temperature, time, cell, id);
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

    std::optional<double> boundary_value_at(const model::Model& model, std::span<const double> temperature, double time,
        model::CellId cell, const model::Vec3& point)
    {
        return boundary_value(model, temperature, time, cell, point, {});
    }

    double sample_point(const mhs::model::Model& model, std::span<const double> temperature, double time,
        mhs::model::CellId cell, const mhs::model::Vec3& p, const BoundarySampler& sample_boundary)
    {
        if (cell == mhs::core::invalid_id)
            return std::numeric_limits<double>::quiet_NaN();
        if (const auto trace = boundary_value(model, temperature, time, cell, p, sample_boundary))
            return *trace;
        const auto gradient = reconstruct_cell_gradient(model, temperature, time, cell, sample_boundary);
        const auto center = model.discrete.cells[cell].center();
        return temperature[cell] + gradient[0] * (p[0] - center[0]) + gradient[1] * (p[1] - center[1])
            + gradient[2] * (p[2] - center[2]);
    }

} // namespace mhs::solver
