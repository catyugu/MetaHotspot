#include "thermal_reconstruction.hpp"
#include "thermal_physics.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <span>

namespace mhs::solver {
    mhs::model::CellId locate_cell(const mhs::model::DiscreteMesh& mesh, const mhs::model::Vec3& p)
    {
        using namespace mhs::model;
        const auto& cells = mesh.cells;
        // Cache once per probe. Shared-face ties select the lexicographically
        // greatest lower corner, independent of cell storage/DOF ordering.
        CellId result = mhs::core::invalid_id;
        for (std::size_t i = 0; i < cells.size(); ++i) {
            const auto& cell = cells[i];
            bool contains = true;
            for (std::size_t axis = 0; axis < 3; ++axis)
                contains = contains && p[axis] >= cell.min[axis] && p[axis] <= cell.max[axis];
            if (contains && (result == mhs::core::invalid_id || cell.min > cells[result].min))
                result = static_cast<CellId>(i);
        }
        return result;
    }

    namespace {
        double conductivity(const mhs::model::Model& model, std::span<const double> temperature,
            mhs::model::CellId cell, double time, int axis)
        {
            const auto p = model.discrete.cells[cell].center();
            const mhs::expression::FieldContext ctx {p[0], p[1], p[2], temperature[cell], time};
            const auto& m = model.material_table[model.cell_material[cell]];
            return (axis == 0 ? m.kx : axis == 1 ? m.ky : m.kz).eval(ctx);
        }

        // Green-Gauss gradient on an orthogonal cell. Internal face temperatures
        // use the same two-point thermal-resistance law as the diffusion assembly.
        mhs::model::Vec3 reconstruct_cell_gradient(
            const mhs::model::Model& model, std::span<const double> temperature, double time, mhs::model::CellId cell)
        {
            using namespace mhs::core;
            using namespace mhs::model;
            const Vec3 k {conductivity(model, temperature, cell, time, 0),
                conductivity(model, temperature, cell, time, 1), conductivity(model, temperature, cell, time, 2)};
            Vec3 gradient {};
            const auto fluid_index = [&](CellId c) {
                return c < model.fluid.global_to_fluid.size() ? model.fluid.global_to_fluid[c] : mhs::core::invalid_id;
            };
            for (auto id : model.discrete.incident_facets(cell)) {
                const auto& f = model.discrete.facets[id];
                if (f.exterior())
                    continue;
                const auto axis = static_cast<std::size_t>(f.axis);
                const auto neighbor = f.minus == cell ? f.plus : f.minus;
                const double k_neighbor = conductivity(model, temperature, neighbor, time, static_cast<int>(f.axis));
                double a = side_thermal_resistance(model.discrete.cells[cell].size()[axis] / 2, k[axis]);
                double b = side_thermal_resistance(model.discrete.cells[neighbor].size()[axis] / 2, k_neighbor);
                const auto fi = fluid_index(cell), fj = fluid_index(neighbor);
                double r_film = 0.0;
                if (fi != mhs::core::invalid_id && fj == mhs::core::invalid_id) {
                    a = 0.0;
                    r_film = fluid_film_resistance(model.fluid.interface_heat_transfer_factor[fi], k[axis]);
                }
                else if (fj != mhs::core::invalid_id && fi == mhs::core::invalid_id) {
                    b = 0.0;
                    r_film = fluid_film_resistance(model.fluid.interface_heat_transfer_factor[fj], k_neighbor);
                }
                // The film spans the fluid bulk temperature and the shared wall temperature.
                const double r_cell_to_wall = a + (fi != mhs::core::invalid_id ? r_film : 0.0);
                const double r_neighbor_to_wall = b + (fj != mhs::core::invalid_id ? r_film : 0.0);
                const double weighted_temperature
                    = r_neighbor_to_wall * temperature[cell] + r_cell_to_wall * temperature[neighbor];
                const double trace = weighted_temperature / (a + b + r_film);
                gradient[axis] += (f.minus == cell ? 1 : -1) * f.area() * trace;
            }
            for (auto id : model.discrete.incident_facets(cell)) {
                const auto& facet = model.discrete.facets[id];
                if (!facet.exterior())
                    continue;
                const auto axis = static_cast<std::size_t>(facet.axis);
                const auto center = facet.center();
                const mhs::expression::FieldContext context {center[0], center[1], center[2], temperature[cell], time};
                const double d = model.discrete.cells[cell].size()[axis] / 2;
                const auto trace = evaluate_boundary(model.boundary_conditions[model.facet_thermal_condition[id]],
                    context, facet.area(), facet.area() / side_thermal_resistance(d, k[axis]));
                gradient[axis] += facet.outward_sign() * facet.area() * trace.temperature;
            }
            const auto volume = model.discrete.cells[cell].volume();
            return {gradient[0] / volume, gradient[1] / volume, gradient[2] / volume};
        }
    } // namespace

    std::optional<double> boundary_value_at(const mhs::model::Model& model, std::span<const double> temperature,
        double time, mhs::model::CellId cell, const mhs::model::Vec3& p)
    {
        if (cell == mhs::core::invalid_id)
            return std::nullopt;
        const double cell_temperature = temperature[cell];
        double sum = 0, dirichlet_sum = 0;
        size_t count = 0, dirichlet_count = 0;
        for (auto id : model.discrete.incident_facets(cell)) {
            const auto& facet = model.discrete.facets[id];
            if (!facet.exterior() || !facet.contains(p))
                continue;
            const auto axis = static_cast<std::size_t>(facet.axis);
            const auto& condition = model.boundary_conditions[model.facet_thermal_condition[id]];
            const auto center = facet.center();
            const mhs::expression::FieldContext context {center[0], center[1], center[2], cell_temperature, time};
            const double d = model.discrete.cells[cell].size()[axis] / 2;
            const double k = conductivity(model, temperature, cell, time, static_cast<int>(facet.axis));
            const double trace
                = evaluate_boundary(condition, context, facet.area(), facet.area() / side_thermal_resistance(d, k))
                      .temperature;
            sum += trace;
            ++count;
            if (std::holds_alternative<mhs::model::Dirichlet>(condition)) {
                dirichlet_sum += trace;
                ++dirichlet_count;
            }
        }
        if (dirichlet_count)
            return dirichlet_sum / static_cast<double>(dirichlet_count);
        if (count)
            return sum / static_cast<double>(count);
        return std::nullopt;
    }

    double sample_point(const mhs::model::Model& model, std::span<const double> temperature, double time,
        mhs::model::CellId cell, const mhs::model::Vec3& p)
    {
        if (cell == mhs::core::invalid_id)
            return std::numeric_limits<double>::quiet_NaN();
        if (const auto trace = boundary_value_at(model, temperature, time, cell, p))
            return *trace;
        const auto gradient = reconstruct_cell_gradient(model, temperature, time, cell);
        const auto center = model.discrete.cells[cell].center();
        return temperature[cell] + gradient[0] * (p[0] - center[0]) + gradient[1] * (p[1] - center[1])
            + gradient[2] * (p[2] - center[2]);
    }

} // namespace mhs::solver
