#include "thermal_reconstruction.hpp"
#include "thermal_physics.hpp"
#include <algorithm>
#include <limits>
#include <optional>

namespace mhs::solver {
    namespace {
        mhs::model::Vec3 conductivity(const mhs::model::Model& model, std::span<const double> temperature,
            mhs::model::CellId cell, double time, const mhs::model::Vec3& p)
        {
            const mhs::expression::FieldContext ctx {p[0], p[1], p[2], temperature[cell], time};
            const auto& m = model.material_table[model.cells[cell].material];
            return {m.kx.eval(ctx), m.ky.eval(ctx), m.kz.eval(ctx)};
        }
        double boundary_trace(const mhs::model::Model& model, const mhs::model::Facet& facet, std::size_t id,
            std::span<const double> temperature, double time, double k)
        {
            const auto center = facet.center();
            const mhs::expression::FieldContext context {
                center[0], center[1], center[2], temperature[facet.owner()], time};
            const double d = cell_geometry(model.mesh, facet.owner()).size()[static_cast<std::size_t>(facet.axis)] / 2;
            return evaluate_boundary(model.boundary_conditions[model.boundary_thermal_condition[id]], context,
                facet.area(), facet.area() / side_thermal_resistance(d, k))
                .temperature;
        }
    }

    mhs::core::Vector<mhs::model::Vec3> reconstruct_gradients(
        const mhs::model::Model& model, std::span<const double> temperature, double time)
    {
        using namespace mhs::model;
        mhs::core::Vector<Vec3> gradients(model.cells.size()), k(model.cells.size());
        std::visit(
            [&](const auto& mesh) {
                for (std::size_t i = 0; i < k.size(); ++i) {
                    const auto cell = static_cast<CellId>(i);
                    k[i] = conductivity(model, temperature, cell, time, cell_geometry(mesh, cell).center());
                }
                for_each_connection(mesh, [&](Connection c) {
                    const auto a = cell_geometry(mesh, c.a), b = cell_geometry(mesh, c.b);
                    const auto g = connection_facet(c, a, b);
                    const auto axis = static_cast<std::size_t>(g.axis);
                    const auto [ra, rb] = connection_resistances(model, c, a, b, g.axis, k[c.a][axis], k[c.b][axis]);
                    const double trace = (rb * temperature[c.a] + ra * temperature[c.b]) / (ra + rb);
                    gradients[g.minus][axis] += g.area() * trace;
                    gradients[g.plus][axis] -= g.area() * trace;
                });
                for (std::size_t i = 0; i < model.boundary_facets.size(); ++i) {
                    const auto& f = model.boundary_facets[i];
                    const auto axis = static_cast<std::size_t>(f.axis);
                    const auto trace = boundary_trace(model, f, i, temperature, time, k[f.owner()][axis]);
                    gradients[f.owner()][axis] += f.outward_sign() * f.area() * trace;
                }
                for (std::size_t i = 0; i < gradients.size(); ++i) {
                    const auto volume = cell_geometry(mesh, static_cast<CellId>(i)).volume();
                    for (auto& value : gradients[i])
                        value /= volume;
                }
            },
            model.mesh);
        return gradients;
    }

    std::optional<double> boundary_value_at(const mhs::model::Model& model, std::span<const double> temperature,
        double time, mhs::model::CellId cell, const mhs::model::Vec3& p)
    {
        if (cell == mhs::core::invalid_id)
            return std::nullopt;
        const auto k = conductivity(model, temperature, cell, time, cell_geometry(model.mesh, cell).center());
        double sum = 0, dirichlet_sum = 0;
        std::size_t count = 0, dirichlet_count = 0;
        // The compiler emits exterior pieces in owner order, including after partitioning.
        auto it = std::lower_bound(model.boundary_facets.begin(), model.boundary_facets.end(), cell,
            [](const auto& f, mhs::model::CellId c) { return f.owner() < c; });
        for (; it != model.boundary_facets.end() && it->owner() == cell; ++it) {
            const auto& f = *it;
            if (!f.contains(p))
                continue;
            const auto id = static_cast<std::size_t>(it - model.boundary_facets.begin());
            const double trace = boundary_trace(model, f, id, temperature, time, k[static_cast<std::size_t>(f.axis)]);
            sum += trace;
            ++count;
            if (std::holds_alternative<mhs::model::Dirichlet>(
                    model.boundary_conditions[model.boundary_thermal_condition[id]])) {
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
        mhs::model::CellId cell, const mhs::model::Vec3& p, std::span<const mhs::model::Vec3> gradients)
    {
        if (cell == mhs::core::invalid_id)
            return std::numeric_limits<double>::quiet_NaN();
        if (const auto trace = boundary_value_at(model, temperature, time, cell, p))
            return *trace;
        const auto owned = gradients.empty() ? reconstruct_gradients(model, temperature, time)
                                             : mhs::core::Vector<mhs::model::Vec3> {};
        const auto& gradient = gradients.empty() ? owned[cell] : gradients[cell];
        const auto center = cell_geometry(model.mesh, cell).center();
        return temperature[cell] + gradient[0] * (p[0] - center[0]) + gradient[1] * (p[1] - center[1])
            + gradient[2] * (p[2] - center[2]);
    }
} // namespace mhs::solver
