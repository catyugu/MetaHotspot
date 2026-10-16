#include "solver/postprocessor.hpp"
#include "core/boundary_geometry.hpp"
#include "solver/interpolation.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <ranges>
#include <vector>

namespace mhs::post {

    std::vector<double> interpolate_cell_to_node(
        const mhs::core::Model& model, std::span<const double> cell_temperature, double time)
    {
        if (cell_temperature.size() != model.discrete.cells.size())
            throw std::invalid_argument("node sampling state size mismatch");
        std::vector<double> node_T(model.output.points.size(), std::numeric_limits<double>::quiet_NaN());
        for (mhs::core::Index v = 0; v < model.output.points.size(); ++v) {
            const auto p = model.output.points[v];
            double dirichlet_sum = 0;
            size_t count = 0;
            std::vector<mhs::utils::PointTemperatureSample> samples;
            for (auto cell : model.output.incident_cells(v)) {
                const auto row = cell.value;
                const auto T = cell_temperature[row];
                const auto center = model.discrete.cells[cell.value].center();
                const auto& mat = model.material_table[model.cell_material[cell.value]];
                const mhs::core::FieldContext ctx {center[0], center[1], center[2], T, time};
                for (auto id : model.discrete.incident_faces(cell)) {
                    const auto& f = model.discrete.faces[id.value];
                    if (f.neighbor.valid())
                        continue;
                    for (const auto& term : mhs::core::boundary_terms_on_face(model, id)) {
                        const auto& bc = model.boundary_conditions[term.condition];
                        if (const auto* dirichlet = std::get_if<mhs::core::Dirichlet>(&bc);
                            dirichlet && mhs::core::projected_face(model.discrete, term).contains(p)) {
                            dirichlet_sum += dirichlet->temperature.eval({p[0], p[1], p[2], T, time});
                            ++count;
                        }
                    }
                }
                samples.push_back(
                    {center[0], center[1], center[2], T, mat.kx.eval(ctx), mat.ky.eval(ctx), mat.kz.eval(ctx)});
            }
            if (count)
                node_T[v] = dirichlet_sum / static_cast<double>(count);
            else if (!samples.empty())
                node_T[v] = mhs::utils::recover_point_temperature(samples, p[0], p[1], p[2]);
        }

        return node_T;
    }

    double max_temperature(std::span<const double> T)
    {
        auto filtered = T | std::views::filter([](double v) { return !std::isnan(v); });
        if (filtered.empty())
            return std::numeric_limits<double>::quiet_NaN();
        return std::ranges::max(filtered);
    }

    double min_temperature(std::span<const double> T)
    {
        auto filtered = T | std::views::filter([](double v) { return !std::isnan(v); });
        if (filtered.empty())
            return std::numeric_limits<double>::quiet_NaN();
        return std::ranges::min(filtered);
    }

} // namespace mhs::post
