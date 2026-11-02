#include "solver/postprocessor.hpp"
#include "thermal_reconstruction.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace mhs::post {

    namespace {

        // Inverse anisotropic thermal distance Δx²/kx + Δy²/ky + Δz²/kz.
        // The epsilon keeps a query point coincident with a cell center finite.
        double anisotropic_weight(
            const mhs::core::Vec3& center, const mhs::core::Vec3& point, const mhs::core::Vec3& conductivity)
        {
            const double dx = point[0] - center[0], dy = point[1] - center[1], dz = point[2] - center[2];
            return 1.0
                / (dx * dx / conductivity[0] + dy * dy / conductivity[1] + dz * dz / conductivity[2]
                    + std::numeric_limits<double>::epsilon());
        }

    } // namespace

    std::vector<double> interpolate_cell_to_node(
        const mhs::core::Model& model, std::span<const double> cell_temperature, double time)
    {
        if (cell_temperature.size() != model.discrete.cells.size())
            throw std::invalid_argument("node sampling state size mismatch");
        std::vector<double> numerator(model.output.points.size(), 0.0);
        std::vector<double> denominator(model.output.points.size(), 0.0);
        for (mhs::core::Index c = 0; c < model.output.cell_vertices.size(); ++c) {
            const auto center = model.discrete.cells[c].center();
            const auto temperature = cell_temperature[c];
            const mhs::core::FieldContext ctx {center[0], center[1], center[2], temperature, time};
            const auto& mat = model.material_table[model.cell_material[c]];
            const mhs::core::Vec3 conductivity {mat.kx.eval(ctx), mat.ky.eval(ctx), mat.kz.eval(ctx)};
            for (auto v : model.output.cell_vertices[c]) {
                const auto& p = model.output.points[v];
                const double sample
                    = mhs::sim::boundary_value_at(model, cell_temperature, time, {c}, p).value_or(temperature);
                const double w = anisotropic_weight(center, p, conductivity);
                numerator[v] += w * sample;
                denominator[v] += w;
            }
        }
        std::vector<double> node_T(model.output.points.size(), std::numeric_limits<double>::quiet_NaN());
        for (mhs::core::Index v = 0; v < node_T.size(); ++v)
            if (denominator[v] > 0.0)
                node_T[v] = numerator[v] / denominator[v];
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
