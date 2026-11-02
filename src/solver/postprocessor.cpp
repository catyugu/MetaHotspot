#include "solver/postprocessor.hpp"
#include "thermal_reconstruction.hpp"
#include <algorithm>
#include <array>
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
        std::vector<double> node_T(model.output.points.size(), 0.0);
        std::vector<double> weight(node_T.size(), 0.0);
        for (mhs::core::Index c = 0; c < model.output.cell_vertices.size(); ++c) {
            const auto center = model.discrete.cells[c].center();
            const auto temperature = cell_temperature[c];
            const mhs::core::FieldContext ctx {center[0], center[1], center[2], temperature, time};
            const auto& mat = model.material_table[model.cell_material[c]];
            const mhs::core::Vec3 conductivity {mat.kx.eval(ctx), mat.ky.eval(ctx), mat.kz.eval(ctx)};
            const auto& vertices = model.output.cell_vertices[c];
            std::array<double, 8> sum {}, dirichlet_sum {};
            std::array<unsigned, 8> count {}, dirichlet_count {};
            for (auto id : model.discrete.incident_faces({c})) {
                const auto& face = model.discrete.faces[id.value];
                if (face.neighbor.valid())
                    continue;
                for (const auto& term : mhs::sim::boundary_faces_on(model, id)) {
                    const auto patch = face.project(term.bounds);
                    const auto& condition = model.boundary_conditions[term.condition];
                    std::optional<double> trace;
                    for (std::size_t i = 0; i < vertices.size(); ++i) {
                        if (!patch.contains(model.output.points[vertices[i]]))
                            continue;
                        if (!trace)
                            trace = mhs::sim::evaluate_boundary(
                                condition, patch, temperature, conductivity[patch.axis], time)
                                        .temperature;
                        sum[i] += *trace;
                        ++count[i];
                        if (std::holds_alternative<mhs::core::Dirichlet>(condition)) {
                            dirichlet_sum[i] += *trace;
                            ++dirichlet_count[i];
                        }
                    }
                }
            }
            for (std::size_t i = 0; i < vertices.size(); ++i) {
                const auto v = vertices[i];
                const auto& p = model.output.points[v];
                const double sample = dirichlet_count[i] ? dirichlet_sum[i] / dirichlet_count[i]
                    : count[i]                           ? sum[i] / count[i]
                                                         : temperature;
                const double w = anisotropic_weight(center, p, conductivity);
                node_T[v] += w * sample;
                weight[v] += w;
            }
        }
        for (mhs::core::Index v = 0; v < node_T.size(); ++v)
            node_T[v] = weight[v] > 0.0 ? node_T[v] / weight[v] : std::numeric_limits<double>::quiet_NaN();
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
