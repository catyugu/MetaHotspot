#include "solver/postprocessor.hpp"
#include "core/vector.hpp"
#include "thermal_reconstruction.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>

namespace mhs::solver {

    namespace {

        // Inverse anisotropic thermal distance Δx²/kx + Δy²/ky + Δz²/kz.
        // The epsilon keeps a query point coincident with a cell center finite.
        double anisotropic_weight(
            const mhs::model::Vec3& center, const mhs::model::Vec3& point, const mhs::model::Vec3& conductivity)
        {
            const double dx = point[0] - center[0], dy = point[1] - center[1], dz = point[2] - center[2];
            return 1.0
                / (dx * dx / conductivity[0] + dy * dy / conductivity[1] + dz * dz / conductivity[2]
                    + std::numeric_limits<double>::epsilon());
        }

    } // namespace

    mhs::core::Vector<double> interpolate_cell_to_node(
        const mhs::model::Model& model, std::span<const double> cell_temperature, double time)
    {
        if (cell_temperature.size() != model.discrete.cells.size())
            throw std::invalid_argument("node sampling state size mismatch");
        mhs::core::Vector<double> node_T(model.output.points.size(), 0.0);
        mhs::core::Vector<double> weight(node_T.size(), 0.0);
        for (std::size_t c = 0; c < model.output.cell_vertices.size(); ++c) {
            const auto center = model.discrete.cells[c].center();
            const auto temperature = cell_temperature[c];
            const mhs::expression::FieldContext ctx {center[0], center[1], center[2], temperature, time};
            const auto& mat = model.material_table[model.cell_material[c]];
            const mhs::model::Vec3 conductivity {mat.kx.eval(ctx), mat.ky.eval(ctx), mat.kz.eval(ctx)};
            const auto& vertices = model.output.cell_vertices[c];
            std::array<double, 8> sum {}, dirichlet_sum {};
            std::array<unsigned, 8> count {}, dirichlet_count {};
            for (auto id : model.discrete.incident_faces(static_cast<mhs::model::CellId>(c))) {
                const auto& face = model.discrete.faces[id];
                if (!face.exterior())
                    continue;
                for (const auto& term : mhs::solver::boundary_faces_on(model, id)) {
                    const auto& condition = model.boundary_conditions[term.condition];
                    std::optional<double> trace;
                    for (std::size_t i = 0; i < vertices.size(); ++i) {
                        if (!face.contains(model.discrete.cells, model.output.points[vertices[i]], term.bounds))
                            continue;
                        if (!trace)
                            trace = mhs::solver::evaluate_boundary(condition,
                                face.center(model.discrete.cells, term.bounds), face.area(term.bounds),
                                face.distance(model.discrete.cells[face.cell()]), temperature,
                                conductivity[static_cast<std::size_t>(face.axis)], time)
                                        .temperature;
                        sum[i] += *trace;
                        ++count[i];
                        if (std::holds_alternative<mhs::model::Dirichlet>(condition)) {
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
        for (std::size_t v = 0; v < node_T.size(); ++v)
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

} // namespace mhs::solver
