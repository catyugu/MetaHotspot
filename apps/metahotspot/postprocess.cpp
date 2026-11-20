#include "postprocess.hpp"
#include "core/vector.hpp"
#include "solver/solver.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>

namespace mhs::app::postprocess {

    using namespace mhs::solver;
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
        const mhs::model::Fom& model, std::span<const double> cell_temperature, double time)
    {
        const auto& mesh = model.mesh;
        if (cell_temperature.size() != mesh.cell_count())
            throw std::invalid_argument("node sampling state size mismatch");
        const auto& vertices = mesh.vertices;
        const auto node_count = vertices[0].size() * vertices[1].size() * vertices[2].size();
        mhs::core::Vector<double> node_T(node_count, 0.0), weight(node_count, 0.0);
        for (mhs::model::CellId c = 0; c < mesh.cell_count(); ++c) {
            const auto [i, j, k] = mesh.cell_coords(c);
            const auto cell = mesh.cell(c);
            const auto center = cell.center();
            const auto temperature = cell_temperature[c];
            const mhs::expression::FieldContext ctx {center[0], center[1], center[2], temperature, time};
            const auto& mat = model.material_table[model.cell_material[c]];
            const mhs::model::Vec3 conductivity {mat.kx.eval(ctx), mat.ky.eval(ctx), mat.kz.eval(ctx)};
            for (auto x : {i, i + 1})
                for (auto y : {j, j + 1})
                    for (auto z : {k, k + 1}) {
                        const auto point = mesh.vertex(x, y, z);
                        const auto index = mesh.vertex_index(x, y, z);
                        const double sample
                            = boundary_value_at(model, cell_temperature, time, c, point).value_or(temperature);
                        const double w = anisotropic_weight(center, point, conductivity);
                        node_T[index] += w * sample;
                        weight[index] += w;
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

} // namespace mhs::app::postprocess
