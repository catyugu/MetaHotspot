#include "postprocess.hpp"
#include "core/vector.hpp"
#include "solver/solver.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
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

    mhs::core::Vector<double> interpolate_cell_to_node(const mhs::model::FomDomain& model,
        const mhs::model::CompiledLibrary& library, const mhs::model::MeshSpec& grid,
        std::span<const double> cell_temperature, double time)
    {
        if (cell_temperature.size() != model.discrete.cells.size())
            throw std::invalid_argument("node sampling state size mismatch");
        std::map<mhs::model::Vec3, std::size_t> nodes;
        for (double x : grid.x_vertices)
            for (double y : grid.y_vertices)
                for (double z : grid.z_vertices)
                    nodes.emplace(mhs::model::Vec3 {x, y, z}, nodes.size());
        mhs::core::Vector<double> node_T(nodes.size(), 0.0), weight(nodes.size(), 0.0);
        for (std::size_t c = 0; c < model.discrete.cells.size(); ++c) {
            const auto& cell = model.discrete.cells[c];
            const auto center = cell.center();
            const auto temperature = cell_temperature[c];
            const mhs::expression::FieldContext ctx {center[0], center[1], center[2], temperature, time};
            const auto& mat = library.material_table[model.cell_material[c]];
            const mhs::model::Vec3 conductivity {mat.kx.eval(ctx), mat.ky.eval(ctx), mat.kz.eval(ctx)};
            for (double x : {cell.min[0], cell.max[0]})
                for (double y : {cell.min[1], cell.max[1]})
                    for (double z : {cell.min[2], cell.max[2]}) {
                        const mhs::model::Vec3 point {x, y, z};
                        const auto it = nodes.find(point);
                        if (it == nodes.end())
                            continue;
                        const auto v = it->second;
                        const double sample = boundary_value_at(
                            model, library, cell_temperature, time, static_cast<mhs::model::CellId>(c), point)
                                                  .value_or(temperature);
                        const double w = anisotropic_weight(center, point, conductivity);
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

} // namespace mhs::app::postprocess
