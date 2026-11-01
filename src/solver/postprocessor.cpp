#include "solver/postprocessor.hpp"
#include "core/boundary_geometry.hpp"
#include "solver/interpolation.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
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
            std::vector<mhs::utils::PointTemperatureSample> samples;
            for (auto cell : model.output.incident_cells(v)) {
                const auto row = cell.value;
                const auto T = cell_temperature[row];
                const auto center = model.discrete.cells[cell.value].center();
                const auto& mat = model.material_table[model.cell_material[cell.value]];
                const mhs::core::FieldContext ctx {center[0], center[1], center[2], T, time};
                bool boundary_point = false;
                for (auto id : model.discrete.incident_faces(cell)) {
                    const auto& f = model.discrete.faces[id.value];
                    if (f.neighbor.valid())
                        continue;
                    for (const auto& boundary : mhs::core::boundary_faces_on(model, id)) {
                        if (mhs::core::projected_face(model.discrete, boundary).contains(p))
                            boundary_point = true;
                    }
                }
                const double sample
                    = boundary_point ? mhs::utils::sample_point(model, cell_temperature, time, cell, p) : T;
                samples.push_back(
                    {center[0], center[1], center[2], sample, mat.kx.eval(ctx), mat.ky.eval(ctx), mat.kz.eval(ctx)});
            }
            if (!samples.empty())
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
