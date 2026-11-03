// Rectilinear finite-volume model of the stack in stack.hpp.
//
// Cells are cell-centred boxes. A cell exists where its centre lies inside its
// layer's footprint, so the layers with different outlines (100 mm PCB, 60 mm
// substrate, 45 x 35 mm chips) share one graded in-plane axis. Faces are the
// area-weighted harmonic mean of the two half-cell resistances, with the
// through-plane and in-plane conductivities of Table I kept separate.
#pragma once

#include "stack.hpp"

#include <Eigen/Sparse>

#include <cmath>
#include <vector>

namespace mhs::hdim {

    inline constexpr double kGrowth = 1.3;

    /// In-plane axis: uniform at `h` over the chip outline, then growing outward
    /// by `kGrowth` until the widest footprint is reached.
    inline std::vector<double> graded_axis(double half_chip, double half_outer, double h)
    {
        int inner = static_cast<int>(std::lround(2.0 * half_chip / h));
        if (inner % 2 != 0)
            ++inner;
        inner = std::max(2, inner);
        const double step = 2.0 * half_chip / inner;

        std::vector<double> right {0.0}; // cell edges > 0, ascending
        for (int i = 1; i <= inner / 2; ++i)
            right.push_back(i * step);
        double edge = right.back();
        double size = step;
        while (edge < half_outer - 1e-12) {
            size *= kGrowth;
            edge = std::min(edge + size, half_outer);
            right.push_back(edge);
        }

        std::vector<double> axis;
        for (auto it = right.rbegin(); it != right.rend(); ++it)
            if (*it > 1e-15)
                axis.push_back(-*it);
        axis.insert(axis.end(), right.begin(), right.end());
        return axis;
    }

    struct Mesh {
        std::vector<double> x, y; // cell edges
        struct LayerCells {
            int nz = 0;
            double dz = 0.0;
            std::vector<int> id; // (k * ny + j) * nx + i -> global id, -1 outside
        };
        std::vector<LayerCells> layers;
        std::vector<int> cell_layer;
        std::vector<double> volume;
        int nx = 0, ny = 0, n = 0;

        /// Global cell id, or -1 when the index falls outside the layer.
        int id(int layer, int i, int j, int k) const
        {
            const LayerCells& cells = layers[static_cast<std::size_t>(layer)];
            if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= cells.nz)
                return -1;
            return cells.id[(static_cast<std::size_t>(k) * ny + j) * nx + i];
        }
    };

    inline bool inside(const Layer& layer, double cx, double cy)
    { return std::abs(cx) < 0.5 * layer.x && std::abs(cy) < 0.5 * layer.y; }

    inline Mesh build_mesh(const Level& level)
    {
        Mesh mesh;
        mesh.x = graded_axis(0.5 * kChipX, 50.0, level.inplane);
        mesh.y = graded_axis(0.5 * kChipY, 50.0, level.inplane);
        mesh.nx = static_cast<int>(mesh.x.size()) - 1;
        mesh.ny = static_cast<int>(mesh.y.size()) - 1;

        auto cell_count = [&](const Level& lv, const Layer& layer) {
            const double thin = layer.z < 0.02 ? 0.02 : layer.z;
            const bool die = layer.temperature_dependent;
            const int nz = die ? lv.z_die : (thin < 0.05 ? lv.z_thin : std::max(1, lv.z_thin));
            return std::max(1, nz);
        };

        for (const Layer& layer : kStack) {
            Mesh::LayerCells cells;
            cells.nz = cell_count(level, layer);
            cells.dz = layer.z / cells.nz;
            cells.id.assign(static_cast<std::size_t>(mesh.nx) * mesh.ny * cells.nz, -1);
            for (int k = 0; k < cells.nz; ++k)
                for (int j = 0; j < mesh.ny; ++j) {
                    const double cy
                        = 0.5 * (mesh.y[static_cast<std::size_t>(j)] + mesh.y[static_cast<std::size_t>(j + 1)]);
                    for (int i = 0; i < mesh.nx; ++i) {
                        const double cx
                            = 0.5 * (mesh.x[static_cast<std::size_t>(i)] + mesh.x[static_cast<std::size_t>(i + 1)]);
                        if (!inside(layer, cx, cy))
                            continue;
                        auto& slot = cells.id[(static_cast<std::size_t>(k) * mesh.ny + j) * mesh.nx + i];
                        slot = mesh.n++;
                        mesh.cell_layer.push_back(static_cast<int>(&layer - kStack.data()));
                        const double vx = mesh.x[static_cast<std::size_t>(i + 1)] - mesh.x[static_cast<std::size_t>(i)];
                        const double vy = mesh.y[static_cast<std::size_t>(j + 1)] - mesh.y[static_cast<std::size_t>(j)];
                        mesh.volume.push_back(1e-9 * vx * vy * cells.dz); // mm^3 -> m^3
                    }
                }
            mesh.layers.push_back(std::move(cells));
        }
        return mesh;
    }

} // namespace mhs::hdim
