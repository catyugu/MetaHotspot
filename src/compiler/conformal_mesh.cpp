#include "compiler/geometry_compiler.hpp"
#include "compiler/volume_rasterizer.hpp"
#include "core/vector.hpp"
#include "model/constants.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mhs::compiler {
    // Effective region boundaries inside the mesh must lie on input grid planes.
    CompiledGeometry compile_conformal_mesh(const mhs::model::MeshSpec& input, std::span<const VolumeGeometry> volumes)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        CompiledGeometry result;
        auto& mesh = result.mesh;
        const std::array<std::span<const double>, 3> axes {input.x_vertices, input.y_vertices, input.z_vertices};
        std::array<std::size_t, 3> n {};
        for (int a = 0; a < 3; ++a) {
            if (axes[a].size() < 2)
                throw std::invalid_argument("Cartesian axes require at least two vertices");
            for (std::size_t i = 0; i < axes[a].size(); ++i)
                if (!std::isfinite(axes[a][i]) || (i && axes[a][i] <= axes[a][i - 1]))
                    throw std::invalid_argument("Cartesian axes must be finite and strictly increasing");
            n[a] = axes[a].size() - 1;
        }
        for (const auto& volume : volumes)
            for (const auto& box : volume)
                for (int a = 0; a < 3; ++a) {
                    if (!std::isfinite(box.min[a]) || !std::isfinite(box.max[a]) || box.min[a] >= box.max[a])
                        throw std::invalid_argument("box bounds must be finite and increasing");
                }
        auto flatten = [&](std::size_t x, std::size_t y, std::size_t z) { return (x * n[1] + y) * n[2] + z; };
        auto active = rasterize_volumes(axes, volumes);
        for (std::size_t x = 0; x < n[0]; ++x)
            for (std::size_t y = 0; y < n[1]; ++y)
                for (std::size_t z = 0; z < n[2]; ++z) {
                    CellGeometry cell {
                        {axes[0][x], axes[1][y], axes[2][z]}, {axes[0][x + 1], axes[1][y + 1], axes[2][z + 1]}};
                    const auto owner = active[flatten(x, y, z)];
                    if (owner == invalid_id)
                        continue;
                    if (mesh.cells.size() >= invalid_id)
                        throw std::length_error("too many Cartesian cells");
                    const auto cell_id = static_cast<CellId>(mesh.cells.size());
                    active[flatten(x, y, z)] = cell_id;
                    mesh.cells.push_back(cell);
                    result.cell_regions.push_back(owner);
                }
        for (std::size_t x = 0; x < n[0]; ++x)
            for (std::size_t y = 0; y < n[1]; ++y)
                for (std::size_t z = 0; z < n[2]; ++z) {
                    const auto c = active[flatten(x, y, z)];
                    if (c == mhs::core::invalid_id)
                        continue;
                    for (int a = 0; a < 3; ++a)
                        for (int sign : {-1, 1}) {
                            std::array<std::size_t, 3> v {x, y, z};
                            CellId neighbor = mhs::core::invalid_id;
                            if ((sign < 0 && v[a] > 0) || (sign > 0 && v[a] + 1 < n[a])) {
                                if (sign < 0)
                                    --v[a];
                                else
                                    ++v[a];
                                neighbor = active[flatten(v[0], v[1], v[2])];
                            }
                            if ((neighbor != mhs::core::invalid_id) && neighbor < c)
                                continue;
                            const auto& cell = mesh.cells[c];
                            Facet f;
                            f.minus = sign > 0 ? c : neighbor;
                            f.plus = sign > 0 ? neighbor : c;
                            f.axis = static_cast<Axis>(a);
                            f.coordinate = sign > 0 ? cell.max[a] : cell.min[a];
                            const int t = a == 0 ? 1 : 0, u = a == 2 ? 1 : 2;
                            f.bounds = {cell.min[t], cell.max[t], cell.min[u], cell.max[u]};
                            mesh.facets.push_back(f);
                        }
                }
        return result;
    }

    mhs::model::CellId locate_conformal_cell(
        const mhs::model::DiscreteMesh& mesh, const mhs::model::MeshSpec& input, const mhs::model::Vec3& point)
    {
        using namespace mhs::model;
        const std::array<std::span<const double>, 3> axes {input.x_vertices, input.y_vertices, input.z_vertices};
        std::array<std::array<double, 2>, 3> candidates {};
        std::array<int, 3> counts {};
        for (int a = 0; a < 3; ++a) {
            const auto& vertices = axes[a];
            if (!std::isfinite(point[a]) || point[a] < vertices.front() || point[a] > vertices.back())
                return mhs::core::invalid_id;
            const auto it = std::lower_bound(vertices.begin(), vertices.end(), point[a]);
            if (it != vertices.begin())
                candidates[a][counts[a]++] = *(it - 1);
            if (it != vertices.end() - 1 && *it == point[a])
                candidates[a][counts[a]++] = *it;
        }
        CellId found = mhs::core::invalid_id;
        for (int x = 0; x < counts[0]; ++x)
            for (int y = 0; y < counts[1]; ++y)
                for (int z = 0; z < counts[2]; ++z) {
                    const Vec3 minimum {candidates[0][x], candidates[1][y], candidates[2][z]};
                    const auto it = std::lower_bound(mesh.cells.begin(), mesh.cells.end(), minimum,
                        [](const CellGeometry& cell, const Vec3& p) { return cell.min < p; });
                    if (it != mesh.cells.end() && it->min == minimum)
                        found = static_cast<CellId>(it - mesh.cells.begin());
                }
        return found;
    }
} // namespace mhs::compiler
