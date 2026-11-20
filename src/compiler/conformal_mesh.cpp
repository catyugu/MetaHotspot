#include "compiler/geometry_compiler.hpp"
#include "core/vector.hpp"
#include "model/constants.hpp"
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace mhs::compiler {
    namespace {
        std::optional<std::size_t> classify_point(
            const mhs::core::Vector<mhs::model::VolumeSpec>& volumes, const mhs::model::Vec3& p)
        {
            for (std::size_t i = volumes.size(); i-- > 0;) {
                bool inside = false;
                for (const auto& box : volumes[i].geometry) {
                    bool contains = true;
                    for (int a = 0; a < 3; ++a)
                        contains = contains && p[a] >= box.min[a] - mhs::model::geometry_eps
                            && p[a] <= box.max[a] + mhs::model::geometry_eps;
                    if (contains)
                        inside = box.operation == mhs::model::GeometryOperation::Add;
                }
                if (inside)
                    return i;
            }
            return std::nullopt;
        }
    }

    // Conformal Cartesian cells: volume box faces inside the mesh must lie on input grid planes.
    void compile_conformal_mesh(mhs::model::Fom& model, const mhs::model::MeshSpec& input,
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        auto& mesh = model.mesh;
        auto& axes = mesh.vertices;
        axes = {input.x_vertices, input.y_vertices, input.z_vertices};
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
            for (const auto& box : volume.geometry)
                for (int a = 0; a < 3; ++a) {
                    const auto& vertices = axes[a];
                    for (double boundary : {box.min[a], box.max[a]}) {
                        if (boundary <= vertices.front() || boundary >= vertices.back())
                            continue;
                        const auto it = std::lower_bound(vertices.begin(), vertices.end(), boundary);
                        if ((it == vertices.end() || std::abs(*it - boundary) > geometry_eps)
                            && (it == vertices.begin() || std::abs(*(it - 1) - boundary) > geometry_eps))
                            throw std::invalid_argument("volume geometry must align with Cartesian mesh vertices");
                    }
                }
        std::size_t grid_size = 1;
        for (const auto extent : n) {
            if (extent > mesh.grid_to_cell.max_size() / grid_size)
                throw std::length_error("Cartesian mesh is too large");
            grid_size *= extent;
        }
        mesh.grid_to_cell.assign(grid_size, invalid_id);
        for (std::size_t x = 0; x < n[0]; ++x)
            for (std::size_t y = 0; y < n[1]; ++y)
                for (std::size_t z = 0; z < n[2]; ++z) {
                    CellGeometry cell {
                        {axes[0][x], axes[1][y], axes[2][z]}, {axes[0][x + 1], axes[1][y + 1], axes[2][z + 1]}};
                    const auto center = cell.center();
                    const auto owner = classify_point(volumes, center);
                    if (!owner)
                        continue;
                    if (mesh.cell_count() == invalid_id)
                        throw std::length_error("too many active Cartesian cells");
                    const auto cell_id = static_cast<CellId>(mesh.cell_count());
                    const auto index = mesh.grid_index(x, y, z);
                    mesh.grid_to_cell[index] = cell_id;
                    mesh.cell_to_grid.push_back(index);
                    const auto& volume = volumes[*owner];
                    model.cell_material.push_back(volume.material);
                    if (volume.source != invalid_id)
                        model.source_terms.push_back({cell_id, volume.source});
                    for (auto& probe : model.point_probes)
                        if (probe.point[0] >= cell.min[0] && probe.point[0] <= cell.max[0]
                            && probe.point[1] >= cell.min[1] && probe.point[1] <= cell.max[1]
                            && probe.point[2] >= cell.min[2] && probe.point[2] <= cell.max[2])
                            probe.cell = cell_id;
                }
        for (CellId c = 0; c < mesh.cell_count(); ++c) {
            const auto cell = mesh.cell(c);
            for (std::size_t a = 0; a < 3; ++a)
                for (int sign : {-1, 1}) {
                    if (mesh.neighbor(c, a, sign) != invalid_id)
                        continue;
                    const auto axis = static_cast<Axis>(a);
                    const auto [u, v] = tangential_axes(axis);
                    mesh.exterior_facets.push_back(
                        {{axis, static_cast<std::int8_t>(sign), sign > 0 ? cell.max[a] : cell.min[a],
                             {cell.min[u], cell.max[u], cell.min[v], cell.max[v]}},
                            c, invalid_id});
                }
        }
    }
} // namespace mhs::compiler
