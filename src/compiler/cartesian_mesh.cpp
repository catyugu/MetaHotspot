#include "compiler/geometry_compiler.hpp"
#include <cmath>
#include <map>
#include <stdexcept>

namespace mhs::sim {
    // The only structured-index arithmetic in the compiled model pipeline.
    // This adapter consumes supplied axes; it does not choose/refine a mesh.
    void compile_cartesian_mesh(mhs::core::Model& model, const mhs::model::MeshSpec& input, double scale,
        const std::vector<ResolvedLayerGeometry>& layers)
    {
        using namespace mhs::core;
        std::array<std::vector<double>, 3> axes {input.x_vertices, input.y_vertices, input.z_vertices};
        std::array<Index, 3> n {};
        for (int a = 0; a < 3; ++a) {
            if (axes[a].size() < 2)
                throw std::invalid_argument("Cartesian mesh requires at least two vertices per axis");
            for (Index i = 0; i < axes[a].size(); ++i) {
                axes[a][i] *= scale;
                if (!std::isfinite(axes[a][i]) || (i && axes[a][i] <= axes[a][i - 1]))
                    throw std::invalid_argument("Cartesian vertices must be finite and strictly increasing");
            }
            n[a] = axes[a].size() - 1;
        }
        auto flatten = [&](Index x, Index y, Index z) { return (x * n[1] + y) * n[2] + z; };
        auto vertex = [&](Index x, Index y, Index z) { return (x * (n[1] + 1) + y) * (n[2] + 1) + z; };
        std::vector<Index> active(n[0] * n[1] * n[2], invalidIndex);
        model.output.cartesian_shape = {n[0] + 1, n[1] + 1, n[2] + 1};
        for (auto x : axes[0])
            for (auto y : axes[1])
                for (auto z : axes[2])
                    model.output.points.push_back({x, y, z});
        for (Index x = 0; x < n[0]; ++x)
            for (Index y = 0; y < n[1]; ++y)
                for (Index z = 0; z < n[2]; ++z) {
                    CellGeometry cell {
                        {axes[0][x], axes[1][y], axes[2][z]}, {axes[0][x + 1], axes[1][y + 1], axes[2][z + 1]}};
                    if (classify_point(layers, cell.center()).first == invalidIndex)
                        continue;
                    active[flatten(x, y, z)] = model.discrete.cells.size();
                    model.discrete.cells.push_back(cell);
                    model.output.cell_vertices.push_back({vertex(x, y, z), vertex(x + 1, y, z), vertex(x + 1, y + 1, z),
                        vertex(x, y + 1, z), vertex(x, y, z + 1), vertex(x + 1, y, z + 1), vertex(x + 1, y + 1, z + 1),
                        vertex(x, y + 1, z + 1)});
                }
        for (Index x = 0; x < n[0]; ++x)
            for (Index y = 0; y < n[1]; ++y)
                for (Index z = 0; z < n[2]; ++z) {
                    const auto c = active[flatten(x, y, z)];
                    if (c == invalidIndex)
                        continue;
                    for (int a = 0; a < 3; ++a)
                        for (int sign : {-1, 1}) {
                            std::array<Index, 3> v {x, y, z};
                            Index neighbor = invalidIndex;
                            if ((sign < 0 && v[a] > 0) || (sign > 0 && v[a] + 1 < n[a])) {
                                if (sign < 0)
                                    --v[a];
                                else
                                    ++v[a];
                                neighbor = active[flatten(v[0], v[1], v[2])];
                            }
                            if (neighbor != invalidIndex && neighbor < c)
                                continue;
                            const auto& cell = model.discrete.cells[c];
                            FacePatch f;
                            f.owner = {c};
                            f.neighbor = {neighbor};
                            f.axis = a;
                            f.sign = sign;
                            f.min = cell.min;
                            f.max = cell.max;
                            f.min[a] = f.max[a] = sign < 0 ? cell.min[a] : cell.max[a];
                            auto d = cell.size();
                            f.area = d[(a + 1) % 3] * d[(a + 2) % 3];
                            f.owner_distance = d[a] / 2;
                            f.neighbor_distance
                                = neighbor == invalidIndex ? 0 : model.discrete.cells[neighbor].size()[a] / 2;
                            model.discrete.faces.push_back(f);
                        }
                }
        model.discrete.build_incidence();
        model.discrete.validate();
        model.output.build_incidence();
    }
    void build_external_output_mesh(mhs::core::Model& model)
    {
        using namespace mhs::core;
        std::map<Vec3, Index> vertices;
        constexpr int corners[8][3]
            = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
        for (const auto& cell : model.discrete.cells) {
            std::array<Index, 8> ids {};
            for (Index i = 0; i < 8; ++i) {
                Vec3 p {};
                for (int a = 0; a < 3; ++a)
                    p[a] = corners[i][a] ? cell.max[a] : cell.min[a];
                auto [it, inserted] = vertices.emplace(p, model.output.points.size());
                if (inserted)
                    model.output.points.push_back(p);
                ids[i] = it->second;
            }
            model.output.cell_vertices.push_back(ids);
        }
        model.output.build_incidence();
    }
} // namespace mhs::sim
