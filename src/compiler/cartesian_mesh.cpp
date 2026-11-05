#include "compiler/geometry_compiler.hpp"
#include "core/vector.hpp"
#include <map>
#include <stdexcept>

namespace mhs::compiler {
    // The only structured-index arithmetic in the compiled model pipeline.
    // This adapter consumes supplied axes; it does not choose/refine a mesh.
    void compile_cartesian_mesh(mhs::model::Model& model, const mhs::model::MeshSpec& input, double scale,
        const mhs::core::Vector<ResolvedLayerGeometry>& layers)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        std::array<mhs::core::Vector<double>, 3> axes {input.x_vertices, input.y_vertices, input.z_vertices};
        std::array<std::size_t, 3> n {};
        for (int a = 0; a < 3; ++a) {
            if (axes[a].size() < 2)
                throw std::invalid_argument("Cartesian axes require at least two vertices");
            for (auto& coordinate : axes[a])
                coordinate *= scale;
            n[a] = axes[a].size() - 1;
        }
        auto flatten = [&](std::size_t x, std::size_t y, std::size_t z) { return (x * n[1] + y) * n[2] + z; };
        auto vertex = [&](std::size_t x, std::size_t y, std::size_t z) {
            return static_cast<PointId>((x * (n[1] + 1) + y) * (n[2] + 1) + z);
        };
        mhs::core::Vector<CellId> active(n[0] * n[1] * n[2], invalid_id);
        model.output.cartesian_shape = {n[0] + 1, n[1] + 1, n[2] + 1};
        for (auto x : axes[0])
            for (auto y : axes[1])
                for (auto z : axes[2])
                    model.output.points.push_back({x, y, z});
        for (std::size_t x = 0; x < n[0]; ++x)
            for (std::size_t y = 0; y < n[1]; ++y)
                for (std::size_t z = 0; z < n[2]; ++z) {
                    CellGeometry cell {
                        {axes[0][x], axes[1][y], axes[2][z]}, {axes[0][x + 1], axes[1][y + 1], axes[2][z + 1]}};
                    if (!classify_point(layers, cell.center()))
                        continue;
                    active[flatten(x, y, z)] = static_cast<CellId>(model.discrete.cells.size());
                    model.discrete.cells.push_back(cell);
                    model.output.cell_vertices.push_back({vertex(x, y, z), vertex(x + 1, y, z), vertex(x + 1, y + 1, z),
                        vertex(x, y + 1, z), vertex(x, y, z + 1), vertex(x + 1, y, z + 1), vertex(x + 1, y + 1, z + 1),
                        vertex(x, y + 1, z + 1)});
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
                            const auto& cell = model.discrete.cells[c];
                            FacePatch f;
                            f.owner = c;
                            f.neighbor = neighbor;
                            f.axis = static_cast<Axis>(a);
                            f.coordinate = sign < 0 ? cell.min[a] : cell.max[a];
                            const int t = a == 0 ? 1 : 0, u = a == 2 ? 1 : 2;
                            f.bounds = {cell.min[t], cell.max[t], cell.min[u], cell.max[u]};
                            model.discrete.faces.push_back(f);
                        }
                }
        prepare_discrete_mesh(model.discrete);
    }
    void build_external_output_mesh(mhs::model::Model& model)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        std::map<Vec3, PointId> vertices;
        constexpr int corners[8][3]
            = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
        for (const auto& cell : model.discrete.cells) {
            std::array<PointId, 8> ids {};
            for (std::size_t i = 0; i < 8; ++i) {
                Vec3 p {};
                for (int a = 0; a < 3; ++a)
                    p[a] = corners[i][a] ? cell.max[a] : cell.min[a];
                auto [it, inserted] = vertices.try_emplace(p, invalid_id);
                if (inserted) {
                    it->second = static_cast<PointId>(model.output.points.size());
                    model.output.points.push_back(p);
                }
                ids[i] = it->second;
            }
            model.output.cell_vertices.push_back(ids);
        }
    }
} // namespace mhs::compiler
