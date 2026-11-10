#include "compiler/geometry_compiler.hpp"
#include "core/vector.hpp"
#include <cmath>
#include <stdexcept>

namespace mhs::compiler {
    // The only structured-index arithmetic in the compiled model pipeline.
    // This adapter consumes supplied axes; it does not choose/refine a mesh.
    void compile_cartesian_mesh(mhs::model::Model& model, const mhs::model::MeshSpec& input,
        const mhs::model::ModelDefinition& definition)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        std::array<mhs::core::Vector<double>, 3> axes {input.x_vertices, input.y_vertices, input.z_vertices};
        std::array<std::size_t, 3> n {};
        for (int a = 0; a < 3; ++a) {
            if (axes[a].size() < 2)
                throw std::invalid_argument("Cartesian axes require at least two vertices");
            for (std::size_t i = 0; i < axes[a].size(); ++i)
                if (!std::isfinite(axes[a][i]) || (i && axes[a][i] <= axes[a][i - 1]))
                    throw std::invalid_argument("Cartesian axes must be finite and strictly increasing");
            n[a] = axes[a].size() - 1;
        }
        auto flatten = [&](std::size_t x, std::size_t y, std::size_t z) { return (x * n[1] + y) * n[2] + z; };
        mhs::core::Vector<CellId> active(n[0] * n[1] * n[2], invalid_id);
        for (std::size_t x = 0; x < n[0]; ++x)
            for (std::size_t y = 0; y < n[1]; ++y)
                for (std::size_t z = 0; z < n[2]; ++z) {
                    CellGeometry cell {
                        {axes[0][x], axes[1][y], axes[2][z]}, {axes[0][x + 1], axes[1][y + 1], axes[2][z + 1]}};
                    if (!classify_point(definition, cell.center()))
                        continue;
                    const auto cell_id = static_cast<CellId>(model.discrete.cells.size());
                    active[flatten(x, y, z)] = cell_id;
                    model.discrete.cells.push_back(cell);
                    for (auto& probe : model.point_probes)
                        if (probe.point[0] >= cell.min[0] && probe.point[0] <= cell.max[0]
                            && probe.point[1] >= cell.min[1] && probe.point[1] <= cell.max[1]
                            && probe.point[2] >= cell.min[2] && probe.point[2] <= cell.max[2])
                            probe.cell = cell_id;
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
                            Facet f;
                            f.minus = sign > 0 ? c : neighbor;
                            f.plus = sign > 0 ? neighbor : c;
                            f.axis = static_cast<Axis>(a);
                            f.coordinate = sign > 0 ? cell.max[a] : cell.min[a];
                            const int t = a == 0 ? 1 : 0, u = a == 2 ? 1 : 2;
                            f.bounds = {cell.min[t], cell.max[t], cell.min[u], cell.max[u]};
                            model.discrete.facets.push_back(f);
                        }
                }
    }
} // namespace mhs::compiler
