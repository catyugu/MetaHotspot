#include "compiler/geometry_compiler.hpp"
#include "core/vector.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mhs::compiler {
    namespace {
        void locate_point_probes(mhs::model::Model& model, const std::array<mhs::core::Vector<double>, 3>& axes,
            const std::array<std::size_t, 3>& n, const mhs::core::Vector<mhs::model::CellId>& active)
        {
            using mhs::core::invalid_id;
            auto flatten = [&](std::size_t x, std::size_t y, std::size_t z) { return (x * n[1] + y) * n[2] + z; };
            // Query supplied axes after all active cells are known. At shared grid
            // planes, preserve the former last-containing-active-cell convention.
            for (auto& probe : model.point_probes) {
                probe.cell = invalid_id;
                std::array<std::size_t, 3> lo {}, hi {};
                bool contained = true;
                for (int a = 0; a < 3; ++a) {
                    const double p = probe.point[a];
                    if (!std::isfinite(p) || p < axes[a].front() || p > axes[a].back()) {
                        contained = false;
                        break;
                    }
                    const auto it = std::lower_bound(axes[a].begin(), axes[a].end(), p);
                    const auto vertex = static_cast<std::size_t>(it - axes[a].begin());
                    lo[a] = vertex > 0 ? vertex - 1 : 0;
                    hi[a] = *it == p ? std::min(vertex, n[a] - 1) : lo[a];
                }
                if (!contained)
                    continue;
                for (std::size_t x = hi[0] + 1; x-- > lo[0] && probe.cell == invalid_id;)
                    for (std::size_t y = hi[1] + 1; y-- > lo[1] && probe.cell == invalid_id;)
                        for (std::size_t z = hi[2] + 1; z-- > lo[2] && probe.cell == invalid_id;)
                            probe.cell = active[flatten(x, y, z)];
            }
        }
    } // namespace

    // The only structured-index arithmetic in the compiled model pipeline.
    // This adapter consumes supplied axes; it does not choose/refine a mesh.
    void compile_cartesian_mesh(mhs::model::Model& model, const mhs::model::MeshSpec& input,
        const mhs::core::Vector<mhs::model::RegionSpec>& regions,
        const mhs::core::Vector<mhs::model::MaterialAssignment>& materials,
        const mhs::core::Vector<mhs::model::SourceAssignment>& sources)
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
                    if (!project_cell_regions(model, cell, regions, materials, sources))
                        continue;
                    const auto cell_id = static_cast<CellId>(model.discrete.cells.size());
                    active[flatten(x, y, z)] = cell_id;
                    model.discrete.cells.push_back(cell);
                }
        locate_point_probes(model, axes, n, active);
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
