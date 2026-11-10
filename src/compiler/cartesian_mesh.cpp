#include "compiler/geometry_compiler.hpp"
#include "core/vector.hpp"
#include "model/constants.hpp"
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace mhs::compiler {
    namespace {
        constexpr double EPS = mhs::model::geometry_eps;

        std::optional<std::size_t> classify_body(
            const mhs::core::Vector<mhs::model::BodySpec>& bodies, const mhs::model::Vec3& point)
        {
            for (std::size_t i = bodies.size(); i-- > 0;)
                if (geometry_contains(bodies[i].geometry, point))
                    return i;
            return std::nullopt;
        }

        bool append_cell(mhs::model::Model& model, const mhs::core::Vector<mhs::model::BodySpec>& bodies,
            const mhs::model::CellGeometry& cell)
        {
            using namespace mhs::core;
            using namespace mhs::model;
            const auto center = cell.center();
            std::optional<std::size_t> owner;
            std::array<Vector<double>, 3> cuts;

            // Classify the material and collect only geometry planes that cut the cell.
            for (std::size_t i = 0; i < bodies.size(); ++i) {
                bool inside = false;
                for (const auto& box : bodies[i].geometry.operations) {
                    bool contains = true, overlaps = true;
                    for (int a = 0; a < 3; ++a) {
                        contains = contains && center[a] >= box.min[a] - EPS && center[a] <= box.max[a] + EPS;
                        overlaps = overlaps && box.max[a] > cell.min[a] && box.min[a] < cell.max[a];
                    }
                    if (contains)
                        inside = box.operation == GeometryOperation::Add;
                    if (overlaps)
                        for (int a = 0; a < 3; ++a)
                            for (const double v : {box.min[a], box.max[a]})
                                if (v > cell.min[a] + EPS && v < cell.max[a] - EPS)
                                    cuts[a].push_back(v);
                }
                if (inside)
                    owner = i;
            }
            if (!owner)
                return false;

            const auto material = bodies[*owner].material;
            if (!std::all_of(cuts.begin(), cuts.end(), [](const auto& axis) { return axis.empty(); })) {
                for (int a = 0; a < 3; ++a) {
                    cuts[a].push_back(cell.min[a]);
                    cuts[a].push_back(cell.max[a]);
                    std::sort(cuts[a].begin(), cuts[a].end());
                    cuts[a].erase(std::unique(cuts[a].begin(), cuts[a].end()), cuts[a].end());
                }
                for (std::size_t x = 1; x < cuts[0].size(); ++x)
                    for (std::size_t y = 1; y < cuts[1].size(); ++y)
                        for (std::size_t z = 1; z < cuts[2].size(); ++z) {
                            const Vec3 p {(cuts[0][x - 1] + cuts[0][x]) / 2,
                                (cuts[1][y - 1] + cuts[1][y]) / 2, (cuts[2][z - 1] + cuts[2][z]) / 2};
                            const auto location = classify_body(bodies, p);
                            if (location && bodies[*location].material != material)
                                throw std::invalid_argument(
                                    "material interfaces must align with Cartesian cell boundaries");
                        }
            }

            model.discrete.cells.push_back(cell);
            model.cell_material.push_back(material);
            return true;
        }
    } // namespace

    // The only structured-index arithmetic in the compiled model pipeline.
    // This adapter consumes supplied axes; it does not choose/refine a mesh.
    void compile_cartesian_mesh(mhs::model::Model& model, const mhs::model::MeshSpec& input,
        const mhs::core::Vector<mhs::model::BodySpec>& bodies)
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
                    if (!append_cell(model, bodies, cell))
                        continue;
                    const auto cell_id = static_cast<CellId>(model.discrete.cells.size() - 1);
                    active[flatten(x, y, z)] = cell_id;
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
